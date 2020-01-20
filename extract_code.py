from typing import Optional, Iterable, Mapping, Union, Sequence
import argparse
import pickle
import json
import pathlib
import os
import soundfile
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow, LMDBDataset
from vqvae import InferenceVQVAE, VQVAE

from GANsynth_pytorch.pytorch_nsynth_lib.nsynth import (
    NSynth, WavToSpectrogramDataLoader)

HOP_LENGTH = 512
N_FFT = 2048
FS_HZ = 16000


def extract(lmdb_env, loader, model, device, dataset: str,
            label_encoders: Mapping[str, LabelEncoder] = {}):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar_loader = tqdm(loader)

        # store the label encoders along with the database to allow future conversions
        txn.put('label_encoders'.encode('utf-8'), pickle.dumps(label_encoders))

        attribute_names = label_encoders.keys()

        for (sample_batch, *categorical_attributes_batch,
                attributes_batch) in pbar_loader:
            sample_batch = sample_batch.to(device)
            if dataset == 'nsynth':
                sample_names = attributes_batch['note_str']
            elif dataset == 'imagenet':
                sample_names = attributes_batch
                categorical_attributes_batch = [attributes_batch]

            *_, id_t, id_b = model(sample_batch)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for top, bottom, *attributes, sample_name in zip(
                    id_t, id_b, *categorical_attributes_batch, sample_names):
                row = CodeRow(top=top, bottom=bottom,
                              attributes=dict(zip(attribute_names, attributes)),
                              filename=sample_name)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar_loader.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    class StoreDictKeyPair(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            my_dict = {}
            for kv in values.split(","):
                k, v = kv.split("=")
                my_dict[str(k)] = str(v)
            setattr(namespace, self.dest, my_dict)

    parser = argparse.ArgumentParser()
    parser.add_argument('--main_output_dir', type=str, required=True)
    parser.add_argument('--checking_samples_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for the Dataloaders')
    parser.add_argument('--dataset', type=str, choices=['nsynth', 'imagenet'],
                        required=True)
    parser.add_argument('--named_dataset_paths', action=StoreDictKeyPair)
    parser.add_argument('--model_weights_path', type=str, required=True)
    parser.add_argument('--model_parameters_path', type=str, required=True)
    parser.add_argument('--command_line_parameters_path', type=str,
                        required=False)
    parser.add_argument('--size', type=int)
    parser.add_argument('--disable_database_creation', action='store_true')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])

    args = parser.parse_args()

    MAIN_DIR = pathlib.Path(args.main_output_dir)

    VQVAE_MODEL_WEIGHTS_PATH = (
        pathlib.Path(args.model_weights_path).expanduser().absolute())
    VQVAE_MODEL_PARAMETERS_PATH = (
        pathlib.Path(args.model_parameters_path).expanduser().absolute())
    assert (VQVAE_MODEL_WEIGHTS_PATH.is_file()
            and VQVAE_MODEL_PARAMETERS_PATH.is_file())
    # folder containing the vqvae weights is the ID
    vqvae_id = VQVAE_MODEL_WEIGHTS_PATH.parts[-2]
    vqvae_model_filename = VQVAE_MODEL_WEIGHTS_PATH.stem

    OUTPUT_DIR = MAIN_DIR / f'vqvae-{vqvae_id}-weights-{vqvae_model_filename}/'

    if not args.disable_database_creation:
        os.makedirs(OUTPUT_DIR, exist_ok=False)
        # store command-line parameters
        with open(OUTPUT_DIR / 'command_line_parameters.json', 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    device = args.device

    named_dataset_paths = {
        dataset_name: pathlib.Path(dataset_path).expanduser().absolute()
        for dataset_name, dataset_path in args.named_dataset_paths.items()
    }

    assert (len(set(named_dataset_paths.keys()))
            == len(named_dataset_paths.keys())), (
                "Use unique names for all datasets"
                "otherwise the outputs will overwrite one another"
                )

    for dataset_name, dataset_path in named_dataset_paths.items():
        assert dataset_path.is_dir()
        if args.dataset == 'nsynth':
            valid_pitch_range = [24, 84]

            transform = vqvae.output_transform

            nsynth_dataset_with_samples_names = NSynth(
                root=str(dataset_path),
                valid_pitch_range=valid_pitch_range,
                categorical_field_list=['instrument_family_str', 'pitch'],
                squeeze_mono_channel=True)

            # converts wavforms to spectrograms on-the-fly on GPU
            loader = WavToSpectrogramDataLoader(
                nsynth_dataset_with_samples_names,
                batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=False,
                pin_memory=True,
                device=device, n_fft=N_FFT, hop_length=HOP_LENGTH,
                transform=transform,
            )

            label_encoders = nsynth_dataset_with_samples_names.label_encoders

            in_channel = 2
        elif args.dataset == 'imagenet':
            def make_resize_transform(target_size: Union[int, Sequence[int]],
                                      normalize: bool):
                transformations = [
                    transforms.Resize(target_size),
                    transforms.CenterCrop(target_size),
                    transforms.ToTensor()
                ]
                if normalize:
                    transformations.append(transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5]))
                return transforms.Compose(transformations)

            # retrieve size and normalization for the training process of the loaded model
            # TODO(theis:maybe): store those details within the model itself?
            VQVAE_COMMAND_LINE_PARAMETERS_PATH = pathlib.Path(
                args.command_line_parameters_path)
            with open(VQVAE_COMMAND_LINE_PARAMETERS_PATH, 'r') as f:
                vqvae_command_line_parameters = json.load(f)

            transform = make_resize_transform(
                vqvae_command_line_parameters['size'],
                vqvae_command_line_parameters['normalize_input_images'])
            dataset = datasets.ImageFolder(dataset_path,
                                           transform=transform)
            loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
            dataloader_for_gansynth_normalization = None
            normalizer_statistics = None

            class_to_index = dataset.class_to_idx

            # def invert_mapping(mapping: Mapping[any, any]) -> Mapping[any, any]:
            #     return {v: k for k, v in mapping.items()}
            # index_to_class = invert_mapping(class_to_index)
            classes_label_encoder = LabelEncoder()
            classes_label_encoder.classes_ = [
                str(key) for key in class_to_index.keys()]
            label_encoders = {'class': classes_label_encoder}
            in_channel = 3

        vqvae = VQVAE.from_parameters_and_weights(
            VQVAE_MODEL_PARAMETERS_PATH,
            VQVAE_MODEL_WEIGHTS_PATH,
            device=device)

        model = nn.DataParallel(vqvae)
        model = model.to(device)
        model.eval()

        inference_vqvae = InferenceVQVAE(vqvae, device=device,
                                         hop_length=HOP_LENGTH,
                                         n_fft=N_FFT)

        # TODO(theis): compute appropriate size for the map
        map_size = 100 * 1024 * 1024 * 1024

        lmdb_path = (OUTPUT_DIR / dataset_name).absolute()

        if not args.disable_database_creation:
            env = lmdb.open(str(lmdb_path), map_size=map_size)
            with torch.no_grad():
                extract(env, loader, model, device, args.dataset,
                        label_encoders=label_encoders)

        if args.checking_samples_dir:
            # check extracted codes
            codes_dataset = LMDBDataset(str(lmdb_path))
            codes_loader = DataLoader(codes_dataset, batch_size=8,
                                      shuffle=True)
            with torch.no_grad():
                if args.dataset == 'nsynth':
                    codes_top_sample, codes_bottom_sample, instrument_families, pitches = (
                        next(iter(codes_loader)))
                elif args.dataset == 'imagenet':
                    codes_top_sample, codes_bottom_sample, image_class = (
                        next(iter(codes_loader)))
                decoded_sample = vqvae.decode_code(codes_top_sample.to(device),
                                                   codes_bottom_sample.to(device))

                if args.dataset == 'nsynth':
                    def make_audio(mag_and_IF_batch: torch.Tensor) -> np.ndarray:
                        audio_batch = inference_vqvae.mag_and_IF_to_audio(
                            mag_and_IF_batch, use_mel_frequency=True)
                        audio_mono_concatenated = audio_batch.flatten().cpu().numpy()
                        return audio_mono_concatenated

                    os.makedirs(args.checking_samples_dir, exist_ok=True)

                    audio_sample_path = os.path.join(
                        args.checking_samples_dir,
                        f'vqvae_codes_extraction_samples.wav')
                    soundfile.write(audio_sample_path, make_audio(decoded_sample),
                                    samplerate=FS_HZ)
                elif args.dataset == 'imagenet':
                    with torch.no_grad():
                        sanity_check_samples_path = os.path.join(
                            args.checking_samples_dir,
                            f'vqvae_codes_extraction_samples.png')
                        utils.save_image(
                            decoded_sample,
                            sanity_check_samples_path
                        )
