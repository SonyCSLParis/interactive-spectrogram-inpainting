from typing import Mapping
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
import lmdb
from tqdm import tqdm

from utils.datasets.lmdb_dataset import CodeRow, LMDBDataset
from vqvae import VQVAE
from utils.misc import expand_path, get_spectrograms_helper

from pytorch_nsynth import NSynth
from GANsynth_pytorch.loader import WavToSpectrogramDataLoader

# HOP_LENGTH = 512
# N_FFT = 2048
# FS_HZ = 16000


def extract(lmdb_env, loader: WavToSpectrogramDataLoader,
            model: VQVAE,
            device: str,
            label_encoders: Mapping[str, LabelEncoder] = {}):
    index = 0

    parallel_model = nn.DataParallel(model)

    with lmdb_env.begin(write=True) as txn:
        pbar_loader = tqdm(loader)

        # store the label encoders along with the database
        # this allows future conversions
        txn.put('label_encoders'.encode('utf-8'), pickle.dumps(label_encoders))

        attribute_names = label_encoders.keys()

        for (sample_batch, *categorical_attributes_batch,
                attributes_batch) in pbar_loader:
            sample_batch = sample_batch.to(device)
            sample_names = attributes_batch['note_str']

            *_, id_t, id_b = parallel_model(sample_batch)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for top, bottom, *attributes, sample_name in zip(
                    id_t, id_b, *categorical_attributes_batch, sample_names):
                row = CodeRow(top=top, bottom=bottom,
                              attributes=dict(zip(attribute_names,
                                                  attributes)),
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
    parser.add_argument('--categorical_fields', type=str, nargs='*',
                        default=['instrument_family_str', 'pitch'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for the Dataloaders')
    parser.add_argument('--dataset_audio_directory_paths', type=str,
                        nargs='+')
    parser.add_argument('--named_dataset_json_data_paths',
                        action=StoreDictKeyPair)
    parser.add_argument('--vqvae_weights_path', type=str, required=True)
    parser.add_argument('--vqvae_model_parameters_path', type=str,
                        required=True)
    parser.add_argument('--vqvae_training_parameters_path', type=str,
                        required=True)
    parser.add_argument('--size', type=int)
    parser.add_argument('--disable_database_creation', action='store_true')

    args = parser.parse_args()

    MAIN_DIR = pathlib.Path(args.main_output_dir)

    VQVAE_WEIGHTS_PATH = expand_path(args.vqvae_weights_path)
    VQVAE_MODEL_PARAMETERS_PATH = expand_path(args.vqvae_model_parameters_path)
    VQVAE_TRAINING_PARAMETERS_PATH = expand_path(
        args.vqvae_training_parameters_path)
    assert (VQVAE_WEIGHTS_PATH.is_file()
            and VQVAE_MODEL_PARAMETERS_PATH.is_file()
            and VQVAE_MODEL_PARAMETERS_PATH.is_file())
    # folder containing the vqvae weights is the ID
    vqvae_id = VQVAE_WEIGHTS_PATH.parts[-2]
    vqvae_model_filename = VQVAE_WEIGHTS_PATH.stem

    OUTPUT_DIR = MAIN_DIR / f'vqvae-{vqvae_id}-weights-{vqvae_model_filename}/'

    if not args.disable_database_creation:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    audio_directory_paths = [expand_path(path)
                             for path in args.dataset_audio_directory_paths]

    named_dataset_json_data_paths = {
        dataset_name: expand_path(json_data_path)
        for dataset_name, json_data_path
        in args.named_dataset_json_data_paths.items()
    }

    assert (len(set(named_dataset_json_data_paths.keys()))
            == len(named_dataset_json_data_paths.keys())), (
                "Use unique names for all datasets"
                "otherwise the outputs will overwrite one another"
                )

    vqvae = VQVAE.from_parameters_and_weights(
        VQVAE_MODEL_PARAMETERS_PATH,
        VQVAE_WEIGHTS_PATH)
    vqvae.to(device)
    vqvae.eval()

    # retrieve n_fft, hop length, window length parameters...
    with open(VQVAE_TRAINING_PARAMETERS_PATH, 'r') as f:
        vqvae_training_parameters = json.load(f)
    spectrograms_helper = get_spectrograms_helper(
        device=device, **vqvae_training_parameters)

    for dataset_name, json_data_path in named_dataset_json_data_paths.items():
        assert json_data_path.is_file()
        valid_pitch_range = vqvae_training_parameters['valid_pitch_range']
        transform = vqvae.output_transform

        nsynth_dataset_with_samples_names = NSynth(
            audio_directory_paths=audio_directory_paths,
            json_data_path=json_data_path,
            valid_pitch_range=valid_pitch_range,
            categorical_field_list=args.categorical_fields,
            squeeze_mono_channel=True
        )

        # converts wavforms to spectrograms on-the-fly on GPU
        loader = WavToSpectrogramDataLoader(
            nsynth_dataset_with_samples_names,
            spectrograms_helper=spectrograms_helper,
            batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=False,
            transform=transform,
        )

        label_encoders = nsynth_dataset_with_samples_names.label_encoders

        in_channel = 2

        # TODO(theis): compute appropriate size for the map,
        # for now it's a generous overshoot
        map_size = 100 * 1024 * 1024 * 1024

        lmdb_path = expand_path(OUTPUT_DIR / dataset_name)

        if not args.disable_database_creation:
            os.makedirs(lmdb_path, exist_ok=False)
            # store command-line parameters
            with open(OUTPUT_DIR / 'command_line_parameters.json', 'w') as f:
                json.dump(args.__dict__, f, indent=4)

            env = lmdb.open(str(lmdb_path), map_size=map_size)
            with torch.no_grad():
                print("Start extraction for dataset", dataset_name)
                extract(env, loader, vqvae, device,
                        label_encoders=label_encoders)

        print("Start sanity-check for dataset", dataset_name)
        # check extracted codes by saving to disk the audio for a batch
        # of re-synthesized codemaps
        codes_dataset = LMDBDataset(
            str(lmdb_path),
            classes_for_conditioning=args.categorical_fields)
        codes_loader = DataLoader(codes_dataset, batch_size=8,
                                  shuffle=True)
        with torch.no_grad():
            codes_top_sample, codes_bottom_sample, attributes = (
                next(iter(codes_loader)))
            decoded_sample = vqvae.decode_code(
                codes_top_sample.to(device),
                codes_bottom_sample.to(device))

            def make_audio(mag_and_IF_batch: torch.Tensor) -> np.ndarray:
                audio_batch = spectrograms_helper.to_audio(
                    mag_and_IF_batch)
                audio_mono_concatenated = (audio_batch
                                           .flatten().cpu().numpy())
                return audio_mono_concatenated

            audio_sample_path = os.path.join(
                lmdb_path,
                'vqvae_codes_extraction_samples.wav')
            soundfile.write(audio_sample_path,
                            make_audio(decoded_sample),
                            samplerate=vqvae_training_parameters['fs_hz'])
            print("Stored sanity-check decoding of stored codes at",
                  audio_sample_path)
