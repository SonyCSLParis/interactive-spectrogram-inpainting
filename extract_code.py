from typing import Any, Mapping, Optional
import argparse
import pickle
import json
import pathlib
import os

import torchvision
import soundfile
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import lmdb
from tqdm import tqdm

from pytorch_nsynth import NSynth
from GANsynth_pytorch.loader import (WavToSpectrogramDataLoader,
                                     make_masked_phase_transform)

from interactive_spectrogram_inpainting.vqvae import encoder_decoder
from interactive_spectrogram_inpainting.utils.datasets.lmdb_dataset import (
    CodeRow, LMDBDataset)
from interactive_spectrogram_inpainting.vqvae.vqvae import (
    VQVAE)
from interactive_spectrogram_inpainting.utils.datasets.label_encoders import (
    dump_label_encoders)
from interactive_spectrogram_inpainting.utils.misc import (
    expand_path, get_spectrograms_helper)
from interactive_spectrogram_inpainting.utils.distributed import (
    is_master_process)

# HOP_LENGTH = 512
# N_FFT = 2048
# FS_HZ = 16000


def extract(lmdb_env, loader: WavToSpectrogramDataLoader,
            model: DDP,
            device: str,
            label_encoders: Mapping[str, LabelEncoder] = {},
            ):
    codes_db = lmdb_env.open_db(
        'codes'.encode('utf-8'),
        dupsort=False  # skip duplicate keys
    )

    if is_master_process():
        with lmdb_env.begin(write=True) as txn:
            # store the label encoders along with the database
            # this allows for future conversions
            txn.put('label_encoders'.encode('utf-8'), pickle.dumps(
                label_encoders))

    attribute_names = label_encoders.keys()

    pbar_loader = tqdm(loader)
    for (sample_batch, *categorical_attributes_batch,
            attributes_batch) in pbar_loader:
        sample_batch = sample_batch.to(device)
        sample_names = attributes_batch['note_str']

        *_, id_t, id_b = model(sample_batch)
        id_t = id_t.detach().cpu().numpy()
        id_b = id_b.detach().cpu().numpy()

        for top, bottom, *attributes, sample_name in zip(
                id_t, id_b, *categorical_attributes_batch, sample_names):
            row = CodeRow(top=top, bottom=bottom,
                          attributes=dict(zip(attribute_names,
                                              attributes)),
                          filename=sample_name)
            # starting LMDB transaction here to avoid deadlocks on distributed access
            with lmdb_env.begin(db=codes_db, write=True) as txn:
                txn.put(sample_name.encode('utf-8'), pickle.dumps(row))
            # pbar_loader.set_description(f'inserted: {index}')

        # txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")

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
    parser.add_argument('--overwrite_existing_dump', action='store_true')
    parser.add_argument('--disable_database_creation', action='store_true')

    # DistributedDataParallel arguments
    parser.add_argument(
        '--local_rank', type=int, default=0,
        help="This is provided by torch.distributed.launch")
    parser.add_argument(
        '--local_world_size', type=int, default=1,
        help="Number of GPUs per node, required by torch.distributed.launch")

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

    if not args.disable_database_creation and is_master_process():
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'

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

    # retrieve n_fft, hop length, window length parameters...
    with open(VQVAE_TRAINING_PARAMETERS_PATH, 'r') as f:
        vqvae_training_parameters = json.load(f)
    with open(VQVAE_MODEL_PARAMETERS_PATH, 'r') as f:
        vqvae_model_parameters = json.load(f)
    spectrograms_helper = get_spectrograms_helper(**vqvae_training_parameters)
    spectrograms_helper.to(device)

    for dataset_name, json_data_path in named_dataset_json_data_paths.items():
        assert json_data_path.is_file()
        valid_pitch_range = vqvae_training_parameters['valid_pitch_range']
        transform: Optional[torchvision.transforms.Lambda] = None
        if vqvae_model_parameters['output_spectrogram_min_magnitude'] is not None:
            transform = make_masked_phase_transform(
                vqvae_model_parameters['output_spectrogram_min_magnitude'])

        print("loading dataset", dataset_name)
        nsynth_dataset_with_samples_names = NSynth(
            audio_directory_paths=audio_directory_paths,
            json_data_path=json_data_path,
            valid_pitch_range=valid_pitch_range,
            categorical_field_list=args.categorical_fields,
            squeeze_mono_channel=True,
            return_full_metadata=True,
            remove_qualities_str_from_full_metadata=True,
        )

        print("instantiating wav-to-spectrogram dataloader", dataset_name)
        # converts wavforms to spectrograms on-the-fly on GPU
        distributed_sampler = DistributedSampler(
            nsynth_dataset_with_samples_names,
            shuffle=False)
        loader = WavToSpectrogramDataLoader(
            nsynth_dataset_with_samples_names,
            sampler=distributed_sampler,
            spectrograms_helper=spectrograms_helper,
            batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=False,
            transform=transform,
        )
        batch_size, in_channel, image_height, image_width = next(iter(loader))[0].shape
        image_size = image_height, image_width
        print("image_size:", image_size)

        label_encoders = nsynth_dataset_with_samples_names.label_encoders

        encoders: Optional[Mapping[str, Any]] = None
        decoders: Optional[Mapping[str, Any]] = None
        if vqvae_training_parameters['use_resnet']:
            encoders, decoders = encoder_decoder.xresnet_unet_from_json_parameters(
                in_channel,
                image_size,
                VQVAE_TRAINING_PARAMETERS_PATH
            )

        vqvae = VQVAE.from_parameters_and_weights(
            VQVAE_MODEL_PARAMETERS_PATH,
            VQVAE_WEIGHTS_PATH,
            encoders=encoders,
            decoders=decoders)

        model = vqvae.to(device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            module=model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

        model.eval()

        # TODO(theis): compute appropriate size for the map,
        # for now it's a generous overshoot
        map_size = 100 * 1024 * 1024 * 1024

        lmdb_path = expand_path(OUTPUT_DIR / dataset_name)

        if not args.disable_database_creation:
            if is_master_process():
                os.makedirs(lmdb_path, exist_ok=args.overwrite_existing_dump)
                # store command-line parameters
                with open(OUTPUT_DIR / 'command_line_parameters.json', 'w') as f:
                    json.dump(args.__dict__, f, indent=4)

                dump_label_encoders(
                    nsynth_dataset_with_samples_names.label_encoders,
                    lmdb_path)

            env = lmdb.open(
                str(lmdb_path),
                map_size=map_size,
                max_dbs=2,
                )
            with torch.no_grad():
                print("Start extraction for dataset", dataset_name)
                extract(env, loader, model,
                        device,
                        label_encoders=label_encoders)

        print("Start sanity-check for dataset", dataset_name)
        # check extracted codes by saving to disk the audio for a batch
        # of re-synthesized codemaps
        codes_dataset = LMDBDataset(
            lmdb_path,
            classes_for_conditioning=args.categorical_fields,
            dataset_db_name='codes')

        codes_loader = DataLoader(codes_dataset, batch_size=8,
                                  shuffle=True)

        if is_master_process():
            with torch.no_grad():
                codes_top_sample, codes_bottom_sample, attributes = (
                    next(iter(codes_loader)))
                decoded_sample = model.module.decode_code(
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
