from typing import Optional, Iterable, Mapping
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
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow, LMDBDataset
from vqvae import InferenceVQVAE, VQVAE

from GANsynth_pytorch.pytorch_nsynth_lib.nsynth import (
    NSynth, WavToSpectrogramDataLoader)

HOP_LENGTH = 512
N_FFT = 2048
FS_HZ = 16000


def extract(lmdb_env, loader, model, device,
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
            sample_names = attributes_batch['note_str']

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_output_dir', type=str, required=True)
    parser.add_argument('--checking_samples_dir', type=str, default=None)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for the Dataloaders')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_weights_path', type=str, required=True)
    parser.add_argument('--model_parameters_path', type=str, required=True)
    parser.add_argument('--disable_database_creation', action='store_true')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])

    args = parser.parse_args()

    MAIN_DIR = pathlib.Path(args.main_output_dir)

    VQVAE_MODEL_WEIGHTS_PATH = pathlib.Path(args.model_weights_path)
    # folder containing the vqvae weights is the ID
    vqvae_id = VQVAE_MODEL_WEIGHTS_PATH.parts[-2]
    vqvae_model_filename = VQVAE_MODEL_WEIGHTS_PATH.stem

    OUTPUT_DIR = MAIN_DIR / f'vqvae-{vqvae_id}-weights-{vqvae_model_filename}/'

    if not args.disable_database_creation:
        os.makedirs(OUTPUT_DIR, exist_ok=False)

        # store command-line parameters
        with open(OUTPUT_DIR / 'command_line_parameters.json', 'w') as f:
            json.dump(args.__dict__, f)

    device = args.device

    valid_pitch_range = [24, 84]

    nsynth_dataset_with_samples_names = NSynth(
        root=str(args.dataset_path),
        valid_pitch_range=valid_pitch_range,
        categorical_field_list=['instrument_family_str', 'pitch'],
        squeeze_mono_channel=True)

    # converts wavforms to spectrograms on-the-fly on GPU
    loader = WavToSpectrogramDataLoader(
        nsynth_dataset_with_samples_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True,
        device=device, n_fft=N_FFT, hop_length=HOP_LENGTH)

    in_channel = 2

    with open(args.model_parameters_path, 'r') as f:
        vqvae_parameters = json.load(f)

    vqvae = VQVAE(**vqvae_parameters)
    vqvae.load_state_dict(torch.load(args.model_weights_path,
                                     map_location=device))
    inference_vqvae = InferenceVQVAE(vqvae, device=device,
                                     hop_length=HOP_LENGTH,
                                     n_fft=N_FFT)
    model = nn.DataParallel(vqvae)
    model = model.to(device)
    model.eval()

    # TODO(theis): compute appropriate size for the map
    map_size = 100 * 1024 * 1024 * 1024

    lmdb_path = OUTPUT_DIR.absolute()

    if not args.disable_database_creation:
        env = lmdb.open(str(lmdb_path), map_size=map_size)
        with torch.no_grad():
            extract(env, loader, model, device,
                    label_encoders=nsynth_dataset_with_samples_names.label_encoders)

    if args.checking_samples_dir:
        # check extracted codes
        codes_dataset = LMDBDataset(str(lmdb_path))
        codes_loader = DataLoader(codes_dataset, batch_size=8, shuffle=True)
        with torch.no_grad():
            codes_top_sample, codes_bottom_sample, instrument_families, pitches = (
                next(iter(codes_loader)))
            decoded_sample = vqvae.decode_code(codes_top_sample.to(device),
                                               codes_bottom_sample.to(device))

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
