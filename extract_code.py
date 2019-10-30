import argparse
import pickle
import json
import pathlib
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE

from GANsynth_pytorch.pytorch_nsynth_lib.nsynth import (
    NSynth, WavToSpectrogramDataLoader)

HOP_LENGTH = 512
N_FFT = 2048
FS_HZ = 16000


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar_loader = tqdm(loader)

        for sample_batch, sample_name_batch, pitch_batch in pbar_loader:
            sample_batch = sample_batch.to(device)

            *_, id_t, id_b = model(sample_batch)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for file, top, bottom in zip(sample_name_batch, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar_loader.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_output_dir', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for the Dataloaders')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_weights_path', type=str, required=True)
    parser.add_argument('--model_parameters_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])

    args = parser.parse_args()

    MAIN_DIR = pathlib.Path(args.main_output_dir)

    VQVAE_MODEL_WEIGHTS_PATH = pathlib.Path(args.model_weights_path)
    # folder containing the vqvae weights is the ID
    vqvae_id = VQVAE_MODEL_WEIGHTS_PATH.parts[-2]
    vqvae_model_filename = VQVAE_MODEL_WEIGHTS_PATH.stem

    OUTPUT_DIR = MAIN_DIR / f'vqvae-{vqvae_id}-weights-{vqvae_model_filename}/'
    os.makedirs(OUTPUT_DIR, exist_ok=False)

    # store command-line parameters
    with open(OUTPUT_DIR / 'command_line_parameters.json', 'w') as f:
        json.dump(args.__dict__, f)

    device = args.device

    valid_pitch_range = [24, 84]

    nsynth_dataset_with_samples_names = NSynth(
        root=str(args.dataset_path),
        valid_pitch_range=valid_pitch_range,
        categorical_field_list=['note_str'],
        squeeze_mono_channel=True)

    # converts wavforms to spectrograms on-the-fly on GPU
    loader = WavToSpectrogramDataLoader(
        nsynth_dataset_with_samples_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True,
        device=device, n_fft=N_FFT, hop_length=HOP_LENGTH)

    in_channel = 2

    with open(args.model_parameters_path, 'r') as f:
        vqvae_parameters = json.load(f)

    vqvae = VQVAE(**vqvae_parameters)
    vqvae.load_state_dict(torch.load(args.model_weights_path,
                                     map_location=device))
    model = nn.DataParallel(vqvae)
    model = model.to(device)
    model.eval()

    # TODO(theis): compute appropriate size for the map
    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(str(OUTPUT_DIR.absolute()), map_size=map_size)

    with torch.no_grad():
        extract(env, loader, model, device)
