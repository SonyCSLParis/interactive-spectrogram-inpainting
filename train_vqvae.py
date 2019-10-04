import uuid
import argparse
import pathlib
import pickle
from typing import Optional
from tqdm import tqdm
import numpy as np
import os


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard.writer import SummaryWriter

from vqvae import VQVAE, InferenceVQVAE
from scheduler import CycleScheduler

from nsynth_dataset import NSynthH5Dataset
from GANsynth_pytorch.pytorch_nsynth_lib.nsynth import (
    NSynth, get_mel_spectrogram_and_IF,
    make_to_mel_spec_and_IF_image_transform)
import GANsynth_pytorch.utils.plots as gansynthplots

import matplotlib as mpl
# use matplotlib without an X server
# on desktop, this avoids matplotlib windows from popping around
mpl.use('Agg')

DIRPATH = os.path.dirname(os.path.abspath(__file__))


HOP_LENGTH = 512
N_FFT = 2048
FS_HZ = 16000


def train(epoch: int, loader: DataLoader, model: nn.Module,
          optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler._LRScheduler,
          device: str,
          inference_vqvae: InferenceVQVAE,
          run_id: str,
          disable_image_dumps: bool = False,
          tensorboard_writer: Optional[SummaryWriter] = None,
          tensorboard_scalar_interval_epochs: int = 1,
          tensorboard_audio_interval_epochs: int = 5,
          tensorboard_num_audio_samples: int = 10,
          ) -> None:
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    image_dump_sample_size = 25

    mse_sum = 0
    mse_n = 0

    model.train()
    for i, (img, pitch) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        batch_reconstruction_mse = recon_loss.item()
        batch_latent_loss = latent_loss.item()
        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {batch_reconstruction_mse:.5f}; '
                f'latent: {batch_latent_loss:.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )
        if tensorboard_writer is not None:
            # add scalar summaries
            num_batches_seen = epoch*len(loader) + i
            num_samples_seen = num_batches_seen * img.shape[0]

            tensorboard_writer.add_scalar('training/reconstruction_mse',
                                          batch_reconstruction_mse,
                                          num_samples_seen)
            tensorboard_writer.add_scalar('training/latent_loss',
                                          batch_latent_loss,
                                          num_samples_seen)

        if not disable_image_dumps and i % 100 == 0:
            model.eval()

            sample = img[:image_dump_sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            channel_dim = 1
            for channel_index, channel_name in enumerate(
                    ['spectrogram', 'instantaneous_frequency']):
                sample_channel = sample.select(channel_dim, channel_index
                                               ).unsqueeze(channel_dim)
                out_channel = out.select(channel_dim, channel_index
                                         ).unsqueeze(channel_dim)
                utils.save_image(
                    torch.cat([sample_channel, out_channel,
                               (sample_channel-out_channel).abs()], 0),
                    os.path.join(DIRPATH, f'samples/{run_ID}/',
                                 f'{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_{channel_name}.png'),
                    nrow=image_dump_sample_size,
                    # normalize=True,
                    # range=(-1, 1),
                    # scale_each=True,
                )

            model.train()


def evaluate(loader: DataLoader, model: nn.Module, device: str):
    with torch.no_grad():
        loader = tqdm(loader, desc='validation')

        criterion = nn.MSELoss()

        latent_loss_weight = 0.25

        mse_sum = 0
        mse_n = 0
        latent_loss_total = 0

        model.eval()
        for i, (img, pitch) in enumerate(loader):
            img = img.to(device)

            out, latent_loss = model(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss

            mse_sum += recon_loss.item() * img.shape[0]
            mse_n += img.shape[0]
            latent_loss_total += latent_loss

        return mse_sum/mse_n, (latent_loss / len(loader)).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dataset', type=str, choices=['nsynth', 'imagenet'])
    parser.add_argument('--dataset_type', choices=['hdf5', 'wav'],
                        default='wav')
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for the Dataloaders')
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--validation_dataset_path', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--input_normalization', action='store_true')
    parser.add_argument('--precomputed_normalization_statistics', type=str,
                        default=None,
                        help=('Path to a pickle file containing the values'
                              'for the GANSynth_pytorch.DataNormalizer object')
                        )

    args = parser.parse_args()

    perform_input_normalization = (args.input_normalization
                                   or args.precomputed_normalization_statistics
                                   )

    run_ID = str(uuid.uuid4())[:6]

    print(args)

    device = 'cuda'

    train_dataset_path = pathlib.Path(args.train_dataset_path)
    validation_dataset_path = pathlib.Path(args.validation_dataset_path)
    dataset_name = args.dataset
    print("Loading dataset: ", dataset_name)
    vqvae_decoder_activation = None
    if dataset_name == 'imagenet':
        transform = transforms.Compose(
            [
                transforms.Resize(args.size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )
        dataset = datasets.ImageFolder(train_dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4)
        dataloader_for_gansynth_normalization = None
        in_channel = 3
    elif dataset_name == 'nsynth':
        if args.dataset_type == 'wav':
            valid_pitch_range = [24, 84]

            transform = make_to_mel_spec_and_IF_image_transform(
                hop_length=HOP_LENGTH,
                n_fft=N_FFT
            )
            nsynth_dataset = NSynth(
                root=str(train_dataset_path),
                transform=transform,
                valid_pitch_range=valid_pitch_range,
                categorical_field_list=[],
                convert_to_float=True)
            if args.validation_dataset_path:
                nsynth_validation_dataset = NSynth(
                    root=str(validation_dataset_path),
                    transform=transform,
                    valid_pitch_range=valid_pitch_range,
                    categorical_field_list=[],
                    convert_to_float=True)
        elif args.dataset_type == 'hdf5':
            nsynth_dataset = NSynthH5Dataset(
                root_path=train_dataset_path,
                use_mel_frequency_scale=True)
            if args.validation_dataset_path:
                nsynth_validation_dataset = NSynthH5Dataset(
                    root_path=validation_dataset_path,
                    use_mel_frequency_scale=True)
        else:
            assert False
        loader = DataLoader(nsynth_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=True)
        validation_loader = DataLoader(nsynth_validation_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True)

        in_channel = 2

        if not (args.input_normalization
                or args.precomputed_normalization_statistics):
            dataloader_for_gansynth_normalization = None
            normalizer_statistics = None
        else:
            if args.precomputed_normalization_statistics is not None:
                with open(args.precomputed_normalization_statistics, 'rb') as f:
                    normalizer_statistics = pickle.load(f)
            else:
                dataloader_for_gansynth_normalization = loader
    else:
        raise ValueError("Unrecognized dataset name: ",
                         dataset_name)

    print("Initializing model")

    vqvae_parameters = {'in_channel': in_channel,
                        'groups': args.groups}

    vqvae = VQVAE(in_channel=in_channel,
                  decoder_output_activation=vqvae_decoder_activation,
                  dataloader_for_gansynth_normalization=dataloader_for_gansynth_normalization,
                  normalizer_statistics=normalizer_statistics,
                  groups=args.groups,
                  )
    if dataloader_for_gansynth_normalization is not None:
        # store normalization parameters
        data_normalizer = vqvae.data_normalizer
        normalization_statistics_path = path / '../normalization_statistics.pkl'
        data_normalizer.dump_statistics(normalization_statistics_path)

    model = nn.DataParallel(vqvae).to(device)
    inference_vqvae = InferenceVQVAE(model, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    MAIN_DIR = pathlib.Path(DIRPATH)
    checkpoints_dir = MAIN_DIR / f'checkpoints/{run_ID}/'
    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(checkpoints_dir / 'model_parameters.pkl', 'wb') as f:
        pickle.dump(vqvae_parameters, f)

    os.makedirs(MAIN_DIR / f'runs/{run_ID}/', exist_ok=True)
    os.makedirs(MAIN_DIR / f'samples/{run_ID}/', exist_ok=True)

    tensorboard_writer = SummaryWriter(MAIN_DIR / f'runs/{run_ID}')

    print("Starting training")
    for epoch_index in range(args.epoch):
        train(epoch_index, loader, model, optimizer, scheduler, device,
              run_id=run_ID,
              disable_image_dumps=args.test,
              tensorboard_writer=tensorboard_writer,
              tensorboard_audio_interval_epochs=3,
              tensorboard_num_audio_samples=5,
              inference_vqvae=inference_vqvae)

        if args.test:
            pass
        else:
            torch.save(
                    model.module.state_dict(),
                    MAIN_DIR / f'checkpoints/{run_ID}/vqvae_{dataset_name}_{str(epoch_index + 1).zfill(3)}.pt'
            )

        # eval on validation set
        mse_validation, latent_loss_validation = evaluate(
            validation_loader, model, device)

        tensorboard_writer.add_scalar('validation/reconstruction_mse',
                                      mse_validation,
                                      global_step=epoch_index)
        tensorboard_writer.add_scalar('validation/latent_loss',
                                      latent_loss_validation,
                                      global_step=epoch_index)

        if tensorboard_writer is not None:
            # if i+1 % tensorboard_audio_interval_epochs == 0:
            # add audio summaries
            samples, reconstructions = inference_vqvae.sample_reconstructions(
                validation_loader)
            samples = samples[:3]
            reconstructions = reconstructions[:3]
            samples_audio = inference_vqvae.mag_and_IF_to_audio(
                samples, use_mel_frequency=True)
            reconstructions_audio = inference_vqvae.mag_and_IF_to_audio(
                reconstructions, use_mel_frequency=True)
            tensorboard_writer.add_audio('Original (end of epoch, validation data)',
                                         samples_audio.flatten(),
                                         epoch_index)
            tensorboard_writer.add_audio('Reconstructions (end of epoch, validation data)',
                                         reconstructions_audio.flatten(),
                                         epoch_index)
            mel_specs_original, mel_IFs_original = (
                np.swapaxes(samples.data.cpu().numpy(), 0, 1))
            mel_specs_reconstructions, mel_IFs_reconstructions = (
                np.swapaxes(reconstructions.data.cpu().numpy(), 0, 1))
            mel_specs = np.concatenate([mel_specs_original,
                                        mel_specs_reconstructions], axis=0)
            mel_IFs = np.concatenate([mel_IFs_original,
                                      mel_IFs_reconstructions], axis=0)

            spec_figure, _ = gansynthplots.plot_mel_representations_batch(
                log_melspecs=mel_specs, mel_IFs=mel_IFs,
                hop_length=HOP_LENGTH, fs_hz=FS_HZ)
            tensorboard_writer.add_figure('Originals + Reconstructions (mel-scale, logspec/IF, validation data)',
                                          spec_figure,
                                          epoch_index)
