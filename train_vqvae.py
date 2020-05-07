from typing import Optional, Union, Sequence, Dict, List
from datetime import datetime
import uuid
import argparse
import pathlib
import pickle
import json
from tqdm import tqdm
import numpy as np
import os


import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard.writer import SummaryWriter

from vqvae import VQVAE
from scheduler import CycleScheduler

from pytorch_nsynth import NSynth
from GANsynth_pytorch.spectrograms_helper import (SpectrogramsHelper,
                                                  MelSpectrogramsHelper)
from GANsynth_pytorch.loader import (WavToSpectrogramDataLoader,
                                     MaskedPhaseWavToSpectrogramDataLoader)
from GANsynth_pytorch.normalizer import DataNormalizer
import GANsynth_pytorch.utils.plots as gansynthplots
from GANsynth_pytorch.spec_ops import _MEL_BREAK_FREQUENCY_HERTZ

import matplotlib as mpl
# use matplotlib without an X server
# on desktop, this avoids matplotlib windows from popping around
mpl.use('Agg')

DIRPATH = os.path.dirname(os.path.abspath(__file__))


HOP_LENGTH = 512
N_FFT = 2048
FS_HZ = 16000


def get_spectrograms_helper(args) -> SpectrogramsHelper:
    """Return a SpectrogramsHelper instance adapted to this model"""
    spectrogram_parameters = {
        'fs_hz': args.fs_hz,
        'n_fft': args.n_fft,
        'hop_length': args.hop_length,
        'window_length': args.window_length,
        'device': args.device,
    }
    if args.use_mel_scale:
        return MelSpectrogramsHelper(
            **spectrogram_parameters,
            lower_edge_hertz=args.mel_scale_lower_edge_hertz,
            upper_edge_hertz=args.mel_scale_upper_edge_hertz,
            mel_break_frequency_hertz=args.mel_scale_break_frequency_hertz,
            mel_bin_width_threshold_factor=(
                args.mel_scale_expand_resolution_factor)
        )
    else:
        return SpectrogramsHelper(**spectrogram_parameters)


def train(epoch: int, loader: DataLoader, model: nn.Module,
          optimizer: Optimizer,
          scheduler: Optional[optim.lr_scheduler._LRScheduler],
          device: str,
          run_id: str,
          latent_loss_weight: float = 0.25,
          enable_image_dumps: bool = False,
          tensorboard_writer: Optional[SummaryWriter] = None,
          tensorboard_scalar_interval_epochs: int = 1,
          tensorboard_audio_interval_epochs: int = 5,
          tensorboard_num_audio_samples: int = 10,
          dry_run: bool = False,
          clip_grad_norm: Optional[float] = None
          ) -> None:
    num_samples_in_dataset = len(loader.dataset)

    parallel_model = nn.DataParallel(model).to(device)

    tqdm_loader = tqdm(loader, position=1)
    status_bar = tqdm(total=0, position=0, bar_format='{desc}')

    criterion = nn.MSELoss()

    image_dump_sample_size = 25

    mse_sum = 0
    mse_n = 0
    num_samples_seen_epoch = 0
    num_samples_seen_total = epoch * num_samples_in_dataset

    parallel_model.train()
    for i, (img, _) in enumerate(tqdm_loader):
        parallel_model.zero_grad()

        img = img.to(device)

        out, latent_loss, perplexity_t_mean, perplexity_b_mean, *_ = (
            parallel_model(img))
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(parallel_model.parameters(),
                                     clip_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = img.shape[0]
        mse_batch = recon_loss.item()
        mse_sum += mse_batch * batch_size

        num_samples_seen_epoch += batch_size
        num_samples_seen_total += batch_size

        lr = optimizer.param_groups[0]['lr']

        # must take the .mean() again due to DataParallel,
        # perplexity_t_mean is has length the number of GPUs
        batch_perplexity_t_mean = perplexity_t_mean.mean().item()
        batch_perplexity_b_mean = perplexity_b_mean.mean().item()

        batch_reconstruction_mse = recon_loss.item()
        batch_latent_loss = latent_loss.item()
        status_bar.set_description_str(
            (
                f'epoch: {epoch + 1}; '
                f'avg mse: {mse_sum / num_samples_seen_epoch:.4f}; mse: {mse_batch:.4f}; latent: {batch_latent_loss:.4f}; '
                f'perpl_bottom: {batch_perplexity_b_mean:.4f}; perpl_top: {batch_perplexity_t_mean:.4f}'
            )
        )
        if tensorboard_writer is not None:
            # add scalar summaries
            tensorboard_writer.add_scalar('training/reconstruction_mse',
                                          batch_reconstruction_mse,
                                          num_samples_seen_total)
            tensorboard_writer.add_scalar('training/latent_loss',
                                          batch_latent_loss,
                                          num_samples_seen_total)
            tensorboard_writer.add_scalar('training/perplexity_top',
                                          batch_perplexity_t_mean,
                                          num_samples_seen_total)
            tensorboard_writer.add_scalar('training/perplexity_bottom',
                                          batch_perplexity_b_mean,
                                          num_samples_seen_total)

        if enable_image_dumps and i % 100 == 0:
            parallel_model.eval()

            sample = img[:image_dump_sample_size]
            sample_out = out[:image_dump_sample_size]

            channel_dim = 1
            for channel_index, channel_name in enumerate(
                    ['spectrogram', 'instantaneous_frequency']):
                sample_channel = sample.select(channel_dim, channel_index
                                               ).unsqueeze(channel_dim)
                out_channel = sample_out.select(channel_dim, channel_index
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

            parallel_model.train()
        if dry_run:
            break

    if tensorboard_writer is not None:
        tensorboard_writer.flush()


def evaluate(loader: DataLoader, model: nn.Module, device: str,
             latent_loss_weight: float = 0.25,
             dry_run: bool = False):
    with torch.no_grad():
        num_samples_in_dataset = len(loader.dataset)

        loader = tqdm(loader, desc='validation')

        parallel_model = nn.DataParallel(model)

        criterion = nn.MSELoss()

        mse_total = torch.zeros(1)
        perplexity_t_total = torch.zeros(1)
        perplexity_b_total = torch.zeros(1)
        mse_n = torch.zeros(1)
        latent_loss_total = torch.zeros(1)

        parallel_model.eval()
        for i, (img, _) in enumerate(loader):
            img = img.to(device)

            out, latent_loss, perplexity_t_mean, perplexity_b_mean, *_ = (
                parallel_model(img))
            recon_loss = criterion(out, img)
            latent_loss_mean = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss_mean

            mse_total += recon_loss * img.shape[0]
            perplexity_t_total += perplexity_t_mean.mean() * img.shape[0]
            perplexity_b_total += perplexity_b_mean.mean() * img.shape[0]
            latent_loss_total += latent_loss.sum()
            if args.dry_run:
                break

        mse_average = mse_total.item() / num_samples_in_dataset
        latent_loss_average = latent_loss_total.item() / num_samples_in_dataset
        perplexity_t_average = (perplexity_t_total.item()
                                / num_samples_in_dataset)
        perplexity_b_average = (perplexity_b_total.item()
                                / num_samples_in_dataset)

        return (mse_average, latent_loss_average,
                perplexity_t_average, perplexity_b_average)


if __name__ == '__main__':
    class StoreDictKeyPair(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            my_dict = {}
            for kv in values.split(","):
                k, v = kv.split("=")
                my_dict[str(k)] = int(v)
            setattr(namespace, self.dest, my_dict)

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    # parser.add_argument('--strides', nargs='+', type=int, default=[2, 4],
    #                     choices=[2, 4, 8, 16])
    parser.add_argument('--resolution_factors', action=StoreDictKeyPair,
                        default={'top': 2, 'bottom': 2})
    parser.add_argument('--fs_hz', type=int, default=16000)
    parser.add_argument('--window_length', type=int, default=2048)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--use_local_kernels', action='store_true')
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--num_embeddings', type=int, default=512)
    parser.add_argument('--num_hidden_channels', type=int, default=128)
    parser.add_argument('--num_residual_channels', type=int, default=32)
    parser.add_argument('--num_training_epochs', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--latent_loss_weight', type=float, default=0.25)
    parser.add_argument('--clip_grad_norm', type=float, default=None)
    parser.add_argument('--dataset', type=str, choices=['nsynth', 'imagenet'])
    parser.add_argument('--use_mel_scale', action='store_true')
    parser.add_argument('--mel_scale_lower_edge_hertz', type=float,
                        default=0.0)
    parser.add_argument('--mel_scale_upper_edge_hertz', type=float,
                        default=16000/2.0)
    parser.add_argument('--mel_scale_break_frequency_hertz', type=float,
                        default=_MEL_BREAK_FREQUENCY_HERTZ)
    parser.add_argument('--mel_scale_expand_resolution_factor', type=float,
                        default=1.5)
    parser.add_argument('--dataset_type', choices=['hdf5', 'wav'],
                        default='wav')
    parser.add_argument('--normalize_input_images', action='store_true')
    parser.add_argument('--valid_pitch_range', type=int, nargs=2,
                        default=[24, 84])
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_spectrogram_threshold', action='store_true')
    # parser.add_argument('--output_spectrogram_thresholded_value', type=float,
    #                     default=SPEC_THRESHOLD)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for the Dataloaders')
    parser.add_argument('--dataset_audio_directory_paths', type=str,
                        nargs='+')
    parser.add_argument('--train_dataset_json_data_path', type=str,
                        required=True)
    parser.add_argument('--validation_dataset_json_data_path', type=str)
    parser.add_argument('--save_frequency', default=1, type=int,
                        help=('Frequency (in epochs) at which to save'
                              'trained weights'))
    parser.add_argument('--enable_image_dumps', action='store_true',
                        help=('Dump png pictures of the spectrograms during training.'
                              'WARNING: Takes up a lot of space!'))
    parser.add_argument('--disable_writes_to_disk', action='store_true')
    parser.add_argument('--disable_tensorboard', action='store_true')
    parser.add_argument('--dry_run', action='store_true',
                        help=('Test run performing only one step of training'
                              'and evaluation'))
    parser.add_argument('--input_normalization', action='store_true')
    parser.add_argument('--precomputed_normalization_statistics', type=str,
                        default=None,
                        help=('Path to a JSON file containing the values'
                              'for the GANSynth_pytorch.DataNormalizer object')
                        )
    parser.add_argument('--corrupt_codes', choices=['bottom', 'top', 'both'],
                        type=str,
                        help='Whether to corrupt codes using random +/- 1 noise')
    parser.add_argument('--embeddings_initial_variance', type=float, default=1)
    parser.add_argument('--resume_training_from', type=str,
                        help='Path to a checkpoint to resume training from')
    parser.add_argument('--num_validation_samples_audio_tensorboard', type=int,
                        default=3, help=("Number of validation audio samples "
                                         "to store in Tensorboard"))

    args = parser.parse_args()

    perform_input_normalization = (args.input_normalization
                                   or args.precomputed_normalization_statistics
                                   )

    run_ID = datetime.now().strftime('%Y%m%d-%H%M%S-') + str(uuid.uuid4())[:6]

    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def expand_path(path: str) -> pathlib.Path:
        return pathlib.Path(path).expanduser().absolute()

    audio_directory_paths = [expand_path(path)
                             for path in args.dataset_audio_directory_paths]

    train_dataset_json_data_path = expand_path(
        args.train_dataset_json_data_path)
    validation_dataset_json_data_path = expand_path(
        args.validation_dataset_json_data_path)
    dataset_name = args.dataset
    print("Loading dataset: ", dataset_name)
    vqvae_decoder_activation = None
    output_transform = None

    spectrograms_helper = get_spectrograms_helper(args)

    # converts wavforms to spectrograms on-the-fly on GPU
    dataloader_class: WavToSpectrogramDataLoader
    if args.output_spectrogram_threshold:
        dataloader_class = MaskedPhaseWavToSpectrogramDataLoader
    else:
        dataloader_class = WavToSpectrogramDataLoader

    common_dataset_parameters = {
        'valid_pitch_range': args.valid_pitch_range,
        'categorical_field_list': [],
        'squeeze_mono_channel': True
    }
    nsynth_dataset = NSynth(
        audio_directory_paths=audio_directory_paths,
        json_data_path=train_dataset_json_data_path,
        **common_dataset_parameters)
    loader = dataloader_class(
        dataset=nsynth_dataset,
        spectrograms_helper=spectrograms_helper,
        batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True)

    validation_loader: Optional[WavToSpectrogramDataLoader] = None
    if args.validation_dataset_json_data_path:
        nsynth_validation_dataset = NSynth(
            audio_directory_paths=audio_directory_paths,
            json_data_path=validation_dataset_json_data_path,
            **common_dataset_parameters
        )
        validation_loader = dataloader_class(
            dataset=nsynth_validation_dataset,
            spectrograms_helper=spectrograms_helper,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True, pin_memory=True,
        )

    in_channel = next(iter(loader))[0].shape(0)

    dataloader_for_gansynth_normalization = None
    normalizer_statistics = None
    if args.precomputed_normalization_statistics is not None:
        data_normalizer = DataNormalizer.load_statistics(
            expand_path(args.precomputed_normalization_statistics))
        normalizer_statistics = data_normalizer.statistics
    elif args.input_normalization:
        dataloader_for_gansynth_normalization = loader
        # compute normalization parameters
        data_normalizer = DataNormalizer(
            dataloader=dataloader_for_gansynth_normalization)
        # store normalization parameters
        normalization_statistics_path = (
            train_dataset_json_data_path.parent
            / 'normalization_statistics.json')
        data_normalizer.dump_statistics(normalization_statistics_path)
        normalizer_statistics = data_normalizer.statistics

    print("Initializing model")

    corruption_weights_base = [0.1, 0.8, 0.1]
    corruption_weights: Dict[str, Optional[List[float]]] = {
        'top': None,
        'bottom': None
    }
    if args.corrupt_codes is not None:
        if args.corrupt_codes == 'both':
            corruption_weights = {
                'bottom': corruption_weights_base,
                'top': corruption_weights_base
            }
        elif args.corrupt_codes == 'top' or args.corrupt_codes == 'bottom':
            corruption_weights[args.corrupt_codes] = corruption_weights_base
        else:
            assert False, "Not permitted by argparse parameters"

    vqvae_parameters = {'in_channel': in_channel,
                        'groups': args.groups,
                        'num_embeddings': args.num_embeddings,
                        'num_hidden_channels': args.num_hidden_channels,
                        'num_residual_channels': args.num_residual_channels,
                        'corruption_weights': corruption_weights,
                        'embeddings_initial_variance':
                            args.embeddings_initial_variance,
                        # 'resume_training_from': args.resume_training_from
                        'resolution_factors': args.resolution_factors,
                        'output_spectrogram_min_magnitude': (
                            spectrograms_helper.safelog_eps
                            if args.output_spectrogram_threshold else None),
                        'use_local_kernels': args.use_local_kernels,
                        }

    def print_resolution_summary(loader, resolution_factors):
        sample = next(iter(loader))[0][0]
        print(f"Input images shape: {sample.shape}")

        num_channels, height, width = sample.shape
        total_resolution_factor = 1
        for layer_name, resolution_factor in resolution_factors.items():
            total_resolution_factor *= resolution_factor
            print(f"Layer {layer_name}: ")
            print(f"\nAdditional downsampling factor {resolution_factor}")
            C, H, W = sample.shape
            layer_height = height // total_resolution_factor
            layer_width = width // total_resolution_factor
            print(f"\nResolution H={layer_height}, W={layer_width}")

    print_resolution_summary(loader, args.resolution_factors)

    vqvae = VQVAE(normalizer_statistics=normalizer_statistics,
                  **vqvae_parameters
                  )

    model = vqvae.to(device)

    start_epoch = 0
    if args.resume_training_from is not None:
        # TODO(theis): store and retrieve epoch from PyTorch save file
        import re
        checkpoint_path = pathlib.Path(args.resume_training_from)
        epoch_find_regex = '\d+\.pt'
        regex_epoch_output = re.search(epoch_find_regex, checkpoint_path.name)
        if regex_epoch_output is not None:
            start_epoch = int(regex_epoch_output[0][:3])
        else:
            raise ValueError("Could not retrieve epoch from path")
        model.load_state_dict(torch.load(checkpoint_path,
                                         map_location=device)
                              )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr,
            n_iter=len(loader) * args.num_training_epochs, momentum=None
        )

    MAIN_DIR = pathlib.Path(DIRPATH)
    CHECKPOINTS_DIR_PATH = MAIN_DIR / f'checkpoints/{run_ID}/'
    if not (args.dry_run or args.disable_writes_to_disk):
        os.makedirs(CHECKPOINTS_DIR_PATH, exist_ok=True)

        with open(CHECKPOINTS_DIR_PATH / 'command_line_parameters.json', 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        vqvae.store_instantiation_parameters(
            CHECKPOINTS_DIR_PATH / 'model_parameters.json')

        os.makedirs(MAIN_DIR / f'samples/{run_ID}/', exist_ok=True)

    tensorboard_writer = None
    if not (args.dry_run or args.disable_tensorboard
            or args.disable_writes_to_disk):
        tensorboard_dir_path = MAIN_DIR / f'runs/{run_ID}/'
        os.makedirs(tensorboard_dir_path, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_dir_path)

    print("Starting training")
    for epoch_index in range(start_epoch, args.num_training_epochs):
        train(epoch_index, loader, model, optimizer, scheduler, device,
              run_id=run_ID,
              latent_loss_weight=args.latent_loss_weight,
              enable_image_dumps=args.enable_image_dumps,
              tensorboard_writer=tensorboard_writer,
              tensorboard_audio_interval_epochs=3,
              tensorboard_num_audio_samples=5,
              dry_run=args.dry_run,
              clip_grad_norm=args.clip_grad_norm)

        if args.dry_run or args.disable_writes_to_disk:
            pass
        else:
            if (epoch_index == args.num_training_epochs - 1  # save last run
                    or (epoch_index-start_epoch) % args.save_frequency == 0):
                checkpoint_filename = (f'vqvae_{dataset_name}_'
                                       f'{str(epoch_index + 1).zfill(3)}.pt')
                torch.save(
                        model.state_dict(),
                        CHECKPOINTS_DIR_PATH / checkpoint_filename
                )

        if validation_loader is not None:
            # eval on validation set
            with torch.no_grad():
                (mse_validation, latent_loss_validation,
                 perplexity_t_validation, perplexity_b_validation) = evaluate(
                    validation_loader, model,
                    device, dry_run=args.dry_run,
                    latent_loss_weight=args.latent_loss_weight)

                if tensorboard_writer is not None and not (
                        args.dry_run or args.disable_writes_to_disk):
                    tensorboard_writer.add_scalar('validation/reconstruction_mse',
                                                  mse_validation,
                                                  global_step=epoch_index)
                    tensorboard_writer.add_scalar('validation/latent_loss',
                                                  latent_loss_validation,
                                                  global_step=epoch_index)
                    tensorboard_writer.add_scalar('validation/perplexity_top',
                                                  perplexity_t_validation,
                                                  global_step=epoch_index)
                    tensorboard_writer.add_scalar('validation/perplexity_bottom',
                                                  perplexity_b_validation,
                                                  global_step=epoch_index)

                    # if i+1 % tensorboard_audio_interval_epochs == 0:

                    # add audio summaries to Tensorboard
                    model.eval()
                    validation_samples, *_ = next(iter(validation_loader))
                    reconstructions, *_ = (vqvae.forward(validation_samples.to(
                        device)))

                    validation_samples = validation_samples[
                        :args.num_validation_samples_audio_tensorboard]
                    reconstructions = reconstructions[
                        :args.num_validation_samples_audio_tensorboard]

                    validation_samples_audio = spectrograms_helper.to_audio(
                        validation_samples)
                    reconstructions_audio = spectrograms_helper.to_audio(
                        reconstructions)
                    tensorboard_writer.add_audio(
                        'Original (end of epoch, validation data)',
                        validation_samples_audio.flatten(),
                        epoch_index)
                    tensorboard_writer.add_audio(
                        'Reconstructions (end of epoch, validation data)',
                        reconstructions_audio.flatten(),
                        epoch_index)

                    # add spectrogram plots to Tensorboards
                    mel_specs_original, mel_IFs_original = (
                        np.swapaxes(validation_samples.data.cpu().numpy(), 0, 1))
                    mel_specs_reconstructions, mel_IFs_reconstructions = (
                        np.swapaxes(reconstructions.data.cpu().numpy(), 0, 1))
                    mel_specs = np.concatenate([mel_specs_original,
                                                mel_specs_reconstructions],
                                               axis=0)
                    mel_IFs = np.concatenate([mel_IFs_original,
                                              mel_IFs_reconstructions], axis=0)

                    spec_figure, _ = gansynthplots.plot_mel_representations_batch(
                        log_melspecs=mel_specs, mel_IFs=mel_IFs,
                        hop_length=spectrograms_helper.hop_length,
                        fs_hz=spectrograms_helper.fs_hz)
                    tensorboard_writer.add_figure('Originals + Reconstructions (mel-scale, logspec/IF, validation data)',
                                                  spec_figure,
                                                  epoch_index)

                    tensorboard_writer.flush()
