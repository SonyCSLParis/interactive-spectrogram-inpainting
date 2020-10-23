from typing import Optional, Mapping, Dict, List
from datetime import datetime
import uuid
import argparse
import pathlib
import json
from fastai.layers import ResBlock
from tqdm import tqdm
import numpy as np
import os

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.modules.loss import _Loss as Loss
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision

from pytorch_nsynth import NSynth
from GANsynth_pytorch.spectrograms_helper import (SpectrogramsHelper,
                                                  MelSpectrogramsHelper)
from GANsynth_pytorch.loader import (WavToSpectrogramDataLoader,
                                     MaskedPhaseWavToSpectrogramDataLoader)
from GANsynth_pytorch.normalizer import DataNormalizer
import GANsynth_pytorch.utils.plots as gansynthplots
from GANsynth_pytorch.spec_ops import _MEL_BREAK_FREQUENCY_HERTZ

from vqvae.vqvae import VQVAE
from vqvae.encoder_decoder import get_xresnet_unet
from utils.losses.spectral import (
    JukeboxMultiscaleSpectralLoss_fromSpectrogram,
    DDSPMultiscaleSpectralLoss_fromSpectrogram)
from utils.training.scheduler import CycleScheduler

import matplotlib as mpl
# use matplotlib without an X server
# on desktop, this avoids matplotlib windows from popping around
mpl.use('Agg')

DIRPATH = os.path.dirname(os.path.abspath(__file__))


HOP_LENGTH = 512
N_FFT = 2048
FS_HZ = 16000


def is_distributed() -> bool:
    return torch.distributed.is_initialized()


def is_master_process() -> bool:
    return not is_distributed() or torch.distributed.get_rank() == 0


def get_spectrograms_helper(args) -> SpectrogramsHelper:
    """Return a SpectrogramsHelper instance adapted to this model"""
    spectrogram_parameters = {
        'fs_hz': args.fs_hz,
        'n_fft': args.n_fft,
        'hop_length': args.hop_length,
        'window_length': args.window_length,
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


def get_reconstruction_criterion(
        criterion_id: str,
        spectrograms_helper: Optional[SpectrogramsHelper] = None
        ) -> Loss:
    if criterion_id == 'MSE':
        return nn.MSELoss()
    elif criterion_id in ['Jukebox', 'JukeboxMultiscaleSpectralLoss']:
        assert spectrograms_helper is not None
        return JukeboxMultiscaleSpectralLoss_fromSpectrogram(
            spectrograms_helper)
    elif criterion_id in ['DDSP', 'DDSPMultiscaleSpectralLoss']:
        assert spectrograms_helper is not None
        return DDSPMultiscaleSpectralLoss_fromSpectrogram(
            spectrograms_helper)
    else:
        raise ValueError("Unexpected reconstruction criterion identifier "
                         + criterion_id)


def write_vqvae_scalars_to_tensorboard(
        summary_writer: SummaryWriter,
        main_tag: str,
        global_step: int,
        model: VQVAE,
        latent_loss_weight: float,
        reconstruction_criterion_name: str,
        reconstruction_loss: float,
        latent_loss: float,
        codes_perplexity_top: float,
        codes_perplexity_bottom: float,
        ):
    vqvae_scalars = {
        'combined_loss': (reconstruction_loss
                          + latent_loss_weight * latent_loss),
        f'reconstruction_loss ({reconstruction_criterion_name})': (
            reconstruction_loss),
        'latent_loss': latent_loss,
        'codes_perplexity_top': codes_perplexity_top,
        'codes_perplexity_bottom': codes_perplexity_bottom,
        'codes_perplexity_ratio_top': (
            codes_perplexity_top / model.n_embed_t
        ),
        'codes_perplexity_ratio_bottom': (
            codes_perplexity_bottom / model.n_embed_b
        )
    }
    for scalar_name, scalar_value in vqvae_scalars.items():
        summary_writer.add_scalar(main_tag + '/' + scalar_name,
                                  scalar_value,
                                  global_step=global_step)


def train(epoch: int, loader: DataLoader, model: VQVAE,
          reconstruction_criterion: Loss,
          optimizer: Optimizer,
          scheduler: Optional[optim.lr_scheduler._LRScheduler],
          device: str,
          metrics: Mapping[str, Loss],
          run_id: str,
          latent_loss_weight: float = 0.25,
          enable_image_dumps: bool = False,
          tensorboard_writer: Optional[SummaryWriter] = None,
          tensorboard_scalar_interval_epochs: int = 1,
          tensorboard_audio_interval_epochs: int = 5,
          tensorboard_num_audio_samples: int = 10,
          dry_run: bool = False,
          clip_grad_norm: Optional[float] = None,
          train_logs_frequency_batches: int = 1,
          ) -> None:
    num_samples_in_dataset = len(loader.dataset)

    reconstruction_criterion.to(device)

    tqdm_loader = tqdm(loader, position=1)
    status_bar = tqdm(total=0, position=0, bar_format='{desc}')

    image_dump_sample_size = 25

    reconstruction_loss_accumulated = 0
    latent_loss_accumulated = 0
    perplexity_t_accumulated = 0
    perplexity_b_accumulated = 0
    num_samples_seen_epoch = 0
    num_samples_seen_total = epoch * num_samples_in_dataset

    model.train()
    for batch_index, (img, *_) in enumerate(tqdm_loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss, perplexity_t_mean, perplexity_b_mean, *_ = (
            model(img))
        reconstruction_loss = reconstruction_criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = reconstruction_loss + latent_loss_weight * latent_loss
        loss.backward()

        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(),
                                     clip_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            batch_size = img.shape[0]
            reconstruction_loss_batch = reconstruction_loss.item()
            latent_loss_batch = latent_loss.item()
            reconstruction_loss_accumulated += (
                reconstruction_loss_batch * batch_size)
            latent_loss_accumulated += (
                latent_loss_batch * batch_size)

            num_samples_seen_epoch += batch_size

            lr = optimizer.param_groups[0]['lr']

            # must take the .mean() again due to DataParallel,
            # perplexity_t_mean has length the number of GPUs
            batch_perplexity_t_mean = perplexity_t_mean.mean().item()
            perplexity_t_accumulated += batch_perplexity_t_mean * batch_size
            batch_perplexity_b_mean = perplexity_b_mean.mean().item()
            perplexity_b_accumulated += batch_perplexity_b_mean * batch_size

            average_reconstruction_loss_epoch = (
                reconstruction_loss_accumulated / num_samples_seen_epoch)
            average_latent_loss_epoch = (
                latent_loss_accumulated / num_samples_seen_epoch)
            average_perplexity_t_epoch = (
                perplexity_t_accumulated / num_samples_seen_epoch)
            average_perplexity_b_epoch = (
                perplexity_b_accumulated / num_samples_seen_epoch)

            latent_loss_batch = latent_loss.item()
            status_bar.set_description_str((
                f'epoch: {epoch + 1}|'
                f'recons.: {average_reconstruction_loss_epoch:.4f}|'
                f'latent: {average_latent_loss_epoch:.4f}|'
                f'perpl. bottom: {average_perplexity_b_epoch:.4f}|'
                f'perpl. top: {average_perplexity_t_epoch:.4f}'
            ))

            num_samples_seen_total += batch_size

        if (tensorboard_writer is not None
                and batch_index % train_logs_frequency_batches == 0):
            # add scalar summaries
            write_vqvae_scalars_to_tensorboard(
                tensorboard_writer,
                'vqvae-training',
                num_samples_seen_total,
                model.module,
                latent_loss_weight,
                type(reconstruction_criterion).__name__,
                reconstruction_loss_batch,
                latent_loss_batch,
                batch_perplexity_t_mean,
                batch_perplexity_b_mean
                )

            with torch.no_grad():
                for metric_name, metric in metrics.items():
                    tensorboard_writer.add_scalar(
                        'vqvae-training-metrics/' + metric_name,
                        metric(img, out),
                        num_samples_seen_total
                    )

        if enable_image_dumps and batch_index % 100 == 0:
            model.eval()

            sample = img[:image_dump_sample_size]
            sample_out = out[:image_dump_sample_size]

            channel_dim = 1
            for channel_index, channel_name in enumerate(
                    ['spectrogram', 'instantaneous_frequency']):
                sample_channel = sample.select(channel_dim, channel_index
                                               ).unsqueeze(channel_dim)
                out_channel = sample_out.select(channel_dim, channel_index
                                                ).unsqueeze(channel_dim)
                torchvision.utils.save_image(
                    torch.cat([sample_channel, out_channel,
                               (sample_channel-out_channel).abs()], 0),
                    os.path.join(DIRPATH, f'samples/{run_ID}/',
                                 f'{str(epoch + 1).zfill(5)}_{str(batch_index).zfill(5)}_{channel_name}.png'),
                    nrow=image_dump_sample_size,
                    # normalize=True,
                    # range=(-1, 1),
                    # scale_each=True,
                )

            model.train()

        if dry_run:
            break

    if tensorboard_writer is not None:
        tensorboard_writer.flush()


@torch.no_grad()
def evaluate(loader: DataLoader, model: nn.Module,
             reconstruction_criterion: Loss,
             device: str,
             reconstruction_metrics: Dict[str, Loss],
             latent_loss_weight: float = 0.25,
             dry_run: bool = False):
    """Evaluate model and return metrics averaged over a validation/test set"""
    loader = tqdm(loader, desc='validation')

    reconstruction_criterion.to(device)

    reconstruction_loss_total = torch.zeros(1).to(device)
    num_samples_seen = 0
    perplexity_t_total = torch.zeros(1).to(device)
    perplexity_b_total = torch.zeros(1).to(device)
    latent_loss_total = torch.zeros(1).to(device)
    reconstruction_metrics_total = {
        metric_name: torch.zeros(1).to(device)
        for metric_name in reconstruction_metrics.keys()
    }

    model.eval()
    for i, (img, *_) in enumerate(loader):
        batch_size = img.shape[0]
        img = img.to(device)

        out, latent_loss, perplexity_t_mean, perplexity_b_mean, *_ = (
            model(img))
        reconstruction_loss_batch = reconstruction_criterion(out, img)

        reconstruction_loss_total += reconstruction_loss_batch * batch_size
        perplexity_t_total += perplexity_t_mean.mean() * batch_size
        perplexity_b_total += perplexity_b_mean.mean() * batch_size
        latent_loss_total += latent_loss.sum()

        for metric_name, metric in reconstruction_metrics.items():
            metric_batch = metric(img, out).item()
            reconstruction_metrics_total[metric_name] += (
                metric_batch * batch_size)

        num_samples_seen += batch_size

        if dry_run:
            break

    reconstruction_loss_average = (
        reconstruction_loss_total.item() / num_samples_seen)
    latent_loss_average = latent_loss_total.item() / num_samples_seen
    perplexity_t_average = (perplexity_t_total.item()
                            / num_samples_seen)
    perplexity_b_average = (perplexity_b_total.item()
                            / num_samples_seen)

    reconstruction_metrics_average = {
        metric_name: metric_total / num_samples_seen
        for metric_name, metric_total in (
            reconstruction_metrics_total.items())
    }

    return (reconstruction_loss_average, latent_loss_average,
            perplexity_t_average, perplexity_b_average,
            reconstruction_metrics_average)


@torch.no_grad()
def add_audio_and_image_samples_tensorboard(
        model: VQVAE, dataloader: DataLoader,
        tensorboard_writer: SummaryWriter,
        num_samples: int,
        spectrograms_helper: SpectrogramsHelper,
        epoch_index: int,
        subset_name: str,
        device: str) -> None:
    print("Dump image and audio samples to Tensorboard")
    # add audio summaries to Tensorboard
    model.eval()
    samples, *_ = next(iter(dataloader))
    samples = samples[:num_samples]
    reconstructions, *_ = (vqvae.forward(samples.to(
        device)))

    samples_audio = spectrograms_helper.to_audio(
        samples)
    reconstructions_audio = spectrograms_helper.to_audio(
        reconstructions)
    tensorboard_writer.add_audio(
        f'original-{subset_name}',
        samples_audio.flatten(),
        epoch_index,
        sample_rate=spectrograms_helper.fs_hz)
    tensorboard_writer.add_audio(
        f'reconstructions-{subset_name}',
        reconstructions_audio.flatten(),
        epoch_index,
        sample_rate=spectrograms_helper.fs_hz)

    # add spectrogram plots to Tensorboards
    mel_specs_original, mel_IFs_original = (
        np.swapaxes(samples.data.cpu().numpy(), 0, 1))
    mel_specs_reconstructions, mel_IFs_reconstructions = (
        np.swapaxes(reconstructions.data.cpu().numpy(), 0, 1))
    mel_specs = np.concatenate([mel_specs_original,
                                mel_specs_reconstructions],
                               axis=0)
    mel_IFs = np.concatenate([mel_IFs_original,
                              mel_IFs_reconstructions],
                             axis=0)

    spec_figure, _ = gansynthplots.plot_mel_representations_batch(
        log_melspecs=mel_specs, mel_IFs=mel_IFs,
        hop_length=spectrograms_helper.hop_length,
        fs_hz=spectrograms_helper.fs_hz,
        cmap='magma')
    tensorboard_writer.add_figure(('Originals+Reconstructions_' +
                                   'Mel_IF-' + subset_name),
                                  spec_figure,
                                  epoch_index)


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
                my_dict[str(k)] = int(v)
            setattr(namespace, self.dest, my_dict)

    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruction_criterion', type=str,
                        choices=['MSE',
                                 'Jukebox',
                                 'DDSP'])
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
    parser.add_argument('--disable_quantization', action='store_true')
    parser.add_argument('--restarts_usage_threshold', type=float, default=1.)
    parser.add_argument('--embeddings_dimension', type=int, default=64)
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
    parser.add_argument('--validation_frequency', default=1, type=int,
                        help=('Frequency (in epochs) at which to compute'
                              'validation metrics'))
    parser.add_argument('--save_frequency', default=1, type=int,
                        help=('Frequency (in epochs) at which to save'
                              'trained weights'))
    parser.add_argument('--train_logs_frequency_batches', default=1, type=int,
                        help=('Frequency (in batches) at which to store training metrics'
                              'to Tensorboard'))
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
    parser.add_argument('--use_resnet', action='store_true')
    parser.add_argument('--resnet_layers_per_downsampling_block', type=int,
                        default=4)
    parser.add_argument('--resnet_expansion', type=int, default=1)
    parser.add_argument(
        '--local_rank', type=int, default=0,
        help="This is provided by torch.distributed.launch")
    parser.add_argument(
        '--local_world_size', type=int, default=1,
        help="Number of GPUs per node, required by torch.distributed.launch")

    args = parser.parse_args()

    perform_input_normalization = (args.input_normalization
                                   or args.precomputed_normalization_statistics
                                   )

    run_ID = ('VQVAE-'
              + datetime.now().strftime('%Y%m%d-%H%M%S-')
              + str(uuid.uuid4())[:6])

    print(args)

    device = f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'

    def expand_path(path: str) -> pathlib.Path:
        return pathlib.Path(path).expanduser().absolute()

    def maybe_get_sampler(dataset: Dataset) -> Optional[DistributedSampler]:
        return DistributedSampler(dataset) if is_distributed() else None

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

    spectrograms_helper = get_spectrograms_helper(args).to(device)

    # converts wavforms to spectrograms on-the-fly on GPU
    dataloader_class: WavToSpectrogramDataLoader
    if args.output_spectrogram_threshold:
        dataloader_class = MaskedPhaseWavToSpectrogramDataLoader
    else:
        dataloader_class = WavToSpectrogramDataLoader

    common_dataset_parameters = {
        'valid_pitch_range': args.valid_pitch_range,
        'categorical_field_list': [],
        'squeeze_mono_channel': True,
        'return_full_metadata': False
    }
    nsynth_dataset = NSynth(
        audio_directory_paths=audio_directory_paths,
        json_data_path=train_dataset_json_data_path,
        **common_dataset_parameters)

    train_sampler = maybe_get_sampler(nsynth_dataset)
    train_loader = dataloader_class(
        dataset=nsynth_dataset,
        sampler=train_sampler,
        spectrograms_helper=spectrograms_helper,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(train_sampler is None),
        pin_memory=True)

    validation_sampler: Optional[DistributedSampler] = None
    validation_loader: Optional[WavToSpectrogramDataLoader] = None
    non_distributed_validation_loader: Optional[WavToSpectrogramDataLoader] = (
        None)
    if args.validation_dataset_json_data_path:
        nsynth_validation_dataset = NSynth(
            audio_directory_paths=audio_directory_paths,
            json_data_path=validation_dataset_json_data_path,
            **common_dataset_parameters
        )
        validation_sampler = maybe_get_sampler(nsynth_validation_dataset)
        validation_loader = dataloader_class(
            dataset=nsynth_validation_dataset,
            sampler=validation_sampler,
            spectrograms_helper=spectrograms_helper,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=(validation_sampler is None),
            pin_memory=True,
            drop_last=False
        )
        non_distributed_validation_loader = dataloader_class(
            dataset=nsynth_validation_dataset,
            spectrograms_helper=spectrograms_helper,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )

    dataloader_for_gansynth_normalization = None
    normalizer_statistics = None
    if args.precomputed_normalization_statistics is not None:
        data_normalizer = DataNormalizer.load_statistics(
            expand_path(args.precomputed_normalization_statistics))
        normalizer_statistics = data_normalizer.statistics
    elif args.input_normalization:
        normalization_statistics_path = (
            train_dataset_json_data_path.parent
            / 'normalization_statistics.json')
        if is_master_process():
            dataloader_for_gansynth_normalization = (
                dataloader_class(
                    dataset=nsynth_dataset,
                    spectrograms_helper=spectrograms_helper,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=False
                )
            )
            # compute normalization parameters
            data_normalizer = DataNormalizer(
                dataloader=dataloader_for_gansynth_normalization)
            # store normalization parameters
            data_normalizer.dump_statistics(normalization_statistics_path)
        dist.barrier()

        data_normalizer = DataNormalizer.load_statistics(
            normalization_statistics_path)
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

    spectrograms, *_ = next(iter(train_loader))
    in_channel = spectrograms[0].shape[0]

    vqvae_parameters = {'in_channel': in_channel,
                        'groups': args.groups,
                        'num_embeddings': args.num_embeddings,
                        'embed_dim': args.embeddings_dimension,
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
                        'disable_quantization': args.disable_quantization,
                        'restarts_usage_threshold': args.restarts_usage_threshold
                        }

    def get_resolution_summary(self, resolution_factors,
                               verbose: bool = True):
        maybe_print = print if verbose else lambda x: None

        spectrograms, *_ = next(iter(train_loader))
        shapes = {}
        shapes['input'] = spectrograms.shape[-2:]
        maybe_print(f"Input images shape: {spectrograms.shape}")

        batch_size, num_channels, input_height, input_width = (
            spectrograms.shape)
        total_resolution_factor = 1
        for layer_name in ['bottom', 'top']:
            resolution_factor = resolution_factors[layer_name]

            maybe_print(layer_name + "layer:")
            maybe_print("\tEncoder downsampling factor:",
                        resolution_factor)
            total_resolution_factor *= resolution_factor
            maybe_print("\tResulting total downsampling factor:",
                        total_resolution_factor)
            layer_height = input_height // total_resolution_factor
            layer_width = input_width // total_resolution_factor
            shapes[layer_name] = (layer_height, layer_width)
            maybe_print(f"\nResolution H={layer_height}, W={layer_width}")
        return shapes

    resolution_summary = get_resolution_summary(
        train_loader, args.resolution_factors, verbose=True)

    decoders: Optional[Mapping[str, nn.Module]] = None
    encoders: Optional[Mapping[str, nn.Module]] = None
    if args.use_resnet:
        encoders, decoders = get_xresnet_unet(
            in_channel,
            resolution_summary['input'],  # channels-first
            args.resolution_factors,
            hidden_channels=args.num_hidden_channels,
            embeddings_dimension=args.embeddings_dimension,
            layers_per_downsampling_block=args.resnet_layers_per_downsampling_block,
            expansion=args.resnet_expansion,
        )

    vqvae = VQVAE(normalizer_statistics=normalizer_statistics,
                  encoders=encoders,
                  decoders=decoders,
                  adapt_quantized_durations=False,
                  **vqvae_parameters
                  )

    model = vqvae.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        module=model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )

    reconstruction_criterion = get_reconstruction_criterion(
        args.reconstruction_criterion, spectrograms_helper)

    reconstruction_metric_names = ['MSE', 'DDSP', 'Jukebox']
    reconstruction_metrics = {
        metric_name: get_reconstruction_criterion(
            metric_name,
            spectrograms_helper).to(device)
        for metric_name in reconstruction_metric_names}

    start_epoch = 0
    if args.resume_training_from is not None:
        # TODO(theis): store and retrieve start epoch index from PyTorch save file
        import re
        checkpoint_path = pathlib.Path(args.resume_training_from)
        epoch_find_regex = '\d+\.pt'
        regex_epoch_output = re.search(epoch_find_regex, checkpoint_path.name)
        if regex_epoch_output is not None:
            # TODO(theis): is this [:3] truncation really necessary??
            assert int(regex_epoch_output[0][:3]) == int(regex_epoch_output[0])
            start_epoch = int(regex_epoch_output[0][:3])
        else:
            raise ValueError("Could not retrieve epoch from path")

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=lambda storage, loc: storage.cuda(args.local_rank)
                )
            )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr,
            n_iter=len(train_loader) * args.num_training_epochs, momentum=None
        )

    MAIN_DIR = pathlib.Path(DIRPATH) / 'data'
    CHECKPOINTS_DIR_PATH = MAIN_DIR / f'checkpoints/{run_ID}/'
    if is_master_process() and not (
            args.dry_run or args.disable_writes_to_disk):
        os.makedirs(CHECKPOINTS_DIR_PATH, exist_ok=True)

        with open(CHECKPOINTS_DIR_PATH / 'command_line_parameters.json', 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        vqvae.store_instantiation_parameters(
            CHECKPOINTS_DIR_PATH / 'model_parameters.json')

        os.makedirs(MAIN_DIR / f'samples/{run_ID}/', exist_ok=True)

    tensorboard_writer: Optional[SummaryWriter] = None
    if is_master_process() and not (args.dry_run or args.disable_tensorboard
                                    or args.disable_writes_to_disk):
        tensorboard_dir_path = MAIN_DIR / f'runs/{run_ID}/'
        os.makedirs(tensorboard_dir_path, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_dir_path)

    print("Starting training")
    if is_master_process() and (
            tensorboard_writer is not None
            and non_distributed_validation_loader is not None):
        # print initial samples, prior to training
        add_audio_and_image_samples_tensorboard(
            model.module, non_distributed_validation_loader,
            tensorboard_writer,
            args.num_validation_samples_audio_tensorboard,
            spectrograms_helper,
            start_epoch-1,
            'VALIDATION',
            device)

    dist.barrier()
    for epoch_index in range(start_epoch, args.num_training_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_index)

        train(epoch_index, train_loader, model, reconstruction_criterion,
              optimizer, scheduler, device,
              reconstruction_metrics,
              run_id=run_ID,
              latent_loss_weight=args.latent_loss_weight,
              enable_image_dumps=args.enable_image_dumps,
              tensorboard_writer=tensorboard_writer,
              tensorboard_audio_interval_epochs=3,
              tensorboard_num_audio_samples=5,
              dry_run=args.dry_run,
              clip_grad_norm=args.clip_grad_norm,
              train_logs_frequency_batches=args.train_logs_frequency_batches)

        if not is_master_process() or args.dry_run or args.disable_writes_to_disk:
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

        if is_master_process() and tensorboard_writer is not None:
            add_audio_and_image_samples_tensorboard(
                model.module, train_loader,
                tensorboard_writer,
                args.num_validation_samples_audio_tensorboard,
                spectrograms_helper,
                epoch_index,
                'TRAINING',
                device)
        dist.barrier()

        if (validation_loader is not None
                and epoch_index % args.validation_frequency == 0):
            # eval on validation set
            if validation_sampler is not None:
                validation_sampler.set_epoch(epoch_index)

            with torch.no_grad():
                (reconstruction_loss_validation, latent_loss_validation,
                 perplexity_t_validation, perplexity_b_validation,
                 reconstruction_metrics_validation) = evaluate(
                    validation_loader, model, reconstruction_criterion,
                    device,
                    reconstruction_metrics,
                    dry_run=args.dry_run,
                    latent_loss_weight=args.latent_loss_weight)
                dist.barrier()

                if is_master_process() and (
                        tensorboard_writer is not None
                        and not (args.dry_run or args.disable_writes_to_disk)):
                    write_vqvae_scalars_to_tensorboard(
                        tensorboard_writer,
                        'vqvae-validation',
                        epoch_index,
                        model.module,
                        args.latent_loss_weight,
                        args.reconstruction_criterion,
                        reconstruction_loss_validation,
                        latent_loss_validation,
                        perplexity_t_validation,
                        perplexity_b_validation
                        )

                    for metric_name, metric_value in (
                            reconstruction_metrics_validation.items()):
                        tensorboard_writer.add_scalar(
                            'vqvae-validation-metrics/' + metric_name,
                            metric_value,
                            global_step=epoch_index
                        )

                    # if (epoch_index+1 % args.tensorboard_audio_interval_epochs
                    #         == 0):
                    add_audio_and_image_samples_tensorboard(
                        model.module, validation_loader,
                        tensorboard_writer,
                        args.num_validation_samples_audio_tensorboard,
                        spectrograms_helper,
                        epoch_index,
                        'VALIDATION',
                        device)

                    tensorboard_writer.flush()
                dist.barrier()

    # Tear down the process group
    dist.destroy_process_group()
