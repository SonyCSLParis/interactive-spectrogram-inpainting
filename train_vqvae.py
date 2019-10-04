import argparse
import pathlib

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler

from nsynth_dataset import NSynthH5Dataset
from GANsynth_pytorch.pytorch_nsynth_lib.nsynth import (
    NSynth, get_mel_spectrogram_and_IF)

import os
DIRPATH = os.path.dirname(os.path.abspath(__file__))

def train(epoch, loader, model, optimizer, scheduler, device):
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

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if i % 100 == 0:
            model.eval()

            sample = img[:image_dump_sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            channel_dim = 1
            for channel_index, channel_name in enumerate(['spectrogram', 'instantaneous_frequency']):
                sample_channel = sample.select(channel_dim, channel_index).unsqueeze(channel_dim)
                out_channel = out.select(channel_dim, channel_index).unsqueeze(channel_dim)
                utils.save_image(
                    torch.cat([sample_channel, out_channel], 0),
                    os.path.join(DIRPATH, 'sample/',
                                 f'{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_{channel_name}.png'),
                    nrow=image_dump_sample_size,
                    # normalize=True,
                    # range=(-1, 1),
                    # scale_each=True,
                )

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_type', choices=['hdf5', 'wav'],
                        required=True)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--validation_dataset_path', type=str)

    args = parser.parse_args()

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

            def chained_transform(sample):
                mel_spec, mel_IF = get_mel_spectrogram_and_IF(
                    sample, hop_length=HOP_LENGTH)
                mel_spec_and_IF_as_image_tensor = NSynthH5Dataset._to_image(
                    [a.astype(np.float32)
                     for a in [mel_spec, mel_IF]])
                return mel_spec_and_IF_as_image_tensor
            to_mel_spec_and_if = transforms.Lambda(chained_transform)
            nsynth_dataset = NSynth(
                root=str(train_dataset_path),
                transform=chained_transform,
                valid_pitch_range=valid_pitch_range,
                categorical_field_list=[],
                convert_to_float=True)
            if args.validation_dataset_path:
                nsynth_validation_dataset = NSynth(
                    root=str(validation_dataset_path),
                    transform=chained_transform,
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
        
        # vqvae_decoder_activation = nn.Tanh()
        dataloader_for_gansynth_normalization = None
        normalizer_statistics = None
        if args.precomputed_normalization_statistics is not None:
            with open(args.precomputed_normalization_statistics, 'rb') as f:
                normalizer_statistics = pickle.load(f)
        else:
        dataloader_for_gansynth_normalization = loader
    else:
        raise ValueError("Unrecognized dataset name: ",
                         dataset_name)

    print("Initializing model")
    if args.disable_normalization:
        dataloader_for_gansynth_normalization = None

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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    print("Starting training")
    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)
        torch.save(
            model.module.state_dict(), f'checkpoint/vqvae_{dataset_name}_{str(i + 1).zfill(3)}.pt'
        )
