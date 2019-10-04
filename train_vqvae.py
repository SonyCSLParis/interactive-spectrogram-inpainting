import argparse
import pathlib

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler

from nsynth_dataset import NSynthDataset

import os
DIRPATH = os.path.dirname(os.path.abspath(__file__))

def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    image_dump_sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, ((img, pitch), target) in enumerate(loader):
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
    parser.add_argument('--sched', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    path = pathlib.Path(args.path)
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
        dataset = datasets.ImageFolder(args.path, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        dataloader_for_gansynth_normalization = None
        in_channel = 3
    elif dataset_name == 'nsynth':
        nsynth_dataset_path = args.path
        nsynth_dataset = NSynthDataset(
            root_path=nsynth_dataset_path,
            use_mel_frequency_scale=True)
        loader = DataLoader(nsynth_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=True)
        
        in_channel = 2
        
        # vqvae_decoder_activation = nn.Tanh()
        dataloader_for_gansynth_normalization = loader
    else:
        raise ValueError("Unrecognized dataset name: ",
                         dataset_name)

    print("Initializing model")
    vqvae = VQVAE(in_channel=in_channel,
                  decoder_output_activation=vqvae_decoder_activation,
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
