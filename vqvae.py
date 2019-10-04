from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from GANsynth_pytorch.normalizer import DataNormalizer


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)

        # residual connection
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel: int, channel: int, n_res_block: int,
                 n_res_channel: int, stride: int, groups: int = 1):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1,
                          groups=groups),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1, groups=groups),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, channel: int,
                 n_res_block: int, n_res_channel: int, stride: int,
                 groups: int = 1,
                 output_activation: Optional[nn.Module] = None):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2,
                                       padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1,
                        groups=groups),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2,
                                   padding=1, groups=groups)
            )

        if output_activation is not None:
            blocks.append(output_activation)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    """Implementation of the VQ-VAE model
    
    Arguments:
        in_channel (int):
            number of channels in the input images
        channel (int):
            number of channels in the encoder/decoder's first layers
        n_res_block (int):
            number of residual blocks in the encoder/decoder
        n_res_channel (int):
            number of channels in the residual blocks
        embed_dim (int):
            dimension of the latent codes
        n_embed (int):
            size of the latent book (number of different latent codes to learn)
        decay (float, 0 =< decay =< 1):
            decay rate of the latent codes
    """
    def __init__(
        self,
        in_channel: int = 3,
        channel: int = 128,
        n_res_block: int = 2,
        n_res_channel: int = 32,
        embed_dim: int = 64,
        n_embed: int = 512,
        decay: float = 0.99,
        groups: int = 1,
        decoder_output_activation: Optional[nn.Module] = None,
        dataloader_for_gansynth_normalization: Optional[torch.utils.data.DataLoader] = None,
        normalizer_statistics: Optional[object] = None
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel,
                             stride=4, groups=groups)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel,
                             stride=2, groups=groups)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed, decay=decay)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel,
            stride=2, groups=groups
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed, decay=decay)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
            output_activation=decoder_output_activation,
            groups=groups
        )

        self.use_gansynth_normalization = (dataloader_for_gansynth_normalization is not None
                                           or normalizer_statistics is not None)
        self.dataloader = dataloader_for_gansynth_normalization
        self.normalizer_statistics = normalizer_statistics
        if self.normalizer_statistics:
            self.data_normalizer = DataNormalizer(**self.normalizer_statistics)
        elif self.use_gansynth_normalization:
            self.data_normalizer = DataNormalizer(self.dataloader)

    def forward(self, input):
        if self.use_gansynth_normalization:
            input = self.data_normalizer.normalize(input)

        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


class InferenceVQVAE(object):
    def __init__(self, vqvae: VQVAE, device: str):
        self.vqvae = vqvae
        self.device = device

    def sample_reconstructions(self, dataloader):
        self.vqvae.eval()

        iterator = iter(dataloader)
        mag_and_IF_batch, _ = next(iterator)
        with torch.no_grad():
            reconstructed_mag_and_IF_batch, _ = self.vqvae.forward(
                mag_and_IF_batch.to(self.device))

        return mag_and_IF_batch, reconstructed_mag_and_IF_batch

    def mag_and_IF_to_audio(self, mag_and_IF: torch.Tensor,
                            use_mel_frequency: bool = True):
        input_is_batch = True
        if mag_and_IF.ndim == 3:
            # a single sample was provided, wrap it as a batch
            input_is_batch = False
            mag_and_IF = mag_and_IF.unsqueeze(0)
        elif mag_and_IF.ndim == 4:
            pass
        else:
            raise ValueError("Input must either be a sample of shape "
                             "[channels, freq, time] or a batch of such")

        if use_mel_frequency:
            spec_to_audio = (GANsynth_pytorch.spectrograms_helper
                             .mel_mag_and_IF_to_audio)
        else:
            spec_to_audio = (GANsynth_pytorch.spectrograms_helper
                             .mag_and_IF_to_audio)

        channel_dimension = 1
        mag_batch = mag_and_IF.select(channel_dimension, 0
                                      ).data.cpu().numpy()
        IF_batch = mag_and_IF.select(channel_dimension, 1
                                     ).data.cpu().numpy()

        audios = []
        for mag, IF in zip(mag_batch, IF_batch):
            audio_np = spec_to_audio(mag, IF)
            audio = torch.from_numpy(audio_np)
            audios.append(audio)
        if input_is_batch:
            return torch.cat([audio.unsqueeze(0) for audio in audios],
                             0)
        else:
            return audios[0]
