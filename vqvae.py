from typing import Optional, Iterable, Union, List, Mapping
import numpy as np
import pathlib
import json

import torch
from torch import nn
from torch.nn import functional as F

import GANsynth_pytorch
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
    def __init__(self, dim: int, n_embed: int, decay: float = 0.99,
                 eps: float = 1e-5, embeddings_initial_variance: float = 1,
                 corruption_weights: Optional[List[float]] = None):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.corruption_weights = corruption_weights
        self.embeddings_initial_variance = embeddings_initial_variance

        # initialize embeddings
        embed = (torch.randn(dim, n_embed)
                 * np.sqrt(self.embeddings_initial_variance))
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

        if self.training and self.corruption_weights is not None:
            num_indexes = embed_ind.numel()
            random_plus_minus_one = (
                (torch.multinomial(torch.Tensor(self.corruption_weights),
                                   num_indexes,
                                   replacement=True)
                 - 1)
                .reshape(embed_ind.shape)
                .to(embed_ind.device)
            )
            embed_ind = (embed_ind + random_plus_minus_one) % self.n_embed

        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay,
                                                      embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        average_embedding_probas = embed_onehot.mean(dim=0)
        code_assignation_perplexity = torch.exp(
            - torch.sum(average_embedding_probas
                        * torch.log(average_embedding_probas.clamp(min=1e-7))))
        return quantize, diff, embed_ind, code_assignation_perplexity

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel: int, channel: int):
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
                 n_res_channel: int, resolution_factor: int, groups: int = 1):
        super().__init__()

        # downsampling module
        if resolution_factor == 16:
            blocks = [
                nn.Conv2d(in_channel, channel // 4, 4, stride=2,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, 3 * channel // 4, 4, stride=2,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(3 * channel // 4, channel, 4, stride=2, padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1, groups=groups),
            ]
        elif resolution_factor == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channel // 2, channel, 4, stride=2, padding=1,
                    groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1, groups=groups),
            ]
        elif resolution_factor == 4:
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
        elif resolution_factor == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1, groups=groups),
            ]
        else:
            raise ValueError(f"Unexpected resolution factor {resolution_factor}")

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, channel: int,
                 n_res_block: int, n_res_channel: int, resolution_factor: int,
                 groups: int = 1,
                 output_activation: Optional[nn.Module] = None):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # upsampling module
        if resolution_factor == 16:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, 3 * channel // 4, 4, stride=2,
                                       padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        3 * channel // 4, channel // 2, 4, stride=2, padding=1,
                        groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, channel // 4, 4, stride=2, padding=1,
                        groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 4, out_channel, 4, stride=2, padding=1,
                        groups=groups)
                ]
            )
        elif resolution_factor == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2,
                                       padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, channel // 2, 4, stride=2, padding=1,
                        groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1,
                        groups=groups)
                ]
            )
        elif resolution_factor == 4:
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
        elif resolution_factor == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2,
                                   padding=1, groups=groups)
            )
        else:
            raise ValueError(f"Unexpected stride value {stride}")

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
            number of channels in the encoder/decoder
        n_res_block (int):
            number of residual blocks in the encoder/decoder
        n_res_channel (int):
            number of channels in the residual blocks
        embed_dim (int or Iterable[int]):
            dimension of the latent codes
        n_embed (int):
            size of the latent book (number of different latent codes to learn)
        decay (float, 0 =< decay =< 1):
            decay rate of the latent codes
        resolution_factors (Mapping[str, int]):
            associates each code layer to its down/up-sampling factor.
            NOTE: the factors are meant respective to the previous
            code layer, e.g. for a two-level VA-VAE, with top obtained
            by further downsampling bottom, the resolution_factors['top']
            is the down/up-sampling factor from the bottom layer to the top
            layer.
    """
    def __init__(
        self,
        in_channel: int = 3,
        num_hidden_channels: int = 128,
        n_res_block: int = 2,
        num_residual_channels: int = 32,
        embed_dim: int = 64,
        num_embeddings: Union[int, Iterable[int]] = 512,
        decay: float = 0.99,
        groups: int = 1,
        resolution_factors: Mapping[str, int] = {
            'bottom': 4,
            'top': 2,
        },
        embeddings_initial_variance: float = 1,
        decoder_output_activation: Optional[nn.Module] = None,
        dataloader_for_gansynth_normalization: Optional[torch.utils.data.DataLoader] = None,
        normalizer_statistics: Optional[object] = None,
        corruption_weights: Mapping[str, Optional[List[float]]] = {'top': None,
                                                                   'bottom': None}
    ):
        # store instantiation parameters
        self.in_channel = in_channel
        self.num_hidden_channels = num_hidden_channels
        self.n_res_block = n_res_block
        self.num_residual_channels = num_residual_channels
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.groups = groups
        self.resolution_factors = resolution_factors
        self.embeddings_initial_variance = embeddings_initial_variance
        self.decoder_output_activation = decoder_output_activation
        self.dataloader_for_gansynth_normalization = dataloader_for_gansynth_normalization
        self.normalizer_statistics = normalizer_statistics
        self.corruption_weights = corruption_weights

        self._instantiation_parameters = self.__dict__.copy()

        super().__init__()

        self.enc_b = Encoder(
            self.in_channel, self.num_hidden_channels, self.n_res_block,
            self.num_residual_channels,
            resolution_factor=self.resolution_factors['bottom'],
            groups=self.groups)
        self.enc_t = Encoder(
            self.num_hidden_channels, self.num_hidden_channels,
            self.n_res_block, self.num_residual_channels,
            resolution_factor=self.resolution_factors['top'],
            groups=self.groups)
        try:
            self.n_embed_t, self.n_embed_b = self.num_embeddings
        except TypeError:
            # provided `num_embeddings` is no tuple, assumed to be a single int
            # use this same value for both code layers
            self.n_embed_t, self.n_embed_b = [self.num_embeddings] * 2
        self.quantize_conv_t = nn.Conv2d(self.num_hidden_channels,
                                         self.embed_dim, 1)
        self.quantize_t = Quantize(
            self.embed_dim, self.n_embed_t, decay=self.decay,
            corruption_weights=self.corruption_weights['top'],
            embeddings_initial_variance=self.embeddings_initial_variance)
        self.dec_t = Decoder(
            self.embed_dim, self.embed_dim, self.num_hidden_channels,
            self.n_res_block, self.num_residual_channels, groups=self.groups,
            resolution_factor=self.resolution_factors['top']
        )
        self.quantize_conv_b = nn.Conv2d(
            self.embed_dim + self.num_hidden_channels,
            self.embed_dim, 1)
        self.quantize_b = Quantize(
            self.embed_dim, self.n_embed_b, decay=self.decay,
            corruption_weights=self.corruption_weights['bottom'],
            embeddings_initial_variance=self.embeddings_initial_variance)

        # upsample from 'top' layer back to 'bottom' layer resolution
        upsampling_layers = []
        num_upsampling_layers = int(np.log2(self.resolution_factors['top']))
        for i in range(num_upsampling_layers):
            upsampling_layers.append(
                nn.ConvTranspose2d(
                    self.embed_dim, self.embed_dim, kernel_size=4,
                    stride=2, padding=1)
            )
        self.upsample_top_to_bottom = nn.Sequential(*upsampling_layers)

        self.dec = Decoder(
            self.embed_dim + self.embed_dim,
            self.in_channel,
            self.num_hidden_channels,
            self.n_res_block,
            self.num_residual_channels,
            resolution_factor=self.resolution_factors['bottom'],
            output_activation=self.decoder_output_activation,
            groups=self.groups
        )

        self.use_gansynth_normalization = (
            self.dataloader_for_gansynth_normalization is not None
            or self.normalizer_statistics is not None)
        self.dataloader = self.dataloader_for_gansynth_normalization
        self.normalizer_statistics = self.normalizer_statistics
        if self.normalizer_statistics:
            self.data_normalizer = DataNormalizer(**self.normalizer_statistics)
        elif self.dataloader:
            self.data_normalizer = DataNormalizer(self.dataloader)

    def forward(self, input):
        if self.use_gansynth_normalization:
            input = self.data_normalizer.normalize(input)

        quant_t, quant_b, diff, id_t, id_b, perplexity_t, perplexity_b = self.encode(
            input)
        dec = self.decode(quant_t, quant_b)

        if self.use_gansynth_normalization:
            dec = self.data_normalizer.denormalize(dec)
        return dec, diff, perplexity_t, perplexity_b, id_t, id_b

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t, perplexity_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b, perplexity_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return (quant_t, quant_b, diff_t + diff_b, id_t, id_b,
                perplexity_t, perplexity_b)

    def decode(self, quant_t, quant_b):
        upsample_top_to_bottom = self.upsample_top_to_bottom(quant_t)
        quant = torch.cat([upsample_top_to_bottom, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

    @classmethod
    def from_parameters_and_weights(
            cls, parameters_json_path: pathlib.Path,
            model_weights_checkpoint_path: pathlib.Path,
            device: Union[str, torch.device] = 'cpu') -> 'VQVAE':
        """Re-instantiate a stored model using init parameters and weights

        Arguments:
            parameters_json_path (pathlib.Path)
                Path to the a json file containing the keyword arguments used
                to initialize the object
            model_weights_checkpoint_path (pathlib.Path)
                Path to a model weights checkpoint file as created by
                torch.save
            device (str or torch.device, default 'cpu')
                Device on which to load the stored weights
        """
        with open(parameters_json_path, 'r') as f:
            parameters = json.load(f)
            vqvae = cls(**parameters)

        model_state_dict = torch.load(model_weights_checkpoint_path,
                                      map_location=device)
        if 'model' in model_state_dict.keys():
            model_state_dict = model_state_dict['model']
        vqvae.load_state_dict(model_state_dict)
        return vqvae

    def store_instantiation_parameters(self, path: pathlib.Path) -> None:
        """Store the parameters used to create this instance as JSON"""
        with open(path, 'w') as f:
            json.dump(self._instantiation_parameters, f)


class InferenceVQVAE(object):
    def __init__(self, vqvae: VQVAE, device: str, hop_length: int, n_fft: int):
        self.vqvae = vqvae
        self.device = device
        self.hop_length = hop_length
        self.n_fft = n_fft

    def sample_reconstructions(self, dataloader):
        self.vqvae.eval()

        iterator = iter(dataloader)
        mag_and_IF_batch, _ = next(iterator)
        with torch.no_grad():
            reconstructed_mag_and_IF_batch, *_, id_t, id_b = (
                self.vqvae.forward(
                    mag_and_IF_batch.to(self.device)))

        return mag_and_IF_batch, reconstructed_mag_and_IF_batch, id_t, id_b

    def mag_and_IF_to_audio(self, mag_and_IF: torch.Tensor,
                            use_mel_frequency: bool = True) -> torch.Tensor:
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
                             .mel_logmag_and_IF_to_audio)
        else:
            spec_to_audio = (GANsynth_pytorch.spectrograms_helper
                             .logmag_and_IF_to_audio)

        channel_dimension = 1
        mag_batch = mag_and_IF.select(channel_dimension, 0
                                      )
        IF_batch = mag_and_IF.select(channel_dimension, 1
                                     )

        audio_batch = spec_to_audio(mag_batch, IF_batch,
                                    hop_length=self.hop_length,
                                    n_fft=self.n_fft)

        return audio_batch
