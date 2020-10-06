from typing import Optional, Iterable, Union, List, Mapping, Tuple
import numpy as np
import pathlib
import json

import torch
from torch import nn
from torch import Tensor, FloatTensor, LongTensor

from GANsynth_pytorch.loader import make_masked_phase_transform
from GANsynth_pytorch.normalizer import (DataNormalizer,
                                         DataNormalizerStatistics)

from .encoder_decoder import RosinalityEncoder, RosinalityDecoder
from .bottleneck import QuantizedBottleneck, UnquantizedBottleneck


class BiasedNonLinearity(nn.Module):
    def __init__(self, activation: nn.Module, bias: float,
                 channel_index: int = 0):
        super().__init__()
        self.activation = activation
        self.bias = bias
        self.channel_index = channel_index

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # retrieve channel
        channel = input[:, self.channel_index]
        channel = self.bias + self.non_linearity(channel)
        input[:, self.channel_index] = channel
        return input


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
            code layer, e.g. for a two-level VQ-VAE, with top obtained
            by further downsampling bottom, the resolution_factors['top']
            is the down/up-sampling factor from the bottom layer to the top
            layer.
    """
    def __init__(
        self,
        encoders: Optional[Mapping[str, nn.Module]] = None,
        decoders: Optional[Mapping[str, nn.Module]] = None,
        in_channel: int = 3,
        num_hidden_channels: int = 128,
        n_res_block: int = 2,
        num_residual_channels: int = 32,
        embed_dim: int = 64,
        num_embeddings: Union[int, Iterable[int]] = 512,
        decay: float = 0.99,
        groups: int = 1,
        use_local_kernels: bool = False,
        output_activation_type: Optional[str] = None,
        output_spectrogram_min_magnitude: Optional[float] = None,
        resolution_factors: Mapping[str, int] = {
            'bottom': 4,
            'top': 2,
        },
        embeddings_initial_variance: float = 1,
        # TODO(theis): remove this parameter, hard to serialize and save
        decoder_output_activation: Optional[nn.Module] = None,
        normalizer_statistics: Optional[
            Union[DataNormalizerStatistics, Mapping[str, float]]] = None,
        corruption_weights: Mapping[str, Optional[List[float]]] = {'top': None,
                                                                   'bottom': None},
        adapt_quantized_durations: bool = True,
        disable_quantization: bool = False
    ):
        if decoder_output_activation is not None:
            raise NotImplementedError("TODO")

        # store instantiation parameters
        self.in_channel = in_channel
        self.num_hidden_channels = num_hidden_channels
        self.n_res_block = n_res_block
        self.num_residual_channels = num_residual_channels
        self.embed_dim = embed_dim
        self.use_local_kernels = use_local_kernels
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.groups = groups
        self.resolution_factors = resolution_factors
        self.embeddings_initial_variance = embeddings_initial_variance
        self.output_activation_type = output_activation_type
        self.decoder_output_activation = decoder_output_activation
        self.corruption_weights = corruption_weights
        self.output_spectrogram_min_magnitude = (
            output_spectrogram_min_magnitude)

        if isinstance(normalizer_statistics, DataNormalizerStatistics):
            self.normalizer_statistics = normalizer_statistics.__dict__
        else:
            self.normalizer_statistics = normalizer_statistics

        self._instantiation_parameters = self.__dict__.copy()

        super().__init__()

        if encoders is None:
            self.enc_b = RosinalityEncoder(
                self.in_channel, self.num_hidden_channels, self.n_res_block,
                self.num_residual_channels,
                resolution_factor=self.resolution_factors['bottom'],
                groups=self.groups,
                use_local_kernels=self.use_local_kernels)
            self.enc_t = RosinalityEncoder(
                self.num_hidden_channels, self.num_hidden_channels,
                self.n_res_block, self.num_residual_channels,
                resolution_factor=self.resolution_factors['top'],
                groups=self.groups,
                use_local_kernels=self.use_local_kernels)
        else:
            self.enc_t = encoders['top']
            self.enc_b = encoders['bottom']
        try:
            self.n_embed_t, self.n_embed_b = self.num_embeddings
        except TypeError:
            # provided `num_embeddings` is no tuple, assumed to be a single int
            # use this same value for both code layers
            self.n_embed_t, self.n_embed_b = [self.num_embeddings] * 2
        self.quantize_conv_t = nn.Conv2d(self.num_hidden_channels,
                                         self.embed_dim, 1)

        if not disable_quantization:
            bottleneck = QuantizedBottleneck
        else:
            bottleneck = UnquantizedBottleneck

        self.quantize_t = bottleneck(
            self.embed_dim, self.n_embed_t, decay=self.decay,
            corruption_weights=self.corruption_weights['top'],
            embeddings_initial_variance=self.embeddings_initial_variance)
        if decoders is None:
            self.dec_t = RosinalityDecoder(
                self.embed_dim, self.embed_dim, self.num_hidden_channels,
                self.n_res_block, self.num_residual_channels, groups=self.groups,
                resolution_factor=self.resolution_factors['top'],
                use_local_kernels=self.use_local_kernels,
            )
        else:
            self.dec_t = decoders['top']
        self.quantize_conv_b = nn.Conv2d(
            self.embed_dim + self.num_hidden_channels,
            self.embed_dim, 1)
        self.quantize_b = bottleneck(
            self.embed_dim, self.n_embed_b, decay=self.decay,
            corruption_weights=self.corruption_weights['bottom'],
            embeddings_initial_variance=self.embeddings_initial_variance)

        # upsample from 'top' layer back to 'bottom' layer resolution
        upsampling_layers = []
        num_upsampling_layers = int(np.log2(self.resolution_factors['top']))

        upsampling_stride = 2
        if not self.use_local_kernels:
            upsampling_kernel_size = upsampling_stride * 2
        else:
            upsampling_kernel_size = upsampling_stride

        for i in range(num_upsampling_layers):
            upsampling_layers.append(
                nn.ConvTranspose2d(
                    self.embed_dim, self.embed_dim,
                    kernel_size=upsampling_kernel_size,
                    stride=upsampling_stride,
                    padding=1)
            )
        self.upsample_top_to_bottom = nn.Sequential(*upsampling_layers)

        if decoders is None:
            self.dec = RosinalityDecoder(
                self.embed_dim + self.embed_dim,
                self.in_channel,
                self.num_hidden_channels,
                self.n_res_block,
                self.num_residual_channels,
                resolution_factor=self.resolution_factors['bottom'],
                groups=self.groups,
                use_local_kernels=self.use_local_kernels,
                output_activation=self.decoder_output_activation,
            )
        else:
            self.dec = decoders['bottom']

        self.use_gansynth_normalization = (self.normalizer_statistics
                                           is not None)

        if self.use_gansynth_normalization:
            data_normalizer_statistics = DataNormalizerStatistics(
                **self.normalizer_statistics)
            self.data_normalizer = DataNormalizer(data_normalizer_statistics)
        else:
            self.data_normalizer = None

        if self.output_activation_type == 'threshold_gelu':
            self.output_activation = BiasedNonLinearity(
                nn.GELU(), self.output_spectrogram_min_magnitude,
                channel_index=0
            )
        else:
            assert self.output_activation_type is None, (
                "Unexpected output activation type")

        self.output_transform = None
        if self.output_spectrogram_min_magnitude is not None:
            self.output_transform = make_masked_phase_transform(
                self.output_spectrogram_min_magnitude)

        self.adapt_quantized_durations = adapt_quantized_durations

    def forward(self, input):
        quant_t, quant_b, diff, id_t, id_b, perplexity_t, perplexity_b = self.encode(
            input)
        dec = self.decode(quant_t, quant_b)
        return dec, diff, perplexity_t, perplexity_b, id_t, id_b

    def encode(self, input: Tensor) -> Tuple[
            FloatTensor, FloatTensor, FloatTensor,
            LongTensor, LongTensor, FloatTensor, FloatTensor]:
        if self.use_gansynth_normalization:
            input = self.data_normalizer.normalize(input)

        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t, perplexity_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        if self.adapt_quantized_durations:
            quantized_duration = min(dec_t.shape[-1], enc_b.shape[-1])
            dec_t = dec_t[..., :quantized_duration]
            enc_b = enc_b[..., :quantized_duration]
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

        dec = self.post_process(dec)
        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)
        return dec

    def post_process(self, dec: torch.Tensor) -> torch.Tensor:
        if self.use_gansynth_normalization:
            dec = self.data_normalizer.denormalize(dec)
        if self.output_transform is not None:
            dec = self.output_transform(dec)
        return dec

    @classmethod
    def from_parameters_and_weights(
            cls, parameters_json_path: pathlib.Path,
            model_weights_checkpoint_path: pathlib.Path,
            device: Union[str, torch.device] = 'cpu',
            encoders: Optional[Mapping[str, nn.Module]] = None,
            decoders: Optional[Mapping[str, nn.Module]] = None,
            ) -> 'VQVAE':
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
            vqvae = cls(**parameters, encoders=encoders,
                        decoders=decoders)

        model_state_dict = torch.load(model_weights_checkpoint_path,
                                      map_location=device)
        if 'model' in model_state_dict.keys():
            model_state_dict = model_state_dict['model']
        # HACK(theis): uses strict=False to allow loading old models where
        # DataNormalizer was not yet a submodule
        # TODO(theis): remove this eventually
        vqvae.load_state_dict(model_state_dict, strict=False)
        return vqvae

    def store_instantiation_parameters(self, path: pathlib.Path) -> None:
        """Store the parameters used to create this instance as JSON"""
        with open(path, 'w') as f:
            json.dump(self._instantiation_parameters, f, indent=4)
