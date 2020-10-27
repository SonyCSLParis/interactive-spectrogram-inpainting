import pathlib
from typing import Iterable, Mapping, Optional, Tuple, Union, overload
import json

from torch import Tensor
from torch import nn

from fastai.vision.models import unet, xresnet
from fastai.layers import (BatchNorm, ConvLayer, SequentialEx,
                           PixelShuffle_ICNR, SigmoidRange)
import fastai.layers
from fastai.torch_core import apply_init, defaults
from fastai.torch_core import Module as FastAIModule
from fastai.callback.hook import model_sizes, dummy_eval
from fastai.vision.models.unet import _get_sz_change_idxs


class RosinalityResBlock(nn.Module):
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


class RosinalityEncoder(nn.Module):
    def __init__(self, in_channel: int, channel: int, n_res_block: int,
                 n_res_channel: int, resolution_factor: int, groups: int,
                 use_local_kernels: bool):
        super().__init__()
        self.use_local_kernels = use_local_kernels

        downsampling_stride = 2
        if not use_local_kernels:
            # downsampling using overlapping kernels
            downsampling_kernel_size = 2 * downsampling_stride
        else:
            # downsampling kernels do not overlap
            downsampling_kernel_size = downsampling_stride
        # downsampling module
        if resolution_factor == 16:
            blocks = [
                nn.Conv2d(in_channel, channel // 4,
                          downsampling_kernel_size,
                          stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, 3 * channel // 4,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(3 * channel // 4, channel,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1, groups=groups),
            ]
        elif resolution_factor == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channel // 2, channel,
                    downsampling_kernel_size, stride=downsampling_stride,
                    padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1, groups=groups),
            ]
        elif resolution_factor == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1,
                          groups=groups),
            ]
        elif resolution_factor == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1, groups=groups),
            ]
        else:
            raise ValueError(
                f"Unexpected resolution factor {resolution_factor}")

        for i in range(n_res_block):
            blocks.append(RosinalityResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class RosinalityDecoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, channel: int,
                 n_res_block: int, n_res_channel: int, resolution_factor: int,
                 groups: int, use_local_kernels: bool,
                 output_activation: Optional[nn.Module] = None
                 ):
        super().__init__()
        self.use_local_kernels = use_local_kernels

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(RosinalityResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        upsampling_stride = 2
        if not use_local_kernels:
            # upsampling using overlapping kernels
            upsampling_kernel_size = 2 * upsampling_stride
        else:
            # upsampling kernels do not overlap
            upsampling_kernel_size = upsampling_stride
        # upsampling module
        if resolution_factor == 16:
            blocks.extend(
                [
                    nn.ConvTranspose2d(
                        channel, 3 * channel // 4,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        3 * channel // 4, channel // 2,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, channel // 4,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 4, out_channel,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups)
                ]
            )
        elif resolution_factor == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2,
                                       upsampling_kernel_size,
                                       stride=upsampling_stride,
                                       padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, channel // 2,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups)
                ]
            )
        elif resolution_factor == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2,
                                       upsampling_kernel_size,
                                       stride=upsampling_stride,
                                       padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                ]
            )
        elif resolution_factor == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel,
                                   upsampling_kernel_size,
                                   stride=upsampling_stride,
                                   padding=1, groups=groups)
            )
        else:
            raise ValueError(
                f"Unexpected resolution factor {resolution_factor}")

        if output_activation is not None:
            blocks.append(output_activation)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class NoSkipUnetBlock(unet.UnetBlock):
    """A U-Net upsampling block with no skip connection from the encoder"""
    def __init__(self, up_in_c: int, **kwargs):
        hook = None
        x_in_c = 0
        super().__init__(up_in_c, x_in_c, hook, **kwargs)

    def forward(self, up_in: Tensor) -> Tensor:
        up_out = self.shuf(up_in)
        cat_x = self.relu(up_out)
        return self.conv2(self.conv1(cat_x))


# HACK(theis)
class NoSkipDynamicUnet(SequentialEx):
    """Create a U-Net from a given architecture.

    This subclass removes all skip-connections from encoder to decoder.
    """
    def __init__(self, encoder, n_classes, img_size, blur=False,
                 blur_final=True, self_attention=False,
                 y_range=None, bottle=False,
                 act_cls=defaults.activation,
                 init=nn.init.kaiming_normal_, norm_type=None,
                 include_encoder=True,
                 include_middle_conv=True,
                 **kwargs):
        imsize = img_size
        sizes = model_sizes(encoder, size=imsize)
        sz_chg_idxs = list(reversed(_get_sz_change_idxs(sizes)))
        # self.sfs = hook_outputs([encoder[i] for i in sz_chg_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        layers = []
        if include_encoder:
            layers.append(encoder)

        if include_middle_conv:
            ni = sizes[-1][1]
            middle_conv = (nn.Sequential(ConvLayer(ni, ni*2, act_cls=act_cls,
                                                   norm_type=norm_type,
                                                   **kwargs),
                                         ConvLayer(ni*2, ni, act_cls=act_cls,
                                                   norm_type=norm_type,
                                                   **kwargs))
                           ).eval()
            x = middle_conv(x)
            layers += [BatchNorm(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sz_chg_idxs):
            not_final = (i != len(sz_chg_idxs)-1)
            up_in_c = int(x.shape[1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sz_chg_idxs)-3)
            noskip_unet_block = NoSkipUnetBlock(
                up_in_c, final_div=not_final,
                blur=do_blur, self_attention=sa,
                act_cls=act_cls, init=init, norm_type=norm_type,
                **kwargs
                ).eval()
            layers.append(noskip_unet_block)
            x = noskip_unet_block(x)

        ni = x.shape[1]
        if imsize != sizes[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, act_cls=act_cls,
                                            norm_type=norm_type))

        layers += [ConvLayer(ni, n_classes, ks=1, act_cls=None,
                             norm_type=norm_type, **kwargs)]

        if include_middle_conv:
            apply_init(nn.Sequential(layers[3], layers[-2]), init)
            apply_init(nn.Sequential(layers[2]), init)

        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)


class NoFlattenXResNet(xresnet.XResNet):
    @xresnet.delegates(fastai.layers.ResBlock)
    def __init__(self, block, expansion: int, layers: Iterable[int],
                 p: float = 0.0, c_in: int = 3, n_out: int = 1000,
                 stem_szs: Iterable[int] = (32, 32, 64),
                 widen: float = 1.0, sa: bool = False,
                 act_cls=defaults.activation, ndim: int = 2,
                 ks: int = 3, stride: int = 2,
                 **kwargs):
        xresnet.store_attr('block,expansion,act_cls,ndim,ks')
        if ks % 2 == 0:
            raise Exception('kernel size has to be odd!')
        stem_szs = [c_in, *stem_szs]
        stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=ks,
                          stride=stride if i == 0 else 1,
                          act_cls=act_cls, ndim=ndim)
                for i in range(3)]

        block_szs = [int(o * widen)
                     for o in [64, 128, 256, 512] + [256]*(len(layers)-4)]
        block_szs = block_szs[:len(layers)]
        block_szs = [64 // expansion] + block_szs
        blocks = self._make_blocks(layers, block_szs, sa, stride, **kwargs)

        super(xresnet.XResNet, self).__init__(
            *stem,
            # MaxPool(ks=ks, stride=stride, padding=ks//2, ndim=ndim),
            *blocks,
            ConvLayer(block_szs[-1]*expansion, n_out, stride=1,
                      act_cls=act_cls, ndim=ndim),
        )
        xresnet.init_cnn(self)


def get_xresnet_unet(in_channels: int,
                     image_size: Tuple[int, int],
                     downsampling_factors: Mapping[str, int],
                     hidden_channels: int,
                     embeddings_dimension: int,
                     layers_per_downsampling_block: Union[int, Mapping[str, int]],
                     expansion: int,
                     block: Union[nn.Module, FastAIModule] = fastai.layers.ResBlock,
                     ):
    """Return a U-Net-like XResNet-based encoder/decoder pair for use in a VQ-VAE

    Arguments:
        * in_channels (int): number of channels in the input images
        * image_size (Tuple[int, int]): input images shape, in (H, W) format
        * resolution_factors (Mapping[str, int]): Total downsampling amount
            per VQ-VAE layer
        * hidden_channels (int): number of hidden channels throughout the model
        * layers_per_downsampling_block (int):
            number of successive residual layers for a full downsampling block
            of a factor 2
        * block (torch.nn.Module):
    """
    import math
    # common parameters for both encoders
    encoders_kwargs = dict(
        ndim=2,
        stride=2
    )
    # each ResNet layers of  downsamples by 2
    num_blocks_b = int(math.log2(downsampling_factors['bottom']))
    encoder_b = NoFlattenXResNet(
        block,
        expansion,
        [layers_per_downsampling_block] * (num_blocks_b),
        c_in=in_channels,
        n_out=hidden_channels,
        **encoders_kwargs)

    num_blocks_t = int(math.log2(downsampling_factors['top']))
    encoder_t = NoFlattenXResNet(
        block,
        expansion,
        [layers_per_downsampling_block] * num_blocks_t,
        c_in=hidden_channels,
        n_out=hidden_channels,
        **encoders_kwargs)

    encoders = {'top': encoder_t,
                'bottom': encoder_b}

    # common parameters for both decoders
    decoders_kwargs = dict(
        # only return the upsampling-decoder half of the U-Net
        include_encoder=False,
        include_middle_conv=False
    )

    bottom_resolution = [n // downsampling_factors['bottom']
                         for n in image_size]
    decoder_t = NoSkipDynamicUnet(
        nn.Sequential(*list(encoder_t.children()),
                      nn.Conv2d(
                          hidden_channels,
                          embeddings_dimension,
                          1,
                          stride=1)
                      ),
        embeddings_dimension,
        bottom_resolution,
        **decoders_kwargs)
    decoder_b = NoSkipDynamicUnet(
        nn.Sequential(*list(encoder_b.children()),
                      nn.Conv2d(
                          hidden_channels,
                          2 * embeddings_dimension,
                          1,
                          stride=1)
                      ),
        in_channels,
        image_size,
        **decoders_kwargs)
    decoders = {'top': decoder_t,
                'bottom': decoder_b}
    return encoders, decoders


def xresnet_unet_from_json_parameters(
        in_channels: int,
        image_size: Tuple[int, int],
        parameters_json_path: pathlib.Path):
    with open(parameters_json_path, 'r') as f:
        command_line_parameters = json.load(f)

    assert command_line_parameters['use_resnet']
    encoders, decoders = get_xresnet_unet(
        in_channels,
        image_size,
        command_line_parameters['resolution_factors'],
        hidden_channels=command_line_parameters['num_hidden_channels'],
        embeddings_dimension=command_line_parameters['embeddings_dimension'],
        layers_per_downsampling_block=command_line_parameters['resnet_layers_per_downsampling_block'],
        expansion=command_line_parameters['resnet_expansion'],
    )
    return encoders, decoders
