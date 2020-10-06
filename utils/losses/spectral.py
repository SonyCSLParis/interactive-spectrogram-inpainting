from typing import (Iterable, Optional, List, Union)
import math

import torch
from torch import nn
from torch.nn.modules.loss import _Loss as Loss

from GANsynth_pytorch.spectrograms_helper import SpectrogramsHelper


class MultiscaleSpectralLoss(Loss):
    """
    Compute the FFT of a signal at multiple scales

    Expression and default parameters taken from Magenta's DDSP

    Arguments:
            n_ffts (Iterable[int], optional,
                    default = [64, 128, 256, 512, 1024, 2048]):
                list of the multiple n_ffts used for the analysis
            window_lengths (Iterable[int], optional, default = None):
                list of the multiple window lengths used for the analysis,
                if provided, must have as many elements as n_ffts,
                if not provided, defaults to the values in n_ffts
            overlap_ratio (float, optional, default = 0.75):
                STFT window overlap ratio
            loss (torch.nn.modules.loss._Loss, default nn.L1Loss):
                loss used to compute the linear and logarithmic losses
                at each scale (DDSP uses the L1, Jukebox uses the MSE)
            log_loss_alpha (float, optional, default = 1.):
                scale factor between the linear and logarithmic losses
                (DDSP uses 1., Jukebox uses 0.)
            spectrograms_helper (SpectrogramsHelper, optional,
                    default is None):
                if not None, expects Spec and IF representations as input and
                first converts those to audio before computing the
                multiscale STFTs
    """
    def __init__(self, n_ffts: Iterable[int] = [64, 128, 256, 512, 1024, 2048],
                 window_lengths: Optional[Iterable[int]] = None,
                 overlap_ratio: float = 0.75,
                 loss: Loss = nn.L1Loss(),
                 lin_loss_alpha: float = 1.,
                 log_loss_alpha: float = 1.,
                 ):
        super().__init__()
        self.n_ffts = n_ffts
        if window_lengths is not None:
            assert len(window_lengths) == len(n_ffts)
            self.window_lengths = window_lengths
        else:
            self.window_lengths = self.n_ffts
        self.overlap_ratio = overlap_ratio
        self.loss = loss

        # pre-generate and store the windows
        # ensures that they are located on the appropriate device
        self.windows = nn.ParameterList(
            nn.Parameter(
                torch.hann_window(window_length).float(), requires_grad=False)
            for window_length in self.window_lengths)

        assert lin_loss_alpha >= 0. and log_loss_alpha >= 0., (
            "Loss ratios must be non-negative"
        )
        assert not(lin_loss_alpha == 0 and log_loss_alpha == 0.), (
            "Loss will always return 0!")
        self.lin_loss_alpha = lin_loss_alpha
        self.log_loss_alpha = log_loss_alpha
        self.safelog_eps = 1e-6

    @staticmethod
    def magnitude(t: torch.Tensor) -> torch.Tensor:
        """Magnitude for complex numbers in cartesian form"""
        return t.norm(2, dim=-1)

    def forward(self, audio_pred: torch.Tensor, audio_target: torch.Tensor
                ) -> torch.Tensor:
        lin_losses = []
        log_losses = []
        for (n_fft, window_length, window) in zip(self.n_ffts,
                                                  self.window_lengths,
                                                  self.windows):
            hop_length = math.ceil((1-self.overlap_ratio) * window_length)

            spec_mag_pred, spec_mag_target = (
                self.magnitude(torch.stft(
                    audio, n_fft=n_fft, hop_length=hop_length,
                    win_length=window_length,
                    window=window, center=False))
                for audio in (audio_pred, audio_target)
            )

            if self.lin_loss_alpha > 0:
                # compute this only if it will actually be used
                lin_losses.append(self.loss(spec_mag_pred,
                                            spec_mag_target))
            # "perceptual" logarithmic loss
            if self.log_loss_alpha > 0:
                # compute this only if it will actually be used
                log_losses.append(self.loss(
                    torch.log(spec_mag_pred + self.safelog_eps),
                    torch.log(spec_mag_target + self.safelog_eps)))

        def mean(tensors: List[torch.Tensor]) -> Union[float, torch.Tensor]:
            if len(tensors) > 0:
                return sum(tensors) / len(tensors)
            else:
                return 0

        return (self.lin_loss_alpha * mean(lin_losses)
                + self.log_loss_alpha * mean(log_losses)).mean()


class MultiscaleSpectralLoss_fromSpectrogram(MultiscaleSpectralLoss):
    def __init__(self, spectrograms_helper: SpectrogramsHelper,
                 **kwargs):
        super().__init__(**kwargs)
        self.spectrograms_helper = spectrograms_helper

    def forward(self, spec_input: torch.Tensor, spec_target: torch.Tensor
                ) -> torch.Tensor:
        audio_input = self.spectrograms_helper.to_audio(spec_input)
        audio_target = self.spectrograms_helper.to_audio(spec_target)
        return super().forward(audio_input, audio_target)


# as used in Magenta's DDSP paper
DDSPMultiscaleSpectralLoss_kwargs = dict(
    n_ffts=[64, 128, 256, 512, 1024, 2048],
    window_lengths=None,
    overlap_ratio=0.75,
    loss=torch.nn.L1Loss(),
    log_loss_alpha=1.
)


class DDSPMultiscaleSpectralLoss_fromSpectrogram(
        MultiscaleSpectralLoss_fromSpectrogram):
    def __init__(self, spectrograms_helper: SpectrogramsHelper):
        super().__init__(spectrograms_helper,
                         **DDSPMultiscaleSpectralLoss_kwargs)


class L2Loss(nn.Module):
    """Stable L2Loss"""
    def forward(self, x_pred, x_target):
        difference = x_target - x_pred
        difference_norm = (difference
                           .view(difference.shape[0], -1)
                           .norm(2, dim=-1))
        return difference_norm


# as used in the Jukebox paper by OpenAI
JukeboxMultiscaleSpectralLoss_kwargs = dict(
    n_ffts=[2048, 1024, 512],
    window_lengths=[1200, 600, 240],
    overlap_ratio=0.80,
    loss=nn.MSELoss(),
    log_loss_alpha=0.
)


# as used in the Jukebox paper by OpenAI
class JukeboxMultiscaleSpectralLoss_fromSpectrogram(
        MultiscaleSpectralLoss_fromSpectrogram):
    def __init__(self, spectrograms_helper: SpectrogramsHelper):
        super().__init__(spectrograms_helper,
                         **JukeboxMultiscaleSpectralLoss_kwargs)
