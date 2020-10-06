
from typing import Union
import pathlib

from GANsynth_pytorch.spectrograms_helper import (
    SpectrogramsHelper, MelSpectrogramsHelper
)


def get_spectrograms_helper(device: str = 'cuda', **kwargs
                            ) -> SpectrogramsHelper:
    """Return a SpectrogramsHelper instance using the provided parameters"""
    spectrogram_parameters = {
        'fs_hz': kwargs['fs_hz'],
        'n_fft': kwargs['n_fft'],
        'hop_length': kwargs['hop_length'],
        'window_length': kwargs['window_length'],
        'device': device,
    }
    if kwargs['use_mel_scale']:
        return MelSpectrogramsHelper(
            **spectrogram_parameters,
            lower_edge_hertz=kwargs['mel_scale_lower_edge_hertz'],
            upper_edge_hertz=kwargs['mel_scale_upper_edge_hertz'],
            mel_break_frequency_hertz=kwargs['mel_scale_break_frequency_hertz'],
            mel_bin_width_threshold_factor=(
                kwargs['mel_scale_expand_resolution_factor'])
        )
    else:
        return SpectrogramsHelper(**spectrogram_parameters)


def expand_path(p: Union[str, pathlib.Path]) -> pathlib.Path:
    return pathlib.Path(p).expanduser().absolute()