from typing import Optional, List
import numpy as np
import torch
import h5py
import torch.utils.data as udata
from torchvision.datasets import DatasetFolder
from h5torch import HDF5Dataset


class NSynthDataset(DatasetFolder):
    """A simple hdf5-based dataloader for pre-computed Nsynth representations

    The representations are shaped as images for use with Conv2D layers
    
    Taken from https://github.com/ss12f32v/GANsynth-pytorch/
    """
    def __init__(self, root_path: str, transform=None,
                 use_mel_frequency_scale: bool = True):
        self.use_mel_frequency_scale = use_mel_frequency_scale
        self.data_prefix = 'mel_' if self.use_mel_frequency_scale else ''
        super().__init__(root_path, loader=self.loader, extensions='.h5',
                         transform=transform)

    @staticmethod
    def _to_image(*channels_np: List[np.ndarray]):
        """Reshape data into nn.Conv2D-compatible image shape"""
        channel_dimension = 0
        channels = []
        for data_array in channels_np:
            data_tensor = torch.as_tensor(data_array, dtype=torch.float32)
            data_tensor_as_image_channel = data_tensor.unsqueeze(
                channel_dimension)
            channels.append(data_tensor_as_image_channel)

        return torch.cat(channels, channel_dimension)
    
    def loader(self, file_path):
        with h5py.File(file_path, 'r') as sample_file:
            channel_arrays = []
            for channel_name in ['Spec', 'IF']:
                full_channel_name = self.data_prefix + channel_name
                channel_arrays.append(sample_file[full_channel_name][()])
            sample = self._to_image(*channel_arrays)
            
            pitch = sample_file.attrs['pitch']
            pitch_tensor = torch.LongTensor([pitch])

        return (sample, pitch_tensor)