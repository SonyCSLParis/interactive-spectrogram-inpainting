from typing import Optional
from abc import ABC, abstractmethod

import torch


class CodemapsHelper(ABC):
    """Abstract-class for definining classes to reshape and flatten codemaps"""
    def __init__(self, frequencies, duration):
        self.frequencies = frequencies
        self.duration = duration

        self.predict_frequencies_first = True
        self.predict_low_frequencies_first = True

    def to_time_frequency_map(self, sequence: torch.Tensor,
                              permute_output_as_logits: bool = False
                              ) -> torch.Tensor:
        # assumes frequency-first modeling
        (batch_dim, time_dim, frequency_dim) = (0, 1, 2)
        batch_size = sequence.shape[0]

        if not self.predict_low_frequencies_first:
            raise NotImplementedError

        sequence_dimensions = sequence.dim()
        if sequence_dimensions == 2:
            sequence = sequence.unsqueeze(-1)
            is_logits = False
        elif sequence_dimensions == 3:
            is_logits = True
        else:
            raise ValueError("Unexpected number of dimensions "
                             f"{sequence_dimensions} for input sequence")

        embedding_size = sequence.shape[2]

        output_dimensions = batch_dim, frequency_dim, time_dim, logit_dim = (
            0, frequency_dim, time_dim, 3)

        # subclass-specific data reordering
        sequence = self.reorder_sequence_for_delinearization(sequence)

        output_shape = (batch_size, self.duration, self.frequencies,
                        embedding_size)
        time_frequency_map = sequence.reshape(*output_shape)

        time_frequency_map = time_frequency_map.transpose(
            time_dim, frequency_dim)
        (frequency_dim, time_dim) = (1, 2)

        if is_logits and permute_output_as_logits:
            # permute dimensions to follow PyTorch logits convention
            time_frequency_map = (time_frequency_map.permute(0, 3, 1, 2))
        if not is_logits:
            time_frequency_map = time_frequency_map.squeeze(-1)
        return time_frequency_map

    def to_sequence(self, codemap: torch.Tensor) -> torch.Tensor:
        batch_dim, frequency_dim, time_dim = (0, 1, 2)
        batch_size, frequencies, duration = codemap.shape[:3]

        # push frequencies to innermost dimension for frequency-first modeling
        codemap = codemap.transpose(time_dim, frequency_dim)
        (frequency_dim, time_dim) = (2, 1)

        if not self.predict_low_frequencies_first:
            raise NotImplementedError

        codemap = self.reorder_codemap_for_linearization(codemap)

        flattened_codemap = (
            codemap.reshape(batch_size,
                            frequencies * duration,
                            -1)
            )
        if codemap.ndim == 3:
            flattened_codemap = flattened_codemap.squeeze(-1)
        dimensions = (batch_dim, sequence_dim, embedding_dim) = (0, 1, 2)
        return flattened_codemap

    @abstractmethod
    def reorder_codemap_for_linearization(self, codemap: torch.Tensor
                                          ) -> torch.Tensor:
        """Reordering of codemaps which is performed prior to linearization"""
        raise NotImplementedError("Subclass this")

    @abstractmethod
    def reorder_sequence_for_delinearization(self, sequence: torch.Tensor
                                             ) -> torch.Tensor:
        """Inverse operation of self.reorder_codemap_for_linearization"""
        raise NotImplementedError("Subclass this")


class SimpleCodemapsHelper(CodemapsHelper):
    """Helper class performing simple linear flattening of codemaps"""
    def reorder_codemap_for_linearization(self, codemap: torch.Tensor
                                          ) -> torch.Tensor:
        """SimpleCodemapsHelper performs no reordering"""
        return codemap

    def reorder_sequence_for_delinearization(self, sequence: torch.Tensor
                                             ) -> torch.Tensor:
        """SimpleCodemapsHelper performs no reordering"""
        return sequence


class ZigZagCodemapsHelper(CodemapsHelper):
    """Convert codemaps in a 2D-patch-based fashion

    This reshaping is adapted to upsampling VQ-Spectrogram Transformers,
    where a patch from the upsampled version should be aligned with the single
    code from which it is obtained in the downsampled version.
    """
    def __init__(self, frequencies: int, duration: int,
                 patch_frequencies: int, patch_duration: int):
        super().__init__(frequencies, duration)
        # frequency_slices_size = self.target_frequencies // self.source_frequencies
        self.patch_frequencies = patch_frequencies
        # patch_duration = self.target_duration // self.source_duration
        self.patch_duration = patch_duration

    def reorder_codemap_for_linearization(self, codemap: torch.Tensor
                                          ) -> torch.Tensor:
        """Re-order codemap in a patch-based fashion"""
        # patches of the target should be identified according to
        # the position they occupy below their respective "source" patch

        inserted_dummy_inner_feature_dimension = False
        if codemap.ndim == 3:
            # reshaping operation designed for 4D tensors
            # insert a singletion dim
            codemap = codemap.unsqueeze(-1)
            inserted_dummy_inner_feature_dimension = True

        # slice along the time-axis
        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 2, 1, 3)

        batch_size, _, _, embedding_size = codemap.shape

        codemap = codemap.unfold(
            time_dim,
            self.patch_duration,
            self.patch_duration)

        # slice along the frequency-axis
        codemap = codemap.unfold(
            frequency_dim,
            self.patch_frequencies,
            self.patch_frequencies)

        # push back the embedding dimension as innermost dimension
        codemap = codemap.permute(
            batch_dim, time_dim, frequency_dim, 4, 5, embedding_dim)

        codemap = codemap.reshape(
            batch_size, self.duration, self.frequencies,
            embedding_size)

        if inserted_dummy_inner_feature_dimension:
            # remove the dummy feature dimension
            codemap = codemap.squeeze(-1)
        return codemap

    def reorder_sequence_for_delinearization(self, sequence: torch.Tensor
                                             ) -> torch.Tensor:
        """Invert patch-based sequence reordering"""
        sequence_dim: Optional[int] = None
        batch_dim, sequence_dim, embedding_dim = (0, 1, 2)

        batch_size, sequence_size, embedding_size = sequence.shape

        # retrieve the frequency-axis slices
        sequence = sequence.unfold(
            sequence_dim,
            self.patch_frequencies,
            self.patch_frequencies)
        # unfolding introduces a new dimension in innermost position
        patch_frequency_dim = sequence.dim() - 1
        # push-back embedding dimension as innermost dimension
        sequence = sequence.transpose(patch_frequency_dim, embedding_dim)
        patch_frequency_dim, embedding_dim = embedding_dim, patch_frequency_dim

        # retrieve the time-axis slices
        sequence = sequence.unfold(
            sequence_dim,
            self.patch_duration,
            self.patch_duration)
        # unfolding introduces a new dimension in innermost position
        patch_time_dim = sequence.dim() - 1
        # push-back embedding dimension as innermost dimension
        sequence = sequence.transpose(patch_time_dim, embedding_dim)
        patch_time_dim, embedding_dim = embedding_dim, patch_time_dim

        # transpose to index first along the frequency slices
        sequence = sequence.transpose(patch_time_dim, patch_frequency_dim)
        patch_time_dim, patch_frequency_dim = patch_frequency_dim, patch_time_dim

        # retrieve source frequency slices
        source_frequencies = self.frequencies // self.patch_frequencies
        sequence = sequence.unfold(
            sequence_dim,
            source_frequencies,
            source_frequencies
        )
        # unfolding introduces a new dimension in innermost position
        source_channels_dim = sequence.dim() - 1
        # push back the embedding dimension as innermost dimension
        sequence = sequence.transpose(source_channels_dim, embedding_dim)
        source_channels_dim, embedding_dim = embedding_dim, source_channels_dim
        source_events_dim = sequence_dim
        sequence_dim = None

        # tranpose to index first along frequency slices, then along time slices,
        # then along source events
        sequence = sequence.permute(
            batch_dim,
            source_events_dim, source_channels_dim,
            patch_time_dim, patch_frequency_dim,
            embedding_dim)
        (batch_dim, source_events_dim, source_channels_dim, patch_time_dim,
         patch_frequency_dim, embedding_dim) = tuple(range(sequence.dim()))

        # transpose to index in time-frequency order:
        # with time decomposed into: patch time -> source events
        # and frequency decomposed as: patch frequency -> source channel
        # thus the full order is:
        # patch time -> source events -> patch frequency -> source channel
        sequence = sequence.permute(
            batch_dim,
            source_channels_dim, patch_frequency_dim,
            source_events_dim, patch_time_dim,
            embedding_dim)

        codemap = sequence.reshape(
            batch_size, self.frequencies, self.duration,
            embedding_size)

        # convert to frequency-first format
        codemap = codemap.transpose(1, 2)

        sequence = codemap.reshape(batch_size, -1, embedding_size)
        return sequence


