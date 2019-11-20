from typing import Iterable, Mapping, Union, Optional, Sequence
import pathlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class UnconditionalTransformer(nn.Module):
    """Causal, bi-directional transformer"""
    def __init__(
        self,
        shape: Iterable[int],  # [num_frequencies, frame_duration]
        n_class: int,
        channel: int,
        kernel_size: int,
        n_block: int,
        n_res_block: int,
        res_channel: int,
        attention: bool = True,
        dropout: float = 0.1,
        n_cond_res_block: int = 0,
        cond_res_channel: int = 0,
        cond_res_kernel: int = 3,
        n_out_res_block: int = 0,
        predict_frequencies_first: bool = False,
        predict_low_frequencies_first: bool = True,
        d_model: int = 512,
        embeddings_dim: int = 32,
        positional_embeddings_dim: int = 16,
    ):
        self.shape = shape

        self.n_class = n_class
        self.channel = channel

        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
        else:
            self.kernel_size = kernel_size

        self.n_block = n_block
        self.n_res_block = n_res_block
        self.res_channel = res_channel
        self.attention = attention
        self.dropout = dropout
        self.n_cond_res_block = n_cond_res_block
        self.cond_res_channel = cond_res_channel
        self.cond_res_kernel = cond_res_kernel
        self.n_out_res_block = n_out_res_block
        self.predict_frequencies_first = predict_frequencies_first
        self.predict_low_frequencies_first = predict_low_frequencies_first
        self.d_model = d_model
        self.embeddings_dim = embeddings_dim
        # ensure an even value
        self.positional_embeddings_dim = 2 * (positional_embeddings_dim // 2)

        self._instantiation_parameters = self.__dict__.copy()

        super().__init__()

        self.frequencies, self.duration = self.shape
        self.transformer_sequence_length = self.frequencies * self.duration

        self.positional_embeddings_frequency = nn.Parameter(
            torch.randn((1,  # batch-size
                         self.frequencies,  # frequency-dimension
                         1,  # time-dimension
                         self.positional_embeddings_dim//2))
        )

        self.positional_embeddings_time = nn.Parameter(
            torch.randn((1,  # batch-size
                         1,  # frequency-dimension
                         self.duration,  # time-dimension
                         self.positional_embeddings_dim//2))
        )

        if self.embeddings_dim is None:
            self.embeddings_dim = self.d_model-self.positional_embeddings_dim

        self.embeddings_linear = nn.Linear(
            self.embeddings_dim,
            self.d_model-self.positional_embeddings_dim)

        self.embed = torch.nn.Embedding(self.n_class, self.embeddings_dim)

        # convert Transformer outputs to class-probabilities (as logits)
        self.project_transformer_outputs_to_logits = (
            nn.Linear(self.d_model, self.n_class))

        # TODO reduce dimensionality of start symbol
        self.start_symbol = nn.Parameter(
            torch.randn((1, 1, self.d_model))
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    @property
    def combined_positional_embeddings(self) -> torch.Tensor:
        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 1, 2, 3)

        repeated_frequency_embeddings = (
            self.positional_embeddings_frequency
            .repeat(1, 1, self.duration, 1))
        repeated_time_embeddings = (
            self.positional_embeddings_time
            .repeat(1, self.frequencies, 1, 1))

        return torch.cat([repeated_frequency_embeddings,
                          repeated_time_embeddings],
                         dim=embedding_dim)

    @property
    def causal_mask(self) -> torch.Tensor:
        """Generate a mask to impose causality"""
        mask = (torch.triu(torch.ones(self.transformer_sequence_length,
                                      self.transformer_sequence_length)) == 1
                ).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def forward(self, input: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                cache: Optional[Mapping[str, torch.Tensor]] = None):
        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 1, 2, 3)
        batch_size = input.shape[0]

        causal_mask = self.causal_mask

        embedded_input = self.embeddings_linear(self.embed(input))

        # add positional embeddings
        # combine time and frequency embeddings
        positional_embeddings = self.combined_positional_embeddings
        input_with_positions = torch.cat(
            [embedded_input, positional_embeddings.repeat(batch_size, 1, 1, 1)],
            dim=embedding_dim
        )

        if self.predict_frequencies_first:
            input_with_positions = input_with_positions.transpose(
                time_dim, frequency_dim)
            (frequency_dim, time_dim) = (2, 1)

        flattened_input_with_positions = (
            input_with_positions.reshape(batch_size,
                                         self.transformer_sequence_length,
                                         self.d_model)
        )
        (batch_dim, sequence_dim, embedding_dim) = (0, 1, 2)

        # Shift inputs
        # we do this so that the output of the transformer can be readily
        # interpreted as the probability of generating each possible output
        # at that position
        shifted_sequence_with_positions = torch.cat(
            [self.start_symbol.repeat(batch_size, 1, 1),
             flattened_input_with_positions.narrow(
                 sequence_dim,
                 0,
                 self.transformer_sequence_length-1)],
            dim=sequence_dim
        )

        # nn.Transformer inputs are in time-major shape
        transformer_input_sequence = shifted_sequence_with_positions.transpose(
            batch_dim, sequence_dim)
        (batch_dim, sequence_dim) = (1, 0)
        output_sequence = self.transformer(transformer_input_sequence,
                                           mask=causal_mask)
        # transpose back to batch-major shape
        output_sequence = output_sequence.transpose(
            batch_dim, sequence_dim)
        (batch_dim, sequence_dim) = (0, 1)

        # convert outputs to class probabilities
        logits = self.project_transformer_outputs_to_logits(output_sequence)

        output_dimensions = batch_dim, frequency_dim, time_dim, logit_dim = (
            0, frequency_dim, time_dim, 3)
        output_sizes = (batch_size, self.frequencies, self.duration,
                        self.n_class)

        output_shape = [None] * len(output_sizes)
        for dim_index, size in zip(output_dimensions, output_sizes):
            output_shape[dim_index] = size

        # reshape output to time-frequency format
        time_frequency_logits = output_sequence.reshape(*output_shape)

        if self.predict_frequencies_first:
            time_frequency_logits = time_frequency_logits.transpose(
                time_dim, frequency_dim)
            (frequency_dim, time_dim) = (1, 2)

        time_frequency_logits = (time_frequency_logits
                                 .permute(0, 3, 1, 2))
        return time_frequency_logits, None

    @classmethod
    def from_parameters_and_weights(
            cls, parameters_json_path: pathlib.Path,
            model_weights_checkpoint_path: pathlib.Path,
            device: Union[str, torch.device] = 'cpu') -> 'PixelSNAIL':
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
            model = cls(**parameters)

        model_state_dict = torch.load(model_weights_checkpoint_path,
                                      map_location=device)
        if 'model' in model_state_dict.keys():
            model_state_dict = model_state_dict['model']
        model.load_state_dict(model_state_dict)
        return model

    def store_instantiation_parameters(self, path: pathlib.Path) -> None:
        """Store the parameters used to create this instance as JSON"""
        with open(path, 'w') as f:
            json.dump(self._instantiation_parameters, f)


class ConditionalTransformer(nn.Module):
    """Transformer-based generative model for latent maps

    Inputs are expected to be of shape [num_frequency_bands, time_duration],
    with sample[0, 0] in a image the energy in the lowest frequency band
    in the first time-frame.

    Arguments:
        * predict_frequencies_first (bool, optional, default is False):
            if True, transposes the inputs to predict time-frame by time-frame,
            potentially bringing more coherence in frequency within a given
            time-frame by bringing the dependencies closer.
        * predict_low_frequencies_first (bool, optional, default is True):
            if False, flips the inputs so that frequencies are predicted in
            decreasing order, starting at the harmonics.
    """
    def __init__(
        self,
        shape: Iterable[int],
        n_class: int,
        channel: int,
        kernel_size: int,
        n_block: int,
        n_res_block: int,
        res_channel: int,
        attention: bool = True,
        dropout: float = 0.1,
        n_cond_res_block: int = 0,
        cond_res_channel: int = 0,
        cond_res_kernel: int = 3,
        n_out_res_block: int = 0,
        predict_frequencies_first: bool = False,
        predict_low_frequencies_first: bool = True
    ):
        self.shape = shape

        self.n_class = n_class
        self.channel = channel

        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
        else:
            self.kernel_size = kernel_size

        self.n_block = n_block
        self.n_res_block = n_res_block
        self.res_channel = res_channel
        self.attention = attention
        self.dropout = dropout
        self.n_cond_res_block = n_cond_res_block
        self.cond_res_channel = cond_res_channel
        self.cond_res_kernel = cond_res_kernel
        self.n_out_res_block = n_out_res_block
        self.predict_frequencies_first = predict_frequencies_first
        self.predict_low_frequencies_first = predict_low_frequencies_first

        self._instantiation_parameters = self.__dict__.copy()

        super().__init__()

        self.frequencies, self.duration = self.shape

        self.self.source_positional_embeddings_time = nn.Parameter(
            torch.randn((1,
                         source_sequence_duration,
                         positional_embedding_size))
        )

        self.self.source_positional_embeddings_frequency = nn.Parameter(
            torch.randn((1,
                         source_sequence_,
                         positional_embedding_size))
        )

        self.target_positional_embeddings = nn.Parameter(
            torch.randn((1,
                         target_sequence_length,
                         positional_embedding_size))
        )

    def forward(self, input: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                cache: Optional[Mapping[str, torch.Tensor]] = None):
        batch_dim, frequency_dim, time_dim = (0, 1, 2)



    @classmethod
    def from_parameters_and_weights(
            cls, parameters_json_path: pathlib.Path,
            model_weights_checkpoint_path: pathlib.Path,
            device: Union[str, torch.device] = 'cpu') -> 'PixelSNAIL':
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
            model = cls(**parameters)

        model_state_dict = torch.load(model_weights_checkpoint_path,
                                      map_location=device)
        if 'model' in model_state_dict.keys():
            model_state_dict = model_state_dict['model']
        model.load_state_dict(model_state_dict)
        return model

    def store_instantiation_parameters(self, path: pathlib.Path) -> None:
        """Store the parameters used to create this instance as JSON"""
        with open(path, 'w') as f:
            json.dump(self._instantiation_parameters, f)
