from typing import Iterable, Mapping, Union, Optional, Sequence, Tuple
import pathlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VQNSynthTransformer(nn.Module):
    """Transformer-based generative model for latent maps

    Inputs are expected to be of shape [num_frequency_bands, time_duration],
    with `sample[0, 0]` in the image the energy in the lowest frequency band
    in the first time-frame.

    Arguments:
        * predict_frequencies_first (bool, optional, default is False):
            if True, transposes the inputs to predict time-frame by time-frame,
            potentially bringing more coherence in frequency within a given
            time-frame by bringing the dependencies closer.
        * predict_low_frequencies_first (bool, optional, default is True):
            if False, flips the inputs so that frequencies are predicted in
            decreasing order, starting at the harmonics.
        * class_conditioning_num_classes_per_modality (iterable of int,
            optional, default is None):
            when using class-labels for conditioning, provide in this iterable
            the number of classes per modality (e.g. pitch, instrument class...)
            to intialize the embeddings layer
        * class_conditioning_embedding_dim_per_modality (iterable of int,
        optional, default is None):
            when using class-labels for conditioning, provide in this iterable the
            dimension of embeddings to use for each modality
    """
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
        class_conditioning_num_classes_per_modality: Optional[Mapping[str, int]] = None,
        class_conditioning_embedding_dim_per_modality: Optional[Mapping[str, int]] = None,
        class_conditioning_prepend_to_dummy_input: bool = False,
        conditional_model: bool = False,
        condition_shape: Optional[Tuple[int, int]] = None,
        conditional_model_num_encoder_layers: int = 12,
        conditional_model_nhead: int = 16,
        unconditional_model_num_encoder_layers: int = 6,
        unconditional_model_nhead: int = 8,
    ):
        self.shape = shape
        self.conditional_model = conditional_model
        if self.conditional_model:
            assert condition_shape is not None
        self.condition_shape = condition_shape

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

        self.class_conditioning_num_classes_per_modality = class_conditioning_num_classes_per_modality
        self.class_conditioning_embedding_dim_per_modality = class_conditioning_embedding_dim_per_modality
        self.class_conditioning_prepend_to_dummy_input = class_conditioning_prepend_to_dummy_input

        if self.conditional_model:
            self.conditional_model_num_encoder_layers = conditional_model_num_encoder_layers
            self.conditional_model_nhead = conditional_model_nhead
        else:
            self.unconditional_model_num_encoder_layers = unconditional_model_num_encoder_layers
            self.unconditional_model_nhead = unconditional_model_nhead

        self._instantiation_parameters = self.__dict__.copy()

        super().__init__()

        if self.conditional_model:
            self.source_frequencies, self.source_duration = self.condition_shape
        else:
            self.source_frequencies, self.source_duration = self.shape
        self.source_transformer_sequence_length = (
            self.source_frequencies * self.source_duration)

        if self.conditional_model:
            self.target_frequencies, self.target_duration = self.shape
            self.target_transformer_sequence_length = (
                self.target_frequencies * self.target_duration)

        if self.conditional_model:
            self.output_sizes = (-1, self.target_frequencies,
                                 self.target_duration,
                                 self.n_class)
        else:
            self.output_sizes = (-1, self.source_frequencies,
                                 self.source_duration,
                                 self.n_class)

        self.source_positional_embeddings_frequency = nn.Parameter(
            torch.randn((1,  # batch-size
                         self.source_frequencies,  # frequency-dimension
                         1,  # time-dimension
                         self.positional_embeddings_dim//2))
        )

        self.source_positional_embeddings_time = nn.Parameter(
            torch.randn((1,  # batch-size
                         1,  # frequency-dimension
                         self.source_duration,  # time-dimension
                         self.positional_embeddings_dim//2))
        )

        if self.conditional_model:
            self.target_positional_embeddings_frequency = nn.Parameter(
                torch.randn((1,  # batch-size
                            self.target_frequencies,  # frequency-dimension
                            1,  # time-dimension
                            self.positional_embeddings_dim//2))
            )

            self.target_positional_embeddings_time = nn.Parameter(
                torch.randn((1,  # batch-size
                            1,  # frequency-dimension
                            self.target_duration,  # time-dimension
                            self.positional_embeddings_dim//2))
            )

        if self.embeddings_dim is None:
            self.embeddings_dim = self.d_model-self.positional_embeddings_dim

        self.source_embeddings_linear = nn.Linear(
            self.embeddings_dim,
            self.d_model-self.positional_embeddings_dim)
        self.source_embed = torch.nn.Embedding(self.n_class,
                                               self.embeddings_dim)

        if self.conditional_model:
            self.target_embeddings_linear = nn.Linear(
                self.embeddings_dim,
                self.d_model-self.positional_embeddings_dim)

            self.target_embed = torch.nn.Embedding(self.n_class,
                                                   self.embeddings_dim)

        # convert Transformer outputs to class-probabilities (as logits)
        self.project_transformer_outputs_to_logits = (
            nn.Linear(self.d_model, self.n_class))

        self.class_conditioning_total_embedding_dim = 0
        self.class_conditioning_num_modalities = 0
        self.class_conditioning_embedding_layers = nn.ModuleDict()
        self.class_conditioning_class_to_index_per_modality = {}
        self.class_conditioning_start_positions_per_modality = {}

        if self.class_conditioning_num_classes_per_modality is not None:
            self.class_conditioning_num_modalities = len(
                self.class_conditioning_embedding_dim_per_modality.values())
            self.class_conditioning_total_dim = sum(
                self.class_conditioning_embedding_dim_per_modality.values())

            # initialize class conditioning embedding layers
            for (modality_name, modality_num_classes), modality_embedding_dim in zip(
                    self.class_conditioning_num_classes_per_modality.items(),
                    self.class_conditioning_embedding_dim_per_modality.values()):
                self.class_conditioning_embedding_layers[modality_name] = (
                    torch.nn.Embedding(modality_num_classes,
                                       modality_embedding_dim)
                )

            # initialize start positions for class conditioning in start symbol
            if self.class_conditioning_prepend_to_dummy_input:
                # insert class conditioning at beginning of the start symbol
                current_position = 0
                direction = +1
            else:
                # insert class conditioning at end of the start symbol
                current_position = self.d_model
                direction = -1

            for modality_name, modality_embedding_dim in (
                    self.class_conditioning_embedding_dim_per_modality.items()):
                current_position = current_position + direction*modality_embedding_dim
                self.class_conditioning_start_positions_per_modality[modality_name] = (
                    current_position
                )

        self.source_start_symbol_dim = self.d_model
        # TODO reduce dimensionality of start symbol
        self.source_start_symbol = nn.Parameter(
            torch.randn((1, 1, self.source_start_symbol_dim))
        )

        if self.conditional_model:
            self.target_start_symbol_dim = (
                self.d_model
                # - self.class_conditioning_total_embedding_dim
            )
            self.target_start_symbol = nn.Parameter(
                torch.randn((1, 1, self.target_start_symbol_dim))
            )

        if self.conditional_model:
            self.transformer = nn.Transformer(
                nhead=self.conditional_model_nhead,
                num_encoder_layers=self.conditional_model_num_encoder_layers,
                d_model=self.d_model)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.unconditional_model_nhead)
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.unconditional_model_num_encoder_layers)

    def embed_data(self, input: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == 'source':
            return self.source_embeddings_linear(self.source_embed(input))
        elif kind == 'target' and self.conditional_model:
            return self.target_embeddings_linear(self.target_embed(input))
        else:
            raise ValueError(f"Unexpected value {kind} for kind option")

    def _get_combined_positional_embeddings(self, kind: str) -> torch.Tensor:
        if kind == 'source':
            positional_embeddings_frequency = (
                self.source_positional_embeddings_frequency)
            positional_embeddings_time = self.source_positional_embeddings_time
            frequencies = self.source_frequencies
            duration = self.source_duration
        elif kind == 'target' and self.conditional_model:
            positional_embeddings_frequency = (
                self.target_positional_embeddings_frequency)
            positional_embeddings_time = self.target_positional_embeddings_time
            frequencies = self.target_frequencies
            duration = self.target_duration
        else:
            raise ValueError(f"Unexpected value {kind} for kind option")

        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 1, 2, 3)

        repeated_frequency_embeddings = (
            positional_embeddings_frequency
            .repeat(1, 1, duration, 1))
        repeated_time_embeddings = (
            positional_embeddings_time
            .repeat(1, frequencies, 1, 1))

        return torch.cat([repeated_frequency_embeddings,
                          repeated_time_embeddings],
                         dim=embedding_dim)

    @property
    def combined_positional_embeddings_source(self) -> torch.Tensor:
        return self._get_combined_positional_embeddings('source')

    @property
    def combined_positional_embeddings_target(self) -> torch.Tensor:
        return self._get_combined_positional_embeddings('target')

    @property
    def causal_mask(self) -> torch.Tensor:
        """Generate a mask to impose causality"""
        if self.conditional_model:
            # masking is applied only on the target, access is allowed to
            # all positions on the conditioning input
            causal_mask_length = self.target_transformer_sequence_length
        else:
            # apply causal mask to the input for the prediction task
            causal_mask_length = self.source_transformer_sequence_length

        mask = (torch.triu(torch.ones(causal_mask_length,
                                      causal_mask_length)) == 1
                ).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def prepare_data(self, input: torch.Tensor, kind: Optional[str] = None,
                     class_conditioning: Mapping[str, torch.Tensor] = {}
                     ) -> torch.Tensor:
        if not self.conditional_model:
            assert kind == 'source' or kind is None
            kind = 'source'
        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 1, 2, 3)
        batch_size, frequencies, duration = input.shape

        embedded_input = self.embed_data(input, kind)

        # add positional embeddings
        # combine time and frequency embeddings
        if kind == 'source':
            positional_embeddings = self.combined_positional_embeddings_source
            start_symbol = self.source_start_symbol
            transformer_sequence_length = (
                self.source_transformer_sequence_length)
        elif kind == 'target':
            positional_embeddings = self.combined_positional_embeddings_target
            start_symbol = self.target_start_symbol
            transformer_sequence_length = (
                self.target_transformer_sequence_length)
        else:
            raise ValueError(f"Unexpected value {kind} for kind option")

        # repeat start-symbol over whole batch
        start_symbol = start_symbol.repeat(batch_size, 1, 1)

        for condition_name, class_condition in class_conditioning.items():
            embeddings = (
                self.class_conditioning_embedding_layers[
                    condition_name](class_condition)).squeeze(1)
            start_position = (
                self.class_conditioning_start_positions_per_modality[
                    condition_name])
            start_symbol[:, 0, start_position:start_position+embeddings.shape[1]] = embeddings

        input_with_positions = torch.cat(
            [embedded_input, positional_embeddings.repeat(batch_size, 1, 1, 1)],
            dim=embedding_dim
        )

        if self.predict_frequencies_first:
            input_with_positions = input_with_positions.transpose(
                time_dim, frequency_dim)
            (frequency_dim, time_dim) = (2, 1)
        if not self.predict_low_frequencies_first:
            raise NotImplementedError

        flattened_input_with_positions = (
            input_with_positions.reshape(batch_size,
                                         frequencies * duration,
                                         self.d_model)
        )
        (batch_dim, sequence_dim, embedding_dim) = (0, 1, 2)

        # Shift inputs
        # we do this so that the output of the transformer can be readily
        # interpreted as the probability of generating each possible output
        # at that position
        if not (self.conditional_model and kind == 'source'):
            shifted_sequence_with_positions = torch.cat(
                [start_symbol,
                 flattened_input_with_positions.narrow(
                     sequence_dim,
                     0,
                     transformer_sequence_length-1)],
                dim=sequence_dim
            )

            prepared_sequence = shifted_sequence_with_positions
        else:
            prepared_sequence = flattened_input_with_positions

        return prepared_sequence, (
            (batch_dim, frequency_dim, time_dim))

    def forward(self, input: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                class_conditioning: Mapping[str, torch.Tensor] = {},
                cache: Optional[Mapping[str, torch.Tensor]] = None):
        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 1, 2, 3)
        batch_size = input.shape[0]

        causal_mask = self.causal_mask

        # TODO(theis): maybe not very readable...
        if self.conditional_model:
            source = condition
            target = input
        else:
            source = input

        batch_major_source_sequence, (
            batch_dim, frequency_dim, time_dim) = self.prepare_data(
            input=source, kind='source',
            class_conditioning=class_conditioning)

        if self.conditional_model:
            batch_major_target_sequence, _ = self.prepare_data(
                input=target, kind='target',
                class_conditioning=class_conditioning)
        (batch_dim, sequence_dim, embedding_dim) = (0, 1, 2)

        # nn.Transformer inputs are in time-major shape
        time_major_source_sequence = batch_major_source_sequence.transpose(
            batch_dim, sequence_dim)
        if self.conditional_model:
            time_major_target_sequence = batch_major_target_sequence.transpose(
                batch_dim, sequence_dim)
        (batch_dim, sequence_dim) = (1, 0)
        if self.conditional_model:
            output_sequence = self.transformer(
                time_major_source_sequence,
                time_major_target_sequence,
                # no mask on the source to allow performing attention
                # over the whole conditioning sequence
                src_mask=None,
                tgt_mask=causal_mask)
        else:
            output_sequence = self.transformer(time_major_source_sequence,
                                               mask=causal_mask)
        # transpose back to batch-major shape
        output_sequence = output_sequence.transpose(
            batch_dim, sequence_dim)
        (batch_dim, sequence_dim) = (0, 1)

        # convert outputs to class probabilities
        logits = self.project_transformer_outputs_to_logits(output_sequence)

        output_dimensions = batch_dim, frequency_dim, time_dim, logit_dim = (
            0, frequency_dim, time_dim, 3)

        output_shape = [None] * len(self.output_sizes)
        for dim_index, size in zip(output_dimensions, self.output_sizes):
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
            device: Union[str, torch.device] = 'cpu'
            ) -> 'VQNSynthTransformer':
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
            json.dump(self._instantiation_parameters, f, indent=4)
