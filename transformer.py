from typing import Iterable, Mapping, Union, Optional, Sequence, Tuple
import pathlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from HT.transformer_custom import (
    TransformerCustom, TransformerDecoderCustom, TransformerEncoderCustom,
    TransformerDecoderLayerCustom, TransformerEncoderLayerCustom)


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
        use_relative_transformer: bool = False,
        class_conditioning_num_classes_per_modality: Optional[Mapping[str, int]] = None,
        class_conditioning_embedding_dim_per_modality: Optional[Mapping[str, int]] = None,
        class_conditioning_prepend_to_dummy_input: bool = False,
        local_class_conditioning: bool = False,
        add_mask_token_to_symbols: bool = False,
        conditional_model: bool = False,
        self_conditional_model: bool = False,
        condition_shape: Optional[Tuple[int, int]] = None,
        conditional_model_num_encoder_layers: int = 6,
        conditional_model_num_decoder_layers: int = 8,
        conditional_model_nhead: int = 8,
        unconditional_model_num_encoder_layers: int = 6,
        unconditional_model_nhead: int = 8,
        use_identity_memory_mask: bool = False,
    ):
        self.shape = shape

        if self_conditional_model:
            assert use_relative_transformer, (
                "Self conditioning only meanigful for relative transformers")
            assert conditional_model, (
                "Self-conditioning is a specific case of conditioning")
            assert (condition_shape is None or condition_shape == shape)
            assert add_mask_token_to_symbols

        self.conditional_model = conditional_model
        if self.conditional_model:
            assert condition_shape is not None
        self.self_conditional_model = self_conditional_model
        self.add_mask_token_to_symbols = add_mask_token_to_symbols
        self.use_relative_transformer = use_relative_transformer
        if self.use_relative_transformer and not predict_frequencies_first:
            raise (NotImplementedError,
                   "Relative positioning only implemented along time")
        self.condition_shape = condition_shape
        if self.self_conditional_model:
            self.condition_shape = self.shape
        self.local_class_conditioning = local_class_conditioning

        self.n_class = n_class
        self.add_mask_token_to_symbols = add_mask_token_to_symbols

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

        # TODO(theis) unify to self.num_encoder_layers and self.n_head
        # and find a better way to manage the different default values
        if self.conditional_model:
            self.conditional_model_num_encoder_layers = conditional_model_num_encoder_layers
            self.conditional_model_nhead = conditional_model_nhead
            self.conditional_model_num_decoder_layers = conditional_model_num_decoder_layers
            self.use_identity_memory_mask = use_identity_memory_mask
        else:
            self.unconditional_model_num_encoder_layers = unconditional_model_num_encoder_layers
            self.unconditional_model_nhead = unconditional_model_nhead

        self._instantiation_parameters = self.__dict__.copy()

        super().__init__()

        if self.add_mask_token_to_symbols:
            self.n_class_in = self.n_class + 1
            self.mask_token_index = self.n_class_in - 1
            self.n_class_out = self.n_class
        else:
            self.n_class_out = self.n_class_in = self.n_class

        if self.conditional_model:
            self.source_frequencies, self.source_duration = (
                self.condition_shape)
        else:
            self.source_frequencies, self.source_duration = self.shape

        if self.use_relative_transformer:
            self.source_num_channels, self.source_num_events = (
                self.source_frequencies, self.source_duration)

        self.source_transformer_sequence_length = (
            self.source_frequencies * self.source_duration)

        if self.conditional_model:
            self.target_frequencies, self.target_duration = self.shape

            self.target_transformer_sequence_length = (
                self.target_frequencies * self.target_duration)

            if self.use_relative_transformer:
                if not self.self_conditional_model:
                    self.target_events_per_source_patch = (
                        (self.target_duration // self.source_duration)
                        * (self.target_frequencies // self.source_frequencies)
                    )
                    self.target_num_channels = (
                        self.target_events_per_source_patch)

                    self.target_num_events = (
                        self.target_transformer_sequence_length
                        // self.target_num_channels)
                    # downsampling_factor = (
                    #     (self.shape[0]*self.shape[1])
                    #     // (self.condition_shape[0]*self.condition_shape[1]))
                else:
                    self.target_num_channels = self.source_num_channels
                    self.target_num_events = self.source_num_events

        if self.conditional_model:
            self.output_sizes = (-1, self.target_frequencies,
                                 self.target_duration,
                                 self.n_class_out)
        else:
            self.output_sizes = (-1, self.source_frequencies,
                                 self.source_duration,
                                 self.n_class_out)

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
            if not self.use_relative_transformer:
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
            else:
                self.target_positional_embeddings_time = None

                # decoder-level, patch-based relative position embeddings
                # allows to locate elements within a patch of the decoder
                self.target_positional_embeddings_patch = nn.Parameter(
                    torch.randn((1,  # batch-size
                                self.target_frequencies // self.source_frequencies,  # frequency-dimension
                                self.target_duration // self.source_duration,  # time-dimension
                                self.positional_embeddings_dim // 2))
                )

                self.target_positional_embeddings_frequency = nn.Parameter(
                    torch.randn((1,  # batch-size
                                self.target_frequencies,  # frequency-dimension
                                1,  # time-dimension
                                self.positional_embeddings_dim // 2))
                )

        if self.embeddings_dim is None:
            self.embeddings_dim = self.d_model-self.positional_embeddings_dim

        self.source_embeddings_linear = nn.Linear(
            self.embeddings_dim,
            self.d_model-self.positional_embeddings_dim)
        self.source_embed = torch.nn.Embedding(self.n_class_in,
                                               self.embeddings_dim)

        if self.conditional_model:
            self.target_embeddings_linear = nn.Linear(
                self.embeddings_dim,
                self.d_model-self.positional_embeddings_dim)

            self.target_embed = torch.nn.Embedding(self.n_class_out,
                                                   self.embeddings_dim)

        # convert Transformer outputs to class-probabilities (as logits)
        self.project_transformer_outputs_to_logits = (
            nn.Linear(self.d_model, self.n_class_out))

        self.class_conditioning_num_modalities = 0
        self.class_conditioning_total_dim = 0
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
                raise NotImplementedError
                # insert class conditioning at end of the start symbol
                current_position = self.d_model
                direction = -1

            for modality_name, modality_embedding_dim in (
                    self.class_conditioning_embedding_dim_per_modality.items()):
                self.class_conditioning_start_positions_per_modality[modality_name] = (
                    current_position
                )
                current_position = current_position + direction*modality_embedding_dim

            self.class_conditioning_total_dim_with_positions = (
                self.class_conditioning_total_dim
                + self.positional_embeddings_dim)

        self.source_start_symbol_dim = self.d_model
        # TODO reduce dimensionality of start symbol and use a linear layer to expand it
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

        if not self.use_relative_transformer:
            if self.conditional_model:
                self.transformer = nn.Transformer(
                    nhead=self.conditional_model_nhead,
                    num_encoder_layers=self.conditional_model_num_encoder_layers,
                    num_decoder_layers=self.conditional_model_num_decoder_layers,
                    d_model=self.d_model)
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.unconditional_model_nhead)
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=self.unconditional_model_num_encoder_layers)
        else:
            if self.conditional_model:
                encoder_nhead = self.conditional_model_nhead
                encoder_num_layers = self.conditional_model_num_encoder_layers
            else:
                encoder_nhead = self.unconditional_model_nhead
                encoder_num_layers = (
                    self.unconditional_model_num_encoder_layers)

            encoder_layer = TransformerEncoderLayerCustom(
                d_model=self.d_model,
                nhead=encoder_nhead,
                attention_bias_type='relative_attention',
                num_channels=self.source_num_channels,
                num_events=self.source_num_events
            )

            relative_encoder = TransformerEncoderCustom(
                encoder_layer=encoder_layer,
                num_layers=encoder_num_layers
            )
            if self.conditional_model:
                attention_bias_type_cross = 'relative_attention_target_source'
                if self.use_identity_memory_mask:
                    attention_bias_type_cross = 'no_bias'
                decoder_layer = TransformerDecoderLayerCustom(
                    d_model=self.d_model,
                    nhead=self.conditional_model_nhead,
                    attention_bias_type_self='relative_attention',
                    attention_bias_type_cross=attention_bias_type_cross,
                    num_channels_encoder=self.source_num_channels,
                    num_events_encoder=self.source_num_events,
                    num_channels_decoder=self.target_num_channels,
                    num_events_decoder=self.target_num_events,
                    conditional=self.local_class_conditioning,
                    condition_embeddings_dim=(
                        self.class_conditioning_total_dim_with_positions)
                )

                custom_decoder = TransformerDecoderCustom(
                    decoder_layer=decoder_layer,
                    num_layers=self.conditional_model_num_decoder_layers
                )

                self.transformer = TransformerCustom(
                    nhead=self.conditional_model_nhead,
                    custom_encoder=relative_encoder,
                    custom_decoder=custom_decoder,
                    d_model=self.d_model)
            else:
                self.transformer = relative_encoder

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

        if not self.use_relative_transformer:
            repeated_time_embeddings = (
                positional_embeddings_time
                .repeat(1, frequencies, 1, 1))
            return torch.cat([repeated_frequency_embeddings,
                              repeated_time_embeddings],
                             dim=embedding_dim)
        else:
            if kind == 'target':
                positional_embeddings_patch = (
                    self.target_positional_embeddings_patch)

                repeated_patch_embeddings = (
                    positional_embeddings_patch
                    .repeat(
                        1, self.source_frequencies, self.source_duration, 1)
                )
                return torch.cat([repeated_frequency_embeddings,
                                  repeated_patch_embeddings],
                                 dim=embedding_dim)
            else:
                return torch.cat([repeated_frequency_embeddings,
                                  repeated_frequency_embeddings],
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

    @property
    def identity_memory_mask(self) -> torch.Tensor:
        identity_memory_mask = (
            torch.eye(self.source_transformer_sequence_length).float())
        identity_memory_mask = (
            identity_memory_mask
            .masked_fill(identity_memory_mask == 0, float('-inf'))
            .masked_fill(identity_memory_mask == 1, float(0.0))
        )
        return identity_memory_mask

    def zig_zag_reshaping_frequencies_first(self, input: torch.Tensor
                                            ) -> torch.Tensor:
        """Re-order a target tensor via a source-defined block ordering
        """
        # patches of the target should be identified according to
        # the position they occupy below their respective "source" patch

        # slice along the time-axis
        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 2, 1, 3)

        was_reshaped = False
        if input.ndim == 3:
            input = input.unsqueeze(-1)
            was_reshaped = True

        batch_size, _, _, embedding_size = input.shape

        time_slices_size = self.target_duration // self.source_num_events
        input = input.unfold(
            time_dim,
            time_slices_size,
            time_slices_size)

        # slice along the frequency-axis
        frequency_slices_size = self.target_frequencies // self.source_num_channels
        input = input.unfold(
            frequency_dim,
            frequency_slices_size,
            frequency_slices_size)

        # push back the embedding dimension as innermost dimension
        input = input.permute(
            batch_dim, time_dim, frequency_dim, 4, 5, embedding_dim)

        input = input.reshape(
            batch_size, self.target_duration, self.target_frequencies,
            embedding_size)

        if was_reshaped:
            input = input.squeeze(-1)
        return input

    def inverse_zig_zag_reshaping_frequencies_first(self,
                                                    input: torch.Tensor
                                                    ) -> torch.Tensor:
        """Re-order a target tensor via a source-defined block ordering
        """
        # patches of the target should be identified according to
        # the position they occupy below their respective "source" patch

        # slice along the time-axis
        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 2, 1, 3)
        batch_size, _, embedding_size = input.shape

        input = input.reshape(batch_size, self.target_num_channels,
                              self.target_num_events, embedding_size)

        input = input.transpose(1, 2)

        input = input.reshape(batch_size, self.target_frequencies,
                              self.target_duration, embedding_size)

        # slice along the frequency-axis
        frequency_slices_size = self.target_frequencies // self.source_num_channels
        input = input.unfold(
            frequency_dim,
            frequency_slices_size,
            frequency_slices_size)

        time_slices_size = self.target_duration // self.source_num_events
        input = input.unfold(
            time_dim,
            time_slices_size,
            time_slices_size)

        # push back the embedding dimension as innermost dimension
        input = input.permute(
            batch_dim, time_dim, frequency_dim, 4, 5, embedding_dim)

        input = input.reshape(
            batch_size, self.target_frequencies, self.target_duration,
            embedding_size)
        input = input.transpose(1, 2)
        return input

    def prepare_data(self, input: torch.Tensor, kind: Optional[str] = None,
                     class_conditioning: Mapping[str, torch.Tensor] = {},
                     mask: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
        if not self.conditional_model:
            assert kind == 'source' or kind is None
            kind = 'source'
        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 1, 2, 3)
        batch_size, frequencies, duration = input.shape

        input_as_sequence = self.flatten_map(input, kind=kind)

        if self.self_conditional_model and mask is not None:
            if mask.ndim > 2:
                mask = self.flatten_map(mask, kind=kind)
            input_as_sequence = input_as_sequence.masked_fill(
                mask, self.mask_token_index)

        (batch_dim, sequence_dim, embedding_dim) = (0, 1, 2)

        embedded_sequence = self.embed_data(input_as_sequence, kind=kind)

        embedded_sequence_with_positions = self.add_positions_to_sequence(
            embedded_sequence, kind=kind, embedding_dim=embedding_dim
        )

        prepared_sequence = self.maybe_add_start_symbol(
            embedded_sequence_with_positions, kind=kind,
            class_conditioning=class_conditioning, sequence_dim=1
        )

        return prepared_sequence, (
            (batch_dim, frequency_dim, time_dim))

    def add_positions_to_sequence(self, sequence: torch.Tensor, kind: str,
                                  embedding_dim: int):
        # add positional embeddings
        batch_size = sequence.shape[0]
        # combine time and frequency embeddings
        if kind == 'source':
            frequencies = self.source_frequencies
            duration = self.source_duration
            positional_embeddings = self.combined_positional_embeddings_source
            transformer_sequence_length = (
                self.source_transformer_sequence_length)
        elif kind == 'target':
            frequencies = self.target_frequencies
            duration = self.target_duration
            positional_embeddings = self.combined_positional_embeddings_target
            transformer_sequence_length = (
                self.target_transformer_sequence_length)
        else:
            raise ValueError(f"Unexpected value {kind} for kind option")

        # repeat positional embeddings over whole batch
        positional_embeddings = (
            positional_embeddings
            .reshape(1, frequencies, duration, -1)
            .repeat(batch_size, 1, 1, 1))
        positions_as_sequence = self.flatten_map(positional_embeddings,
                                                 kind=kind)

        sequence_with_positions = torch.cat(
            [sequence, positions_as_sequence],
            dim=embedding_dim
        )

        return sequence_with_positions

    def maybe_add_start_symbol(self, sequence_with_positions: torch.Tensor,
                               kind: str,
                               class_conditioning: Mapping[str, torch.Tensor],
                               sequence_dim: int):
        if self.conditional_model and kind == 'source':
            return sequence_with_positions
        # removing the unnecessary else to remove one level of indentation

        batch_size = sequence_with_positions.shape[0]
        # combine time and frequency embeddings
        if kind == 'source':
            start_symbol = self.source_start_symbol
            transformer_sequence_length = (
                self.source_transformer_sequence_length)
        elif kind == 'target':
            start_symbol = self.target_start_symbol
            transformer_sequence_length = (
                self.target_transformer_sequence_length)
        else:
            raise ValueError(f"Unexpected value {kind} for kind option")

        # repeat start-symbol over whole batch
        start_symbol = start_symbol.repeat(batch_size, 1, 1)

        if not self.local_class_conditioning:
            # add conditioning tensors to start-symbol
            for condition_name, class_condition in class_conditioning.items():
                embeddings = (
                    self.class_conditioning_embedding_layers[
                        condition_name](class_condition)).squeeze(1)
                start_position = (
                    self.class_conditioning_start_positions_per_modality[
                        condition_name])
                start_symbol[:, 0, start_position:start_position+embeddings.shape[1]] = embeddings

        sequence_with_start_symbol = torch.cat(
                [start_symbol,
                 sequence_with_positions.narrow(
                     sequence_dim,
                     0,
                     transformer_sequence_length-1)],
                dim=sequence_dim
        )
        return sequence_with_start_symbol

    def flatten_map(self, codemap: torch.Tensor, kind: str) -> torch.Tensor:
        batch_dim, frequency_dim, time_dim = (0, 1, 2)
        batch_size, frequencies, duration = codemap.shape[:3]

        if self.predict_frequencies_first:
            codemap = codemap.transpose(
                time_dim, frequency_dim)
            (frequency_dim, time_dim) = (2, 1)

        if not self.predict_low_frequencies_first:
            raise NotImplementedError

        if (kind == 'target'
                and self.use_relative_transformer
                and not self.self_conditional_model):
            # reshape the data for relative positioning:
            zig_zag_codemap = (
                self.zig_zag_reshaping_frequencies_first(codemap)
            )
            codemap = zig_zag_codemap

        flattened_codemap = (
            codemap.reshape(batch_size,
                            frequencies * duration,
                            -1)
            )
        if codemap.ndim == 3:
            flattened_codemap = flattened_codemap.squeeze(-1)
        dimensions = (batch_dim, sequence_dim, embedding_dim) = (0, 1, 2)
        return flattened_codemap

    def to_sequences(
            self, input: torch.Tensor,
            condition: Optional[torch.Tensor] = None,
            class_conditioning: Mapping[str, torch.Tensor] = {},
            mask: Optional[torch.BoolTensor] = None
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_dim, frequency_dim, time_dim, embedding_dim = (0, 1, 2, 3)
        batch_size = input.shape[0]

        # TODO(theis): maybe not very readable...
        if self.conditional_model:
            source = condition
            target = input
        else:
            source = input

        source_sequence, (
            batch_dim, frequency_dim, time_dim) = self.prepare_data(
                input=source, kind='source',
                class_conditioning=class_conditioning,
                mask=mask)

        if self.conditional_model:
            target_sequence, _ = self.prepare_data(
                input=target, kind='target',
                class_conditioning=class_conditioning)
        else:
            target_sequence = None

        (batch_dim, sequence_dim, embedding_dim) = (0, 1, 2)

        return source_sequence, target_sequence

    def make_condition_sequence(self,
                                class_conditioning: Mapping[str, torch.Tensor],
                                ) -> torch.Tensor:
        kind = 'source'

        # if len(class_conditioning) == 0:
        #     batch_size = 0
        if len(class_conditioning) == 0:
            raise NotImplementedError

        batch_size = list(class_conditioning.values())[0].shape[0]
        embeddings = torch.zeros(batch_size, self.source_frequencies,
                                 self.source_duration,
                                 self.class_conditioning_total_dim
                                 )

        for condition_name, class_condition in class_conditioning.items():
            condition_embeddings = (
                self.class_conditioning_embedding_layers[
                    condition_name](class_condition))
            start_position = (
                    self.class_conditioning_start_positions_per_modality[
                        condition_name])
            end_position = start_position + condition_embeddings.shape[-1]
            embeddings[..., start_position:end_position] = condition_embeddings

        embeddings_sequence = self.flatten_map(embeddings,
                                               kind=kind)

        class_embeddings_sequence_with_positions = (
            self.add_positions_to_sequence(embeddings_sequence,
                                           kind=kind,
                                           embedding_dim=-1))
        return class_embeddings_sequence_with_positions

    def to_time_frequency_map(self, sequence: torch.Tensor, kind: str,
                              permute_output_as_logits: bool = False) -> torch.Tensor:
        if self.predict_frequencies_first:
            (frequency_dim, time_dim) = (2, 1)
        else:
            (frequency_dim, time_dim) = (1, 2)
        batch_size = sequence.shape[0]

        if not self.predict_low_frequencies_first:
            raise NotImplementedError

        sequence_dimensions = sequence.ndim
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

        output_shape = [None] * len(self.output_sizes)
        _, output_frequencies, output_duration, _ = self.output_sizes

        if self.predict_frequencies_first:
            output_shape = (batch_size, output_duration, output_frequencies,
                            embedding_size)
        else:
            output_shape = (batch_size, output_frequencies, output_duration,
                            embedding_size)

        # reshape output to time-frequency format
        time_frequency_map = sequence.reshape(*output_shape)

        if (kind == 'target'
                and self.conditional_model
                and self.use_relative_transformer
                and not self.self_conditional_model):
            # reshape the data for relative positioning:
            zig_zag_map = (
                self.inverse_zig_zag_reshaping_frequencies_first(
                    sequence)
            )
            time_frequency_map = zig_zag_map

        if self.predict_frequencies_first:
            time_frequency_map = time_frequency_map.transpose(
                time_dim, frequency_dim)
            (frequency_dim, time_dim) = (1, 2)

        if is_logits and permute_output_as_logits:
            # permute dimensions to follow PyTorch logits convention
            time_frequency_map = (time_frequency_map.permute(0, 3, 1, 2))
        if not is_logits:
            time_frequency_map = time_frequency_map.squeeze(-1)
        return time_frequency_map

    def forward(self, input: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                class_condition: Optional[torch.Tensor] = None,
                cache: Optional[Mapping[str, torch.Tensor]] = None):
        (batch_dim, sequence_dim) = (0, 1)

        if self.conditional_model:
            target_sequence = input
            source_sequence = condition
        else:
            source_sequence = input
            target_sequence = None

        # transformer inputs are in time-major format
        time_major_source_sequence = source_sequence.transpose(0, 1)
        if self.conditional_model:
            time_major_target_sequence = target_sequence.transpose(0, 1)

        if self.local_class_conditioning:
            time_major_class_condition_sequence = class_condition.transpose(0, 1)
        else:
            time_major_class_condition_sequence = None
        (batch_dim, sequence_dim) = (1, 0)

        memory_mask = None
        if self.use_identity_memory_mask:
            memory_mask = self.identity_memory_mask

        if self.conditional_model:
            output_sequence = self.transformer(
                time_major_source_sequence,
                time_major_target_sequence,
                src_mask=(None if not self.self_conditional_model
                          else self.causal_mask.t()  # anti-causal mask
                          ),
                tgt_mask=self.causal_mask,
                memory_mask=memory_mask,
                condition=time_major_class_condition_sequence)
        else:
            output_sequence = self.transformer(time_major_source_sequence,
                                               mask=self.causal_mask)
        if self.use_relative_transformer:
            output_sequence, *attentions = output_sequence

        # transpose back to batch-major shape
        output_sequence = output_sequence.transpose(
            batch_dim, sequence_dim)
        (batch_dim, sequence_dim) = (0, 1)

        # convert outputs to class probabilities
        logits = self.project_transformer_outputs_to_logits(output_sequence)

        return logits, None

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
