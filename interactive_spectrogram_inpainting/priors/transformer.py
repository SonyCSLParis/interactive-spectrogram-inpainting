from typing import Iterable, Mapping, Union, Optional, Tuple
import pathlib
import json
from abc import abstractmethod
from enum import Enum, auto

import torch
from torch import nn

from .codemaps_helpers import (CodemapsHelper,
                               SimpleCodemapsHelper, ZigZagCodemapsHelper)
from VQCPCB.transformer.transformer_custom import (
    TransformerCustom, TransformerDecoderCustom, TransformerEncoderCustom,
    TransformerDecoderLayerCustom, TransformerEncoderLayerCustom,
    TransformerAlignedDecoderLayerCustom)


class Seq2SeqInputKind(Enum):
    """Types of input sequences for seq2seq models"""
    Source = auto()
    Target = auto()


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

    # helper attributes for appropriately flattening codemaps to sequences
    source_codemaps_helper: SimpleCodemapsHelper
    target_codemaps_helper: CodemapsHelper

    @property
    @abstractmethod
    def use_inpainting_mask_on_source(self) -> bool:
        """Whether to introduce a specific masking token for the source sequences

        This masking token is used to simulate partial information for
        inpaiting operations.
        """
        ...

    def __init__(
        self,
        shape: Iterable[int],  # [num_frequencies, frame_duration]
        n_class: int,
        channel: int,
        kernel_size: int,
        n_block: int,
        n_res_block: int,
        res_channel: int,
        attention: bool = True,  # TODO: remove this parameter
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
        positional_class_conditioning: bool = False,
        add_mask_token_to_symbols: bool = False,
        conditional_model: bool = False,
        self_conditional_model: bool = False,
        use_aligned_decoder: bool = False,
        condition_shape: Optional[Tuple[int, int]] = None,
        conditional_model_num_encoder_layers: int = 6,
        conditional_model_num_decoder_layers: int = 8,
        conditional_model_nhead: int = 8,
        unconditional_model_num_encoder_layers: int = 6,
        unconditional_model_nhead: int = 8,
        use_identity_memory_mask: bool = False,
        use_lstm_DEBUG: bool = False,
        disable_start_symbol_DEBUG: bool = False,
    ):
        if local_class_conditioning:
            raise NotImplementedError(
                "Depecrated in favor of positional class conditioning")

        self.shape = shape

        if self_conditional_model:
            assert use_relative_transformer, (
                "Self conditioning only meanigful for relative transformers")
            assert conditional_model, (
                "Self-conditioning is a specific case of conditioning")
            assert (condition_shape is None or condition_shape == shape)

        assert not (local_class_conditioning and positional_class_conditioning)

        self.conditional_model = conditional_model
        if self.conditional_model:
            assert condition_shape is not None
        self.self_conditional_model = self_conditional_model
        self.use_relative_transformer = use_relative_transformer
        if self.use_relative_transformer and not predict_frequencies_first:
            raise (NotImplementedError,
                   "Relative positioning only implemented along time")
        self.condition_shape = condition_shape
        if self.self_conditional_model:
            self.condition_shape = self.shape.copy()
        self.local_class_conditioning = local_class_conditioning
        self.positional_class_conditioning = positional_class_conditioning

        self.n_class = n_class

        self.channel = channel

        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
        else:
            self.kernel_size = kernel_size

        self.n_block = n_block
        self.n_res_block = n_res_block
        self.res_channel = res_channel
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
        self.conditional_model_num_encoder_layers = conditional_model_num_encoder_layers
        self.conditional_model_nhead = conditional_model_nhead
        self.conditional_model_num_decoder_layers = conditional_model_num_decoder_layers
        self.use_identity_memory_mask = use_identity_memory_mask

        self.use_aligned_decoder = use_aligned_decoder

        self.use_lstm_DEBUG = use_lstm_DEBUG
        self.disable_start_symbol_DEBUG = disable_start_symbol_DEBUG

        self._instantiation_parameters = self.__dict__.copy()

        super().__init__()

        if self.use_inpainting_mask_on_source:
            # add one token to the available source sequence symbols
            self.n_class_source = self.n_class + 1
            self.mask_token_index = self.n_class_source - 1
            # generated, target sequences cannot contain the masking token
            self.n_class_target = self.n_class
        else:
            self.n_class_target = self.n_class_source = self.n_class

        if self.class_conditioning_num_classes_per_modality is not None:
            self.class_conditioning_num_modalities = len(
                self.class_conditioning_embedding_dim_per_modality.values())
            self.class_conditioning_total_dim = sum(
                self.class_conditioning_embedding_dim_per_modality.values())
        else:
            self.class_conditioning_num_modalities = 0
            self.class_conditioning_total_dim = 0

        self.source_frequencies, self.source_duration = (
            self.condition_shape)

        self.source_num_channels, self.source_num_events = (
            1, self.source_frequencies * self.source_duration)

        self.source_transformer_sequence_length = (
            self.source_frequencies * self.source_duration)

        self.target_frequencies, self.target_duration = self.shape

        self.target_transformer_sequence_length = (
            self.target_frequencies * self.target_duration)

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
        # else:
        #     self.target_num_channels = self.source_num_channels
        #     self.target_num_events = self.source_num_events

        self.output_sizes = (-1, self.target_frequencies,
                             self.target_duration,
                             self.n_class_target)

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

        self.source_embed = torch.nn.Embedding(self.n_class_source,
                                               self.embeddings_dim)

        self.embeddings_effective_dim = (self.d_model
                                         - self.positional_embeddings_dim)
        if self.positional_class_conditioning:
            self.embeddings_effective_dim -= self.class_conditioning_total_dim

        self.source_embeddings_linear = nn.Linear(
            self.embeddings_dim,
            self.embeddings_effective_dim
            )

        self.target_embeddings_linear = nn.Linear(
            self.embeddings_dim,
            self.embeddings_effective_dim)

        self.target_embed = torch.nn.Embedding(self.n_class_target,
                                               self.embeddings_dim)

        # convert Transformer outputs to class-probabilities (as logits)
        self.project_transformer_outputs_to_logits = (
            nn.Linear(self.d_model, self.n_class_target))

        self.class_conditioning_embedding_layers = nn.ModuleDict()
        self.class_conditioning_class_to_index_per_modality = {}
        self.class_conditioning_start_positions_per_modality = {}

        if self.class_conditioning_num_classes_per_modality is not None:
            # initialize class conditioning embedding layers
            for (modality_name, modality_num_classes), modality_embedding_dim in zip(
                    self.class_conditioning_num_classes_per_modality.items(),
                    self.class_conditioning_embedding_dim_per_modality.values()):
                self.class_conditioning_embedding_layers[modality_name] = (
                    torch.nn.Embedding(modality_num_classes,
                                       modality_embedding_dim)
                )

            # initialize start positions for class conditioning in start symbol
            if self.positional_class_conditioning or self.class_conditioning_prepend_to_dummy_input:
                # insert class conditioning at beginning of the start symbol
                current_position = 0

                for modality_name, modality_embedding_dim in (
                        self.class_conditioning_embedding_dim_per_modality.items()):
                    self.class_conditioning_start_positions_per_modality[modality_name] = (
                        current_position
                    )
                    current_position = current_position + modality_embedding_dim
            else:
                raise NotImplementedError
                # insert class conditioning at end of the start symbol
                current_position = self.d_model

                for modality_name, modality_embedding_dim in (
                        self.class_conditioning_embedding_dim_per_modality.items()):
                    current_position = current_position - modality_embedding_dim
                    self.class_conditioning_start_positions_per_modality[modality_name] = (
                        current_position
                    )

        self.class_conditioning_total_dim_with_positions = (
            self.class_conditioning_total_dim
            + self.positional_embeddings_dim)

        self.source_start_symbol_dim = self.d_model
        if self.positional_class_conditioning:
            self.source_start_symbol_dim -= self.class_conditioning_total_dim
        # TODO reduce dimensionality of start symbol and use a linear layer to expand it
        self.source_start_symbol = nn.Parameter(
            torch.randn((1, 1, self.source_start_symbol_dim))
        )

        self.source_num_events_with_start_symbol = self.source_num_events + 1

        self.source_transformer_sequence_length_with_start_symbol = (
            self.source_transformer_sequence_length + 1
        )

        self.target_start_symbol_dim = self.d_model
        if self.positional_class_conditioning:
            self.target_start_symbol_dim -= self.class_conditioning_total_dim
        target_start_symbol_duration = self.target_events_per_source_patch
        self.target_start_symbol = nn.Parameter(
            torch.randn((1, target_start_symbol_duration,
                         self.target_start_symbol_dim))
        )

        self.target_num_events_with_start_symbol = (
            self.target_num_events + 1
        )
        self.target_transformer_sequence_length_with_start_symbol = (
            self.target_num_events_with_start_symbol
            * self.target_num_channels
        )

        self.transformer: Union[nn.Transformer, nn.TransformerEncoder,
                                TransformerCustom, TransformerEncoderCustom]
        if self.use_lstm_DEBUG:
            raise NotImplementedError(
                "TODO(theis), debug mode with simple LSTM layers")
        else:
            encoder_nhead = self.conditional_model_nhead
            encoder_num_layers = self.conditional_model_num_encoder_layers

            encoder_layer = TransformerEncoderLayerCustom(
                d_model=self.d_model,
                nhead=encoder_nhead,
                attention_bias_type='relative_attention',
                num_channels=self.source_num_channels,
                num_events=self.source_num_events_with_start_symbol
            )

            relative_encoder = TransformerEncoderCustom(
                encoder_layer=encoder_layer,
                num_layers=encoder_num_layers
            )

            attention_bias_type_cross = 'relative_attention_target_source'
            if self.use_identity_memory_mask:
                attention_bias_type_cross = 'no_bias'

            decoder_layer_implementation: nn.Module
            if self.use_aligned_decoder:
                # hierarchical decoder, use an aligned implementation
                # this computes cross-attention only with tokens from the source
                # that directly condition underlying tokens in the target
                decoder_layer_implementation = (
                    TransformerAlignedDecoderLayerCustom)
            else:
                decoder_layer_implementation = (
                    TransformerDecoderLayerCustom)
            decoder_layer = decoder_layer_implementation(
                d_model=self.d_model,
                nhead=self.conditional_model_nhead,
                attention_bias_type_self='relative_attention',
                attention_bias_type_cross=attention_bias_type_cross,
                num_channels_encoder=self.source_num_channels,
                num_events_encoder=self.source_num_events_with_start_symbol,
                num_channels_decoder=self.target_num_channels,
                num_events_decoder=self.target_num_events_with_start_symbol
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

    def embed_data(self, input: torch.Tensor, kind: Seq2SeqInputKind) -> torch.Tensor:
        if kind == Seq2SeqInputKind.Source:
            return self.source_embeddings_linear(self.source_embed(input))
        elif kind == Seq2SeqInputKind.Target and self.conditional_model:
            return self.target_embeddings_linear(self.target_embed(input))
        else:
            raise ValueError(f"Unexpected value {kind} for kind option")

    def _get_combined_positional_embeddings(self, kind: Seq2SeqInputKind) -> torch.Tensor:
        if kind == Seq2SeqInputKind.Source:
            positional_embeddings_frequency = (
                self.source_positional_embeddings_frequency)
            positional_embeddings_time = self.source_positional_embeddings_time
            frequencies = self.source_frequencies
            duration = self.source_duration
        elif kind == Seq2SeqInputKind.Target and self.conditional_model:
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
            if kind == Seq2SeqInputKind.Target:
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
        return self._get_combined_positional_embeddings(Seq2SeqInputKind.Source)

    @property
    def combined_positional_embeddings_target(self) -> torch.Tensor:
        return self._get_combined_positional_embeddings(Seq2SeqInputKind.Target)

    @property
    def causal_mask(self) -> torch.Tensor:
        """Generate a mask to impose causality"""
        if self.conditional_model:
            # masking is applied only on the target, access is allowed to
            # all positions on the conditioning input
            causal_mask_length = (
                self.target_transformer_sequence_length_with_start_symbol)
        else:
            # apply causal mask to the input for the prediction task
            causal_mask_length = (
                self.source_transformer_sequence_length_with_start_symbol)

        mask = (torch.triu(torch.ones(causal_mask_length,
                                      causal_mask_length)) == 1
                ).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    @property
    def identity_memory_mask(self) -> torch.Tensor:
        identity_memory_mask = (
            torch.eye(self.source_transformer_sequence_length_with_start_symbol).float())
        identity_memory_mask = (
            identity_memory_mask
            .masked_fill(identity_memory_mask == 0, float('-inf'))
            .masked_fill(identity_memory_mask == 1, float(0.0))
        )
        return identity_memory_mask

    def to_sequences(self, input: torch.Tensor,
                     condition: Optional[torch.Tensor] = None,
                     class_conditioning: Mapping[str, torch.Tensor] = {},
                     mask: Optional[torch.BoolTensor] = None,
                     time_indexes_source: Optional[Iterable[int]] = None,
                     time_indexes_target: Optional[Iterable[int]] = None,
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        source_sequence = self.source_codemaps_helper.to_sequence(condition)
        if mask is not None and self.use_inpainting_mask_on_source:
            mask_sequence = self.source_codemaps_helper.to_sequence(mask)
        else:
            mask_sequence = None
        source_sequence, _ = self.prepare_data(
                source_sequence, kind=Seq2SeqInputKind.Source,
                class_conditioning=class_conditioning,
                mask=mask_sequence,
                time_indexes=time_indexes_source)

        target_sequence = self.target_codemaps_helper.to_sequence(input)
        target_sequence, _ = self.prepare_data(
            target_sequence, kind=Seq2SeqInputKind.Target,
            class_conditioning=class_conditioning,
            time_indexes=time_indexes_target)
        return source_sequence, target_sequence

    def prepare_data(self, sequence: torch.Tensor, kind: Seq2SeqInputKind,
                     class_conditioning: Mapping[str, torch.Tensor] = {},
                     mask: Optional[torch.Tensor] = None,
                     time_indexes: Optional[Iterable[int]] = None
                     ) -> torch.Tensor:
        if mask is not None:
            sequence = sequence.masked_fill(mask, self.mask_token_index)

        (batch_dim, sequence_dim, embedding_dim) = (0, 1, 2)

        embedded_sequence = self.embed_data(sequence, kind=kind)

        embedded_sequence_with_positions = self.add_positions_to_sequence(
            embedded_sequence, kind=kind, embedding_dim=embedding_dim,
            time_indexes=time_indexes
        )

        if self.positional_class_conditioning:
            embedded_sequence_with_positions = (
                self.add_class_conditioning_to_sequence(
                    embedded_sequence_with_positions,
                    class_conditioning)
            )

        prepared_sequence = self.add_start_symbol(
            embedded_sequence_with_positions, kind=kind,
            class_conditioning=class_conditioning, sequence_dim=1
        )

        frequency_dim, time_dim = (2, 1)
        return prepared_sequence, (
            (batch_dim, frequency_dim, time_dim))

    def add_positions_to_sequence(self, sequence: torch.Tensor,
                                  kind: Seq2SeqInputKind,
                                  embedding_dim: int,
                                  time_indexes: Optional[Iterable[int]]):
        # add positional embeddings
        batch_size = sequence.shape[0]
        # combine time and frequency embeddings
        if kind == Seq2SeqInputKind.Source:
            frequencies = self.source_frequencies
            duration = self.source_duration
            positional_embeddings = self.combined_positional_embeddings_source
            transformer_sequence_length = (
                self.source_transformer_sequence_length)
        elif kind == Seq2SeqInputKind.Target:
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
        if time_indexes is not None:
            # use non default time indexes, can be used e.g. when performing
            # predictions on sequences with a longer duration than the model's,
            # allowing to bias its operation to avoid introducing attacks or
            # releases in the middle of a longer sound
            positional_embeddings = positional_embeddings[..., time_indexes, :]

        if kind == Seq2SeqInputKind.Source:
            positions_as_sequence = self.source_codemaps_helper.to_sequence(
                positional_embeddings)
        elif kind == Seq2SeqInputKind.Target:
            positions_as_sequence = self.target_codemaps_helper.to_sequence(
                positional_embeddings)

        sequence_with_positions = torch.cat(
            [sequence, positions_as_sequence],
            dim=embedding_dim
        )

        return sequence_with_positions

    def add_class_conditioning_to_sequence(
            self, sequence_with_positions: torch.Tensor,
            class_conditioning: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Overwrite the end of the positional embeddings with the class"""
        embeddings = torch.zeros(
            (*sequence_with_positions.shape[:2],
             self.class_conditioning_total_dim),
            device=sequence_with_positions.device)
        for condition_name, class_condition in class_conditioning.items():
            modality_embeddings = (
                self.class_conditioning_embedding_layers[
                    condition_name](class_condition))
            start_position = (
                self.class_conditioning_start_positions_per_modality[
                    condition_name])
            embeddings[
                :, :, start_position:start_position+modality_embeddings.shape[2]] = (
                    modality_embeddings)
        return torch.cat([sequence_with_positions, embeddings], dim=-1)

    def add_start_symbol(self, sequence_with_positions: torch.Tensor,
                         kind: Seq2SeqInputKind,
                         class_conditioning: Mapping[str, torch.Tensor],
                         sequence_dim: int):
        batch_size = sequence_with_positions.shape[0]
        # combine time and frequency embeddings
        if kind == Seq2SeqInputKind.Source:
            start_symbol = self.source_start_symbol
            transformer_sequence_length = (
                self.source_transformer_sequence_length)
        elif kind == Seq2SeqInputKind.Target:
            start_symbol = self.target_start_symbol
            transformer_sequence_length = (
                self.target_transformer_sequence_length)
        else:
            raise ValueError(f"Unexpected value {kind} for kind option")

        # repeat start-symbol over whole batch
        start_symbol = start_symbol.repeat(batch_size, 1, 1)

        if not self.local_class_conditioning:
            if self.positional_class_conditioning:
                start_symbol = self.add_class_conditioning_to_sequence(
                    start_symbol, class_conditioning
                )
            else:
                # add conditioning tensors to start-symbol
                for condition_name, class_condition in class_conditioning.items():
                    embeddings = (
                        self.class_conditioning_embedding_layers[
                            condition_name](class_condition)).squeeze(1)
                    start_position = (
                        self.class_conditioning_start_positions_per_modality[
                            condition_name])
                    start_symbol[:, :, start_position:start_position+embeddings.shape[1]] = embeddings.unsqueeze(1)

        sequence_with_start_symbol = torch.cat(
                [start_symbol,
                 sequence_with_positions],
                dim=sequence_dim
        )
        return sequence_with_start_symbol

    def make_class_conditioning_sequence(self,
                                         class_conditioning: Mapping[
                                             str, torch.Tensor],
                                         ) -> torch.Tensor:
        """Convert multi-modal class-conditioning maps to a single sequence

        Class conditioning is only added to the source codemap
        """
        kind = Seq2SeqInputKind.Source

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

        embeddings_sequence = self.source_codemaps_helper.to_sequence(
            embeddings)

        class_embeddings_sequence_with_positions = (
            self.add_positions_to_sequence(embeddings_sequence,
                                           kind=kind,
                                           embedding_dim=-1))
        return class_embeddings_sequence_with_positions

    def forward(self, input: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                class_condition: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None):
        (batch_dim, sequence_dim) = (0, 1)

        target_sequence: Optional[torch.Tensor]
        if self.conditional_model:
            target_sequence = input
            source_sequence = condition
        else:
            source_sequence = input
            target_sequence = None
        assert source_sequence is not None

        # transformer inputs are in time-major format
        time_major_source_sequence = source_sequence.transpose(0, 1)
        if target_sequence is not None:
            time_major_target_sequence = target_sequence.transpose(0, 1)

        if class_condition is not None:
            time_major_class_condition_sequence = class_condition.transpose(0, 1)
        else:
            time_major_class_condition_sequence = None
        (batch_dim, sequence_dim) = (1, 0)

        memory_mask = None
        causal_mask = self.causal_mask.to(input.device)
        if self.conditional_model:
            if self.use_identity_memory_mask:
                memory_mask = self.identity_memory_mask
            if memory is None:
                src_mask = None
                if self.self_conditional_model:
                    anti_causal_mask = causal_mask.t()
                    src_mask = anti_causal_mask
                memory = self.transformer.encoder(
                    time_major_source_sequence,
                    mask=src_mask)
                if self.use_relative_transformer:
                    memory, *encoder_attentions = memory

            if time_major_class_condition_sequence is not None:
                output_sequence = self.transformer.decoder(
                    time_major_target_sequence,
                    memory,
                    tgt_mask=causal_mask,
                    memory_mask=memory_mask,
                    condition=time_major_class_condition_sequence)
            else:
                output_sequence = self.transformer.decoder(
                    time_major_target_sequence,
                    memory,
                    tgt_mask=causal_mask,
                    memory_mask=memory_mask)
        else:
            output_sequence = self.transformer(time_major_source_sequence,
                                               mask=causal_mask)
        if self.use_relative_transformer:
            output_sequence, *decoder_attentions = output_sequence

        # trim start symbol
        target_start_symbol_duration = self.target_start_symbol.shape[1]
        output_sequence = output_sequence[target_start_symbol_duration-1:]
        # trim last token, unused in next-token prediction task
        output_sequence = output_sequence[:-1]

        # transpose back to batch-major shape
        output_sequence = output_sequence.transpose(
            batch_dim, sequence_dim)
        (batch_dim, sequence_dim) = (0, 1)

        # convert outputs to class probabilities
        logits = self.project_transformer_outputs_to_logits(output_sequence)

        return logits, memory

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


class SelfAttentiveVQTransformer(VQNSynthTransformer):
    @property
    def use_inpainting_mask_on_source(self) -> bool:
        """Use inpainting mask-token in self-attentive regeneration
        """
        return True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source_codemaps_helper = self.target_codemaps_helper = (
            SimpleCodemapsHelper(self.source_frequencies,
                                 self.source_duration)
        )


class UpsamplingVQTransformer(VQNSynthTransformer):
    @property
    def use_inpainting_mask_on_source(self) -> bool:
        """No inpainting mask for upsampling Transformers

        The whole conditioning information ishould bhe available since
        upsampling is performed after generation of the conditioning source.

        Only attention-masking is performed in the Upsampling Transformers.
        """
        return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source_codemaps_helper = SimpleCodemapsHelper(
            self.source_frequencies,
            self.source_duration)

        self.target_codemaps_helper = ZigZagCodemapsHelper(
            self.target_frequencies,
            self.target_duration,
            self.target_frequencies // self.source_frequencies,
            self.target_duration // self.source_duration
        )
