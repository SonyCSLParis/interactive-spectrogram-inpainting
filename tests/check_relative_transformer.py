from typing import Mapping, List, Optional, Union
import torch
from torch import nn, optim
from transformer import SelfAttentiveVQTransformer, UpsamplingVQTransformer

bottom_model_parameters: Mapping[str, Optional[Union[int,
                                                     List[int],
                                                     Mapping[str, int]]]]
bottom_model_parameters = {
    "n_class": 512,
    "channel": 256,
    "kernel_size": 5,
    "n_block": 4,
    "n_res_block": 4,
    "res_channel": 256,

    "shape": None,
    "condition_shape": None,

    "conditional_model": True,
    "use_relative_transformer": True,
    "predict_frequencies_first": True,
    "predict_low_frequencies_first": True,
    "class_conditioning_prepend_to_dummy_input": True,

    "class_conditioning_num_classes_per_modality": {
        "instrument_family_str": 11,
        "pitch": 61
    },
    "class_conditioning_embedding_dim_per_modality": {
        "instrument_family_str": 64,
        "pitch": 64
    },
}


def check_equality_codemap(a: torch.Tensor, b: torch.Tensor,
                   kind: str, layer: str) -> None:
    if not torch.equal(a, b):
        print(layer, kind)
        print(a[0, 0])
        print(b[0, 0])
        assert False


def check_equality_flattened(a: torch.Tensor, b: torch.Tensor,
                             kind: str, layer: str) -> None:
    if not torch.equal(a, b):
        print(layer, kind)
        print(a[:, 0])
        print(b)
        assert False


batch_size = 2
embedding_dim = 3

top_model_parameters = bottom_model_parameters.copy()
for condition_shape in [[32, 4], [64, 8], [128, 16]]:
    bottom_model_parameters['condition_shape'] = condition_shape
    top_model_parameters['shape'] = condition_shape
    top_model_parameters['condition_shape'] = condition_shape

    top_model = SelfAttentiveVQTransformer(**top_model_parameters)

    top_frequencies, top_duration = condition_shape

    top_codemap = (torch.arange(top_frequencies * top_duration)
                   .reshape(1, top_frequencies, top_duration, 1)
                   .repeat(batch_size, 1, 1, embedding_dim))

    layer = 'top'
    kind = 'source'
    top_flattened = top_model.source_codemaps_helper.to_sequence(
        top_codemap)
    top_remapped = top_model.source_codemaps_helper.to_time_frequency_map(
        top_flattened)
    check_equality_codemap(top_codemap, top_remapped, kind, layer)

    kind = 'target'
    top_flattened_as_target = top_model.target_codemaps_helper.to_sequence(
        top_codemap)
    top_remapped_as_target = top_model.target_codemaps_helper.to_time_frequency_map(
        top_flattened_as_target)
    check_equality_codemap(top_codemap, top_remapped_as_target, kind, layer)

    layer = 'bottom'
    kind = 'target'

    for shape in [[64, 8], [128, 16], [256, 32]]:
        if condition_shape[0] >= shape[0]:
            continue
        bottom_model_parameters['shape'] = shape
        bottom_model = UpsamplingVQTransformer(**bottom_model_parameters)

        bottom_frequencies, bottom_duration = shape
        bottom_codemap = (torch.arange(bottom_frequencies * bottom_duration)
                          .reshape(1, bottom_frequencies, bottom_duration, 1)
                          .repeat(batch_size, 1, 1, embedding_dim))

        bottom_flattened = bottom_model.target_codemaps_helper.to_sequence(
            bottom_codemap)
        target_duration_per_patch = bottom_model.target_duration // bottom_model.source_duration
        target_frequencies_per_patch = bottom_model.target_frequencies // bottom_model.source_frequencies

        if bottom_model.predict_frequencies_first:
            target_expected_first_patch = (
                torch.arange(target_frequencies_per_patch).unsqueeze(1)
                + (torch.arange(target_duration_per_patch).unsqueeze(0) * bottom_model.target_duration)
                ).flatten()
        else:
            target_expected_first_patch = (
                torch.arange(target_duration_per_patch).unsqueeze(0)
                + (torch.arange(target_frequencies_per_patch).unsqueeze(1) * bottom_model.target_frequencies)
                ).flatten()
        check_equality_flattened(
            bottom_flattened[0, :, 0][:bottom_model.target_events_per_source_patch],
            target_expected_first_patch,
            kind, layer)

        bottom_remapped = bottom_model.target_codemaps_helper.to_time_frequency_map(
            bottom_flattened)
        check_equality_codemap(bottom_codemap, bottom_remapped, kind, layer)
