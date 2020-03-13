from typing import Mapping, List, Optional, Union
import torch
from torch import nn, optim
from transformer import VQNSynthTransformer

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


def check_equality(a: torch.Tensor, b: torch.Tensor,
                   kind: str, layer: str) -> None:
    if not torch.equal(a, b):
        print(layer, kind)
        print(a[0, 0])
        print(b[0, 0])
        assert False


batch_size = 2
embedding_dim = 3

top_model_parameters = bottom_model_parameters.copy()
top_model_parameters['self_conditional_model'] = True
top_model_parameters['add_mask_token_to_symbols'] = True

for condition_shape in [[32, 4], [64, 8], [128, 16]]:
    bottom_model_parameters['condition_shape'] = condition_shape
    top_model_parameters['shape'] = condition_shape
    top_model_parameters['condition_shape'] = condition_shape

    top_model = VQNSynthTransformer(**top_model_parameters)

    top_frequencies, top_duration = condition_shape

    top_codemap = (torch.arange(top_frequencies * top_duration)
                   .reshape(1, top_frequencies, top_duration, 1)
                   .repeat(batch_size, 1, 1, embedding_dim))

    layer = 'top'
    kind = 'source'
    top_flattened = top_model.flatten_map(top_codemap,
                                          kind=kind)
    top_remapped = top_model.to_time_frequency_map(top_flattened,
                                                   kind=kind)
    check_equality(top_codemap, top_remapped, kind, layer)

    kind = 'target'
    top_flattened_as_target = top_model.flatten_map(top_codemap,
                                                    kind=kind)
    top_remapped_as_target = top_model.to_time_frequency_map(
        top_flattened_as_target, kind=kind)
    check_equality(top_codemap, top_remapped_as_target, kind, layer)

    layer = 'bottom'
    kind = 'target'

    for shape in [[64, 8], [128, 16], [256, 32]]:
        if condition_shape[0] >= shape[0]:
            continue
        bottom_model_parameters['shape'] = shape
        bottom_model = VQNSynthTransformer(**bottom_model_parameters)

        bottom_frequencies, bottom_duration = shape
        bottom_codemap = (torch.arange(bottom_frequencies * bottom_duration)
                          .reshape(1, bottom_frequencies, bottom_duration, 1)
                          .repeat(batch_size, 1, 1, embedding_dim))

        bottom_flattened = bottom_model.flatten_map(bottom_codemap,
                                                    kind=kind)
        bottom_remapped = bottom_model.to_time_frequency_map(bottom_flattened,
                                                             kind=kind)
        check_equality(bottom_codemap, bottom_remapped, kind, layer)
