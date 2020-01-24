import torch
from torch import nn, optim
from transformer import VQNSynthTransformer

model_parameters = {
    "shape": [
        64,
        8
    ],
    "conditional_model": True,
    "use_relative_transformer": True,
    "condition_shape": [
        32,
        4
    ],
    "n_class": 512,
    "channel": 256,
    "kernel_size": 5,
    "n_block": 4,
    "n_res_block": 4,
    "res_channel": 256,
    "attention": False,
    "dropout": 0.1,
    "n_cond_res_block": 3,
    "cond_res_channel": 256,
    "cond_res_kernel": 3,
    "n_out_res_block": 0,
    "predict_frequencies_first": True,
    "predict_low_frequencies_first": True,
    "d_model": 512,
    "embeddings_dim": 32,
    "positional_embeddings_dim": 16,
    "class_conditioning_num_classes_per_modality": {
        "instrument_family_str": 11,
        "pitch": 61
    },
    "class_conditioning_embedding_dim_per_modality": {
        "instrument_family_str": 64,
        "pitch": 64
    },
    "class_conditioning_prepend_to_dummy_input": True,
    "conditional_model_num_encoder_layers": 4,
    "conditional_model_nhead": 8,
    "conditional_model_num_decoder_layers": 6
}

model = VQNSynthTransformer(**model_parameters)

top_frequencies, top_duration = model_parameters['condition_shape']
bottom_frequencies, bottom_duration = model_parameters['shape']

top_codemap = (torch.arange(top_frequencies * top_duration)
               .reshape(1, top_frequencies, top_duration, 1)
               .repeat(2, 1, 1, 3))

bottom_codemap = (torch.arange(bottom_frequencies * bottom_duration)
                  .reshape(1, bottom_frequencies, bottom_duration, 1)
                  .repeat(2, 1, 1, 3))

kind = 'target'
bottom_flattened, _ = model.flatten_map(bottom_codemap,
                                        kind=kind)

bottom_remapped = model.to_time_frequency_map(bottom_flattened,
                                              kind=kind)

print(bottom_codemap[0, 0])
print(bottom_remapped[0, 0])

STOP