from typing import Union, Optional, Iterable, Tuple, Mapping
import argparse
import pathlib
import os
import json
from datetime import datetime
import uuid
import soundfile
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

from dataset import LMDBDataset
from vqvae import VQVAE
from pixelsnail import PixelSNAIL
from transformer import VQNSynthTransformer
import utils as vqvae_utils

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# use matplotlib without an X server
# on desktop, this prevents matplotlib windows from popping around
mpl.use('Agg')


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x sequence_duration x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def make_conditioning_tensors(
        class_conditioning: Mapping[str, Union[int, str, Tuple[int, int]]],
        label_encoders_per_conditioning: Mapping[str, LabelEncoder],
        ) -> Mapping[str, torch.Tensor]:
    class_conditioning_tensors = {}

    def make_conditioning_tensor(value: Union[int, str, Tuple[int, int]],
                                 modality: str) -> torch.Tensor:
        label_encoder = label_encoders_per_conditioning[modality]
        try:
            # check if value is a 2-uple (useful for pitch ranges)
            range_min, range_max = tuple(int(x) for x in value)
            assert range_min < range_max, (
                "Provide increasing range for range conditioning")
            encoded_label = label_encoder.transform(
                list(range(range_min, range_max))).transpose()
        except BaseException:
            encoded_label = label_encoder.transform([value])

        return torch.from_numpy(encoded_label).long()

    # for value, modality, location in zip(
    #     [args.instrument_family_conditioning_top,
    #      args.instrument_family_conditioning_bottom,
    #      args.pitch_conditioning_top, args.pitch_conditioning_bottom,
    #      ],
    #     ['instrument_family_str', 'instrument_family_str', 'pitch', 'pitch'],
    #     ['top', 'bottom', 'top', 'bottom']
    # ):
    #     maybe_add_conditioning(value, modality, location)

    for modality, value in class_conditioning.items():
        class_conditioning_tensors[modality] = (
            make_conditioning_tensor(value, modality))

    return class_conditioning_tensors


ConditioningMap = Union[Iterable[Iterable[str]],
                        Iterable[Iterable[int]]]


def make_conditioning_map(class_conditioning: Mapping[str, ConditioningMap],
                          label_encoders_per_conditioning: Mapping[
                              str, LabelEncoder],
                          ) -> Mapping[str, torch.Tensor]:
    def map_to_tensor(conditioning_map: ConditioningMap, modality: str):
        label_encoder = label_encoders_per_conditioning[modality]

        num_rows = len(conditioning_map)
        num_columns = len(conditioning_map[0])
        conditioning_tensor = torch.zeros(num_rows, num_columns).long()

        for row_index, row in enumerate(conditioning_map):
            encoded_row = label_encoder.transform(row)
            conditioning_tensor[row_index] = torch.from_numpy(encoded_row)

        return conditioning_tensor.unsqueeze(0)  # prepare in batched format

    return {modality: map_to_tensor(conditioning_map, modality)
            for modality, conditioning_map in class_conditioning.items()}


@torch.no_grad()
def sample_model(model: PixelSNAIL, device: Union[torch.device, str],
                 batch_size: int, codemap_size: Iterable[int],
                 temperature: float, condition: Optional[torch.Tensor] = None,
                 constraint: Optional[torch.Tensor] = None,
                 class_conditioning: Mapping[str, Iterable[int]] = {},
                 initial_code: Optional[torch.Tensor] = None,
                 mask: Optional[torch.Tensor] = None,
                 local_class_conditioning_map: Optional[Mapping[str, Iterable[int]]] = None,
                 top_k_sampling_k: int = 0,
                 top_p_sampling_p: float = 0.0
                 ):
    """Generate a sample from the provided PixelSNAIL

    Arguments:
        model (PixelSNAIL)
        device (torch.device or str):
            The device on which to perform the sampling
        batch_size (int)
        codemap_size (Iterable[int]):
            The size of the codemap to generate
        temperature (float):
            Sampling temperature (lower means the model is more conservative)
        condition (torch.Tensor, optional, default None):
            Another codemap to use as hierarchical conditionning for sampling.
            If not provided, sampling is unconditionned.
        constraint_2D (torch.Tensor, optional, default None):
            If provided, fixes the top-left part of the generated 2D codemap
            to be the given Tensor.
            `constraint_2D.size` should be less or equal to codemap_size.
    """
    if initial_code is None:
        codemap = (torch.full([batch_size] + list(codemap_size),
                              fill_value=model.mask_token_index if model.self_conditional_model else 0,
                              dtype=torch.int64)
                   .to(device)
                   )
    else:
        codemap = initial_code.to(device)

    if not model.local_class_conditioning or (
            local_class_conditioning_map is None):
        class_conditioning_tensors = {
            conditioning_modality: (
                conditioning_tensor.unsqueeze(1).long()
                .expand(batch_size, -1)
                .to(device))
            for conditioning_modality, conditioning_tensor
            in class_conditioning.items()
        }
    else:
        class_conditioning_tensors = local_class_conditioning_map
    parallel_model = nn.DataParallel(model)
    parallel_model.eval()

    constraint_height = 0
    constraint_width = 0
    if constraint is not None:
        raise NotImplementedError

        if list(constraint.shape) > codemap_size:
            raise ValueError("Incorrect size of constraint, constraint "
                             "should be smaller than the target codemap size")
        else:
            _, constraint_height, constraint_width = constraint.shape

            padding_left = padding_top = 0
            padding_bottom = codemap_size[0] - constraint_height
            padding_right = codemap_size[1] - constraint_width
            padder = torch.nn.ConstantPad2d(
                (padding_left, padding_right, padding_top, padding_bottom),
                value=0).to(device)
            channel_dim = 1
            codemap = (padder(constraint.unsqueeze(channel_dim))
                       .detach()
                       .squeeze(channel_dim)
                       )

    if model.self_conditional_model:
        condition = codemap

    sequence_duration = codemap_size[0] * codemap_size[1]
    source_sequence, target_sequence = model.to_sequences(
        codemap, condition,
        class_conditioning=class_conditioning_tensors,
        mask=mask
    )

    if model.conditional_model:
        kind = 'target'
        input_sequence = target_sequence
        condition_sequence = source_sequence
    else:
        kind = 'source'
        input_sequence = source_sequence
        condition_sequence = None

    sequence_duration_without_start_symbol = (
        model.target_transformer_sequence_length)

    source_start_symbol_duration = model.source_start_symbol.shape[1]
    target_start_symbol_duration = model.target_start_symbol.shape[1]

    codemap_as_sequence = model.flatten_map(codemap, kind=kind)

    if mask is not None:
        mask = model.flatten_map(mask, kind=kind).squeeze(0)
    else:
        mask = torch.full((sequence_duration_without_start_symbol, ),
                          True, dtype=bool)

    class_condition_sequence = None
    if model.local_class_conditioning:
        class_condition_sequence = (
            model.make_condition_sequence(class_conditioning_tensors)
            )

    memory = None
    for i in tqdm(range(sequence_duration_without_start_symbol)):
        if not mask[i]:
            continue

        logits_sequence_out, memory = parallel_model(
            input_sequence, condition_sequence,
            class_condition_sequence,
            memory=memory)

        # apply temperature and filter logits
        logits_sequence_out = logits_sequence_out / temperature
        logits_sequence_out = top_k_top_p_filtering(logits_sequence_out,
                                                    top_k=top_k_sampling_k,
                                                    top_p=top_p_sampling_p)

        next_step_probabilities = torch.softmax(
            logits_sequence_out[:, i, :], dim=1)

        sample = torch.multinomial(next_step_probabilities, 1).squeeze(-1)
        codemap_as_sequence[:, i] = sample.long()

        embedded_sample = model.embed_data(sample, kind)

            # translate to account for the added start_symbol!
        input_sequence[:, i+target_start_symbol_duration, :model.embeddings_effective_dim] = (
                embedded_sample)
            if model.self_conditional_model:
            condition_sequence[:, i+source_start_symbol_duration, :model.embeddings_effective_dim] = (
                    embedded_sample)
                # the cached memory remains valid here,
                # because the Top encoder uses anti-causal attention

    codemap = model.to_time_frequency_map(codemap_as_sequence,
                                          kind=kind).long()

    return codemap


def plot_codes(top_codes: torch.LongTensor,
               bottom_codes: torch.LongTensor,
               codes_dictionary_dim_top: int,
               codes_dictionary_dim_bottom: int,
               cmap='viridis', plots_per_row: int = 12) -> None:
    assert (len(top_codes)
            == len(bottom_codes))

    num_maps = len(top_codes)
    num_groups = 2
    plots_per_row = min(num_maps, plots_per_row)
    num_rows_per_codemaps_group = int(np.ceil(num_maps / plots_per_row))
    num_rows = num_groups * num_rows_per_codemaps_group

    figure, subplot_axs = plt.subplots(num_rows, plots_per_row,
                                       figsize=(10 * plots_per_row/12,
                                                2*num_rows))
    for ax in subplot_axs.ravel().tolist():
        ax.set_axis_off()

    def get_ax(codemap_group_index: int, codemap_index: int):
        start_row = codemap_group_index * num_rows_per_codemaps_group
        row = start_row + codemap_index // plots_per_row
        ax = subplot_axs[row][codemap_index % plots_per_row]
        return ax

    for group_index, (maps_group, codes_dictionary_dim) in enumerate(
            zip([top_codes, bottom_codes],
                [codes_dictionary_dim_top,
                 codes_dictionary_dim_bottom])):
        for map_index, codemap in enumerate(maps_group):
            ax = get_ax(group_index, map_index)
            im = ax.matshow(codemap.cpu().numpy(), vmin=0,
                            vmax=codes_dictionary_dim-1,
                            cmap=cmap)

    figure.tight_layout()
    # add colorbar for codemaps
    figure.colorbar(im,
                    ax=(subplot_axs.ravel().tolist()))
    return figure, subplot_axs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dataset', type=str, choices=['nsynth', 'imagenet'],
                        required=True)
    parser.add_argument('--model_type_top', type=str,
                        choices=['PixelSNAIL', 'Transformer'],
                        default='PixelSNAIL')
    parser.add_argument('--model_type_bottom', type=str,
                        choices=['PixelSNAIL', 'Transformer'],
                        default='PixelSNAIL')
    parser.add_argument('--vqvae_training_parameters_path', type=str, required=True)
    parser.add_argument('--vqvae_model_parameters_path', type=str, required=True)
    parser.add_argument('--vqvae_weights_path', type=str, required=True)
    parser.add_argument('--prediction_top_parameters_path', type=str,
                        required=True)
    parser.add_argument('--prediction_top_weights_path', type=str,
                        required=True)
    parser.add_argument('--prediction_bottom_parameters_path', type=str,
                        required=True)
    parser.add_argument('--prediction_bottom_weights_path', type=str,
                        required=True)

    def key_value(arg: str) -> Iterable[Tuple[str, str]]:
        key, value = arg.split(',')
        if len(value.split('...')) == 2:
            value = value.split('...')
        return key, value

    parser.add_argument('--class_conditioning', type=key_value, nargs='*',
                        default=[])
    parser.add_argument('--class_conditioning_top', type=key_value, nargs='*',
                        default=[])
    parser.add_argument('--keep_same_top', action='store_true')
    parser.add_argument('--class_conditioning_bottom', type=key_value, nargs='*',
                        default=[])
    # TODO(theis): change this, store label encoders inside the VQNSynthTransformer model class
    parser.add_argument('--database_path_for_label_encoders', type=str)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p_sampling_p', type=float, default=0.0)
    parser.add_argument('--top_k_sampling_k', type=int, default=0)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--sample_rate_hz', type=int, default=16000)
    parser.add_argument('--condition_top_audio_path', type=str)
    parser.add_argument('--constraint_top_audio_path', type=str)
    parser.add_argument('--constraint_top_num_timesteps', type=int)
    parser.add_argument('--output_directory', type=str, default='./')

    args = parser.parse_args()

    def expand_path(path: str) -> pathlib.Path:
        return pathlib.Path(path).expanduser().absolute()

    OUTPUT_DIRECTORY = expand_path(args.output_directory)

    run_ID = (datetime.now().strftime('%Y%m%d-%H%M%S-')
              + str(uuid.uuid4())[:6])
    print("Sample ID: ", run_ID)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model_type_top == 'PixelSNAIL':
        ModelTop = PixelSNAIL
    elif args.model_type_top == 'Transformer':
        ModelTop = VQNSynthTransformer
    else:
        raise ValueError(
            f"Unexpected value {args.model_type_top} for option model_type_top")

    if args.model_type_bottom == 'PixelSNAIL':
        ModelBottom = PixelSNAIL
    elif args.model_type_bottom == 'Transformer':
        ModelBottom = VQNSynthTransformer
    else:
        raise ValueError(
            f"Unexpected value {args.model_type_bottom} for option model_type_bottom")

    model_vqvae = VQVAE.from_parameters_and_weights(
        expand_path(args.vqvae_model_parameters_path),
        expand_path(args.vqvae_weights_path),
        device=device
        ).to(device).eval()
    model_top = ModelTop.from_parameters_and_weights(
        expand_path(args.prediction_top_parameters_path),
        expand_path(args.prediction_top_weights_path),
        device=device
        ).to(device).eval()
    model_bottom = ModelBottom.from_parameters_and_weights(
        expand_path(args.prediction_bottom_parameters_path),
        expand_path(args.prediction_bottom_weights_path),
        device=device
        ).to(device).eval()

    VQVAE_TRAINING_PARAMETERS_PATH = expand_path(
        args.vqvae_training_parameters_path)
    # retrieve n_fft, hop length, window length parameters...
    with open(VQVAE_TRAINING_PARAMETERS_PATH, 'r') as f:
        vqvae_training_parameters = json.load(f)
    spectrograms_helper = vqvae_utils.get_spectrograms_helper(
        device=device, **vqvae_training_parameters)

    def to_dictionary(key_value_list: Iterable[Tuple[any, any]]
                      ) -> Mapping[any, any]:
        return {key: value for key, value in key_value_list}

    if len(args.class_conditioning_top) > 0:
        assert len(args.class_conditioning_bottom) > 0
        class_conditioning_top = to_dictionary(args.class_conditioning_top)
        class_conditioning_bottom = to_dictionary(
            args.class_conditioning_bottom)
    else:
        # use same conditioning for top and bottom
        class_conditioning_top = to_dictionary(args.class_conditioning)
        class_conditioning_bottom = class_conditioning_top

    classes_for_conditioning = set()
    classes_for_conditioning.update(class_conditioning_top.keys())
    classes_for_conditioning.update(class_conditioning_bottom.keys())
    # additional_modalities = set(modality
    #                             for modality, _ in )
    # classes_for_conditioning.update(additional_modalities)

    if args.database_path_for_label_encoders is not None:
        DATABASE_PATH = expand_path(args.database_path_for_label_encoders)
        dataset = LMDBDataset(
            DATABASE_PATH,
            classes_for_conditioning=list(classes_for_conditioning)
        )
        label_encoders_per_conditioning = dataset.label_encoders

    class_conditioning_tensors_top = make_conditioning_tensors(
        class_conditioning_top,
        label_encoders_per_conditioning)
    class_conditioning_tensors_bottom = make_conditioning_tensors(
        class_conditioning_bottom,
        label_encoders_per_conditioning)

    with torch.no_grad():
        initial_code = None
        if args.condition_top_audio_path is not None:
            condition_mel_spec_and_IF = spectrograms_helper.from_wavfile(
                args.condition_top_audio_path)

            (_, _, _, condition_code_top, condition_code_bottom,
             *_) = model_vqvae.encode(condition_mel_spec_and_IF.to(device))

            # repeat condition for the whole batch
            top_code = condition_code_top.repeat(args.batch_size, 1, 1)
            initial_code = condition_code_bottom.repeat(args.batch_size, 1, 1)
        elif args.constraint_top_audio_path is not None:
            constraint_mel_spec_and_IF = spectrograms_helper.from_wavfile(
                args.constraint_top_audio_path)

            (_, _, _, constraint_code_top, *_) = model_vqvae.encode(
                constraint_mel_spec_and_IF.to(device))
            constraint_code_top_restrained = (
                constraint_code_top[:, :args.constraint_top_num_timesteps-1])
            top_code_sample = sample_model(
                model_top, device, batch_size=1, codemap_size=model_top.shape,
                temperature=args.temperature,
                constraint=constraint_code_top_restrained,
                class_conditioning=class_conditioning_tensors_top,
                top_p_sampling_p=args.top_p_sampling_p,
                top_k_sampling_k=args.top_k_sampling_k
                )

            # repeat condition for the whole batch
            top_code = top_code_sample.repeat(args.batch_size, 1, 1)
        else:
            batch_size_top = args.batch_size
            if args.keep_same_top:
                batch_size_top = 1
            top_code_sample = sample_model(
                model_top, device, batch_size_top, model_top.shape,
                args.temperature,
                class_conditioning=class_conditioning_tensors_top,
                top_p_sampling_p=args.top_p_sampling_p,
                top_k_sampling_k=args.top_k_sampling_k)
            top_code = top_code_sample

            if args.keep_same_top:
                top_code = top_code.repeat(args.batch_size, 1, 1)

        # sample bottom code contitioned on the top code
        bottom_sample = sample_model(
            model_bottom, device, args.batch_size, model_bottom.shape,
            args.temperature, condition=top_code,
            class_conditioning=class_conditioning_tensors_bottom,
            initial_code=initial_code,
            top_p_sampling_p=args.top_p_sampling_p,
            top_k_sampling_k=args.top_k_sampling_k
        )

        decoded_sample = model_vqvae.decode_code(top_code, bottom_sample)

    codes_figure, _ = plot_codes(top_code, bottom_sample,
                                 model_top.n_class,
                                 model_bottom.n_class)

    condition_top_audio = None
    if args.condition_top_audio_path is not None:
        CONDITION_TOP_AUDIO_PATH = expand_path(args.condition_top_audio_path)
        import torchvision.transforms as transforms
        sample_audio, fs_hz = torchaudio.load_wav(CONDITION_TOP_AUDIO_PATH,
                                                  channels_first=True)
        resampler = torchaudio.transforms.Resample(
            orig_freq=fs_hz, new_freq=args.sample_rate_hz)
        sample_audio = resampler(sample_audio.cuda())
        toFloat = transforms.Lambda(lambda x: (x / np.iinfo(np.int16).max))
        sample_audio = toFloat(sample_audio)
        condition_top_audio = sample_audio.flatten().cpu().numpy()

    def make_audio(mag_and_IF_batch: torch.Tensor,
                   condition_audio: Optional[np.ndarray],
                   normalize: bool = False) -> np.ndarray:
        audio_batch = spectrograms_helper.to_audio(mag_and_IF_batch)

        if normalize:
            normalized_audio_batch = (
                audio_batch
                / audio_batch.abs().max(dim=1, keepdim=True)[0])
            audio_batch = normalized_audio_batch

        audio_mono_concatenated = audio_batch.flatten().cpu().numpy()
        if condition_audio is not None:
            audio_mono_concatenated = np.concatenate(
                [condition_audio,
                 np.zeros(condition_audio.shape),
                 audio_mono_concatenated])
        return audio_mono_concatenated

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    with open(OUTPUT_DIRECTORY / f'{run_ID}-command_line_parameters.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    codes_figure.savefig(OUTPUT_DIRECTORY / f'{run_ID}-codemaps.png')

    if args.dataset == 'nsynth':
        audio_sample_path = OUTPUT_DIRECTORY / f'{run_ID}.wav'
        soundfile.write(audio_sample_path,
                        make_audio(decoded_sample, condition_top_audio),
                        samplerate=args.sample_rate_hz)

        # write spectrogram and IF
        channel_dim = 1
        for channel_index, channel_name in enumerate(
                ['spectrogram', 'instantaneous_frequency']):
            channel = decoded_sample.select(channel_dim, channel_index
                                            ).unsqueeze(channel_dim)
            save_image(
                channel,
                os.path.join(args.output_directory, f'{run_ID}-{channel_name}.png'),
                nrow=args.batch_size,
                # normalize=True,
                # range=(-1, 1),
                # scale_each=True,
            )
    elif args.dataset == 'imagenet':
        image_sample_path = os.path.join(args.output_directory, f'{run_ID}.png')
        save_image(
                decoded_sample,
                image_sample_path,
                nrow=args.batch_size
            )
