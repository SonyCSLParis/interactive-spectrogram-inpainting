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

import torch
import torchaudio
from torch import nn
from torchvision.utils import save_image

from dataset import LMDBDataset
from vqvae import VQVAE, InferenceVQVAE
from pixelsnail import PixelSNAIL
from transformer import VQNSynthTransformer
from GANsynth_pytorch.pytorch_nsynth_lib.nsynth import (
    wavfile_to_melspec_and_IF)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# use matplotlib without an X server
# on desktop, this prevents matplotlib windows from popping around
mpl.use('Agg')


@torch.no_grad()
def sample_model(model: PixelSNAIL, device: Union[torch.device, str],
                 batch_size: int, codemap_size: Iterable[int],
                 temperature: float, condition: Optional[torch.Tensor] = None,
                 constraint: Optional[torch.Tensor] = None,
                 class_conditioning: Mapping[str, Iterable[int]] = {}):
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
    codemap = (torch.zeros(batch_size, *codemap_size, dtype=torch.int64)
               .to(device)
               )
    class_conditioning_tensors = {
        conditioning_modality: (
            conditioning_tensor.long()
            .repeat(batch_size, 1)
            .to(device))
        for conditioning_modality, conditioning_tensor
        in class_conditioning.items()
    }
    parallel_model = nn.DataParallel(model)

    constraint_height = 0
    constraint_width = 0
    if constraint is not None:
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

    cache = {}

    sequence_duration = codemap_size[0] * codemap_size[1]
    codemap_as_sequence = torch.zeros(batch_size, sequence_duration).long()
    source_sequence, target_sequence = model.to_sequences(
        codemap, condition, class_conditioning=class_conditioning_tensors
    )

    if model.conditional_model:
        kind = 'target'
        input_sequence = target_sequence
        condition_sequence = source_sequence
    else:
        kind = 'source'
        input_sequence = source_sequence
        condition_sequence = None

    for i in tqdm(range(sequence_duration)):
        logits_sequence_out, _ = parallel_model(
            input_sequence, condition_sequence,
            cache=cache,
            class_conditioning=class_conditioning)

        next_step_probabilities = torch.softmax(
            logits_sequence_out[:, i, :] / temperature,
            1)
        sample = torch.multinomial(next_step_probabilities, 1).squeeze(-1)

        codemap_as_sequence[:, i] = sample.long()

        embedded_sample = model.embed_data(sample, kind)
        if i+1 < sequence_duration:
            # translate to account for the added start_symbol!
            input_sequence[:, i+1, :-model.positional_embeddings_dim] = (
                embedded_sample)

    codemap = model.to_time_frequency_map(codemap_as_sequence).long()

    # if model.predict_frequencies_first:
    #     for j in tqdm(range(codemap_size[1]), position=0):
    #         start_row = (0 if j >= constraint_width
    #                      else constraint_height)
    #         for i in tqdm(range(start_row, codemap_size[0]), position=1):
    #             out, cache = parallel_model(
    #                 codemap, condition=condition,
    #                 cache=cache,
    #                 class_conditioning=class_conditioning)
    #             prob = torch.softmax(out[:, :, i, j] / temperature, 1)
    #             sample = torch.multinomial(prob, 1).squeeze(-1)
    #             codemap[:, i, j] = sample
    # else:
    #     for i in tqdm(range(codemap_size[0]), position=0):
    #         start_column = (0 if i >= constraint_height
    #                         else constraint_width)
    #         for j in tqdm(range(start_column, codemap_size[1]), position=1):
    #             out, cache = parallel_model(
    #                 codemap, condition=condition,
    #                 cache=cache,
    #                 class_conditioning=class_conditioning)
    #             prob = torch.softmax(out[:, :, i, j] / temperature, 1)
    #             sample = torch.multinomial(prob, 1).squeeze(-1)
    #             codemap[:, i, j] = sample

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
    parser.add_argument('--vqvae_parameters_path', type=str, required=True)
    parser.add_argument('--vqvae_weights_path', type=str, required=True)
    parser.add_argument('--prediction_top_parameters_path', type=str,
                        required=True)
    parser.add_argument('--prediction_top_weights_path', type=str,
                        required=True)
    parser.add_argument('--prediction_bottom_parameters_path', type=str,
                        required=True)
    parser.add_argument('--prediction_bottom_weights_path', type=str,
                        required=True)
    parser.add_argument('--pitch_conditioning_top', type=int, default=None)
    parser.add_argument('--instrument_family_conditioning_top', type=str,
                        default=None)
    parser.add_argument('--pitch_conditioning_bottom', type=int, default=None)
    parser.add_argument('--instrument_family_conditioning_bottom', type=str,
                        default=None)

    def key_value(arg: str) -> Iterable[Tuple[str, str]]:
        key, value = arg.split(',')
        return str(key), str(value)

    parser.add_argument('--class_conditioning', type=key_value, nargs='*',
                        default=[])
    # TODO(theis): change this, store label encoders inside the VQNSynthTransformer model class
    parser.add_argument('--database_path_for_label_encoders', type=str)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--use_mel_frequency', type=int, default=True)
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
        expand_path(args.vqvae_parameters_path),
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

    classes_for_conditioning = set()
    if args.pitch_conditioning_top is not None or args.pitch_conditioning_bottom is not None:
        classes_for_conditioning.add('pitch')
    if args.instrument_family_conditioning_top is not None or args.instrument_family_conditioning_bottom is not None:
        classes_for_conditioning.add('instrument_family_str')

    additional_modalities = set(modality
                                for modality, _ in args.class_conditioning)
    classes_for_conditioning.update(additional_modalities)

    if args.database_path_for_label_encoders is not None:
        DATABASE_PATH = expand_path(args.database_path_for_label_encoders)
        dataset = LMDBDataset(
            DATABASE_PATH,
            classes_for_conditioning=list(classes_for_conditioning)
        )
        label_encoders_per_conditioning = dataset.label_encoders

    class_conditioning_top = {}
    class_conditioning_bottom = {}

    def maybe_add_conditioning(value, modality: str, location: str) -> None:
        if value is None:
            return
        label_encoder = label_encoders_per_conditioning[modality]
        encoded_label = label_encoder.transform([value])

        if location == 'top':
            class_conditioning_top[modality] = (
                torch.from_numpy(encoded_label).long())
        elif location == 'bottom':
            class_conditioning_bottom[modality] = (
                torch.from_numpy(encoded_label).long())
        else:
            raise ValueError("Invalid location")

    for value, modality, location in zip(
        [args.instrument_family_conditioning_top,
         args.instrument_family_conditioning_bottom,
         args.pitch_conditioning_top, args.pitch_conditioning_bottom,
         ],
        ['instrument_family_str', 'instrument_family_str', 'pitch', 'pitch',],
        ['top', 'bottom', 'top', 'bottom']
    ):
        maybe_add_conditioning(value, modality, location)

    for modality, value in args.class_conditioning:
        for location in ['bottom', 'top']:
            maybe_add_conditioning(value, modality, location)

    with torch.no_grad():
        if args.condition_top_audio_path is not None:
            condition_mel_spec_and_IF = wavfile_to_melspec_and_IF(
                args.condition_top_audio_path)

            (_, _, _, condition_code_top, condition_code_bottom,
             *_) = model_vqvae.encode(condition_mel_spec_and_IF.to(device))

            # repeat condition for the whole batch
            top_code = condition_code_top.repeat(args.batch_size, 1, 1)
        elif args.constraint_top_audio_path is not None:
            constraint_mel_spec_and_IF = wavfile_to_melspec_and_IF(
                args.constraint_top_audio_path)

            (_, _, _, constraint_code_top, *_) = model_vqvae.encode(
                constraint_mel_spec_and_IF.to(device))
            constraint_code_top_restrained = (
                constraint_code_top[:, :args.constraint_top_num_timesteps-1])
            top_code_sample = sample_model(
                model_top, device, batch_size=1, codemap_size=model_top.shape,
                temperature=args.temperature,
                constraint=constraint_code_top_restrained,
                class_conditioning=class_conditioning_top)

            # repeat condition for the whole batch
            top_code = top_code_sample.repeat(args.batch_size, 1, 1)
        else:
            top_code_sample = sample_model(
                model_top, device, args.batch_size, model_top.shape,
                args.temperature,
                class_conditioning=class_conditioning_top)
            top_code = top_code_sample

        # sample bottom code contitioned on the top code
        bottom_sample = sample_model(
            model_bottom, device, args.batch_size, model_bottom.shape,
            args.temperature, condition=top_code,
            class_conditioning=class_conditioning_bottom
        )

        decoded_sample = model_vqvae.decode_code(top_code, bottom_sample)

    codes_figure, _ = plot_codes(top_code, bottom_sample,
                                 model_top.n_class,
                                 model_bottom.n_class)

    inference_vqvae = InferenceVQVAE(model_vqvae, device,
                                     hop_length=args.hop_length,
                                     n_fft=args.n_fft)

    condition_top_audio = None
    if args.condition_top_audio_path is not None:
        CONDITION_TOP_AUDIO_PATH = expand_path(args.condition_top_audio_path)
        import torchvision.transforms as transforms
        sample_audio, fs_hz = torchaudio.load_wav(CONDITION_TOP_AUDIO_PATH,
                                                  channels_first=True)
        toFloat = transforms.Lambda(lambda x: (x / np.iinfo(np.int16).max))
        sample_audio = toFloat(sample_audio)
        condition_top_audio = sample_audio.flatten().cpu().numpy()

    def make_audio(mag_and_IF_batch: torch.Tensor,
                   condition_audio: Optional[np.ndarray]) -> np.ndarray:
        audio_batch = inference_vqvae.mag_and_IF_to_audio(
            mag_and_IF_batch, use_mel_frequency=args.use_mel_frequency)
        normalized_audio_batch = (
            audio_batch
            / audio_batch.abs().max(dim=1, keepdim=True)[0])
        audio_mono_concatenated = normalized_audio_batch.flatten().cpu().numpy()
        if condition_audio is not None:
            audio_mono_concatenated = np.concatenate([condition_audio,
                                                     np.zeros(condition_audio.shape),
                                                     audio_mono_concatenated])
        return audio_mono_concatenated

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    with open(OUTPUT_DIRECTORY / f'{run_ID}-command_line_parameters.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    codes_figure.savefig(OUTPUT_DIRECTORY / 'codemaps.png')

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
