from interactive_spectrogram_inpainting.vqvae.vqvae import VQVAE
from interactive_spectrogram_inpainting.priors.transformer import (
    SelfAttentiveVQTransformer,
    UpsamplingVQTransformer,
    VQNSynthTransformer)
from sample import (sample_model, make_conditioning_tensors,
                    ConditioningMap, make_conditioning_map)
from dataset import LMDBDataset
from interactive_spectrogram_inpainting.utils.misc import (
    expand_path, get_spectrograms_helper)

from GANsynth_pytorch.spectrograms_helper import SpectrogramsHelper

import soundfile
from typing import Union, Tuple, Mapping, Optional, Dict, List
import click
import tempfile
import os
import json
import functools
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import LabelEncoder
from zipfile import ZipFile
from distutils.util import strtobool
import numpy as np

import torch
import torchaudio
from torch.utils.data import DataLoader

import flask
from flask import request
from flask_cors import CORS

import logging
from logging import handlers as logging_handlers


torchaudio.set_audio_backend('sox_io')

# use matplotlib without an X server
# on desktop, this prevents matplotlib windows from popping around
mpl.use('Agg')

app = flask.Flask(__name__, static_folder='uploads')
CORS(app)


# upload_directory = expand_path(pathlib.Path(tempfile.gettempdir())
#                                / 'vqvae_uploads/')
upload_directory = 'uploads/'
app.config['UPLOAD_FOLDER'] = str('./' + upload_directory)
ALLOWED_EXTENSIONS = {'wav'}

# INITIALIZATION
wav_response_headers = {"Content-Type": "audio/wav"
                        }

vqvae: Optional[VQVAE] = None
spectrograms_helper: Optional[SpectrogramsHelper] = None
transformer_top: Optional[VQNSynthTransformer] = None
transformer_bottom: Optional[VQNSynthTransformer] = None
label_encoders_per_modality: Optional[Mapping[str, LabelEncoder]] = None
codes_dataloader: Optional[DataLoader] = None
FS_HZ: Optional[int] = None
HOP_LENGTH: Optional[int] = None
DEVICE: Optional[str] = None
MAX_SOUND_DURATION_S: Optional[float] = None
SPECTROGRAMS_UPSAMPLING_FACTOR: Optional[int] = None
USE_LOCAL_CONDITIONING: Optional[bool] = None
TOP_K: Optional[int] = None
TOP_P: Optional[float] = None
USE_PREDICTIVE_SAMPLING: bool = False

partial_sample_model = None

_num_iterations = None
_sequence_length_ticks = None
_ticks_per_quarter = None


def full_frame(width=None, height=None):
    """Initialize a full-frame matplotlib figure and axes

    Taken from a GitHub Gist by Kile McDonald:
    https://gist.github.com/kylemcdonald/bedcc053db0e7843ef95c531957cb90f
    """
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes((0, 0, 1, 1), frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(tight=True)
    return fig, ax


def make_spectrogram_image(spectrogram: torch.Tensor,
                           filename: str = 'spectrogram',
                           upsampling_factor: int = 1,
                           format: str = 'png'
                           ) -> pathlib.Path:
    """Generate and save a png image for the provided spectrogram.

    Assumes melscale frequency axis.

    Arguments:
        spectrogram (torch.Tensor): the mel-scale spectrogram to draw
    Returns:
        output_path (str): the path where the image was written
    """
    global FS_HZ
    assert FS_HZ is not None
    global HOP_LENGTH
    assert HOP_LENGTH is not None

    if upsampling_factor > 1:
        spectrogram = (
            torch.nn.functional.interpolate(
                spectrogram.unsqueeze(0).unsqueeze(1),
                mode='bilinear',
                scale_factor=upsampling_factor)).squeeze(0).squeeze(0)
    spectrogram_np = spectrogram.cpu().numpy()

    fig, ax = full_frame(width=12, height=8)
    librosa.display.specshow(spectrogram_np,
                                #  y_axis='mel',
                                ax=ax,
                                sr=FS_HZ * upsampling_factor,
                                cmap='viridis',
                                hop_length=HOP_LENGTH)
    output_path = upload_directory + filename + '.' + format
    fig.savefig(output_path, format=format, dpi=200,
                pad_inches=0, bbox_inches=0)
    fig.clear()
    plt.close()

    return pathlib.Path(output_path)


@torch.no_grad()
@click.command()
@click.option('--vqvae_model_parameters_path', type=pathlib.Path,
              required=True)
@click.option('--vqvae_training_parameters_path', type=pathlib.Path,
              required=True)
@click.option('--vqvae_weights_path', type=pathlib.Path,
              required=True)
@click.option('--prediction_top_parameters_path', type=pathlib.Path,
              required=True)
@click.option('--prediction_top_weights_path', type=pathlib.Path,
              required=True)
@click.option('--prediction_bottom_parameters_path', type=pathlib.Path,
              required=True)
@click.option('--prediction_bottom_weights_path', type=pathlib.Path,
              required=True)
@click.option('--database_path_for_label_encoders', type=pathlib.Path,
              required=True)
@click.option('--database_path_for_sampling', type=pathlib.Path,
              required=True)
@click.option('--fs_hz', default=16000)
@click.option('--max_sound_duration_s', default=60,
              help='Maximum allowed duration for imported samples (in seconds)')
@click.option('--num_iterations', default=50,
              help='number of parallel pseudo-Gibbs sampling iterations (for a single update)')
@click.option('--n_fft', default=2048)
@click.option('--hop_length', default=512)
@click.option('--spectrograms_upsampling_factor', default=4)
@click.option('--use_local_conditioning/--ignore_local_conditioning',
              default=True)
@click.option('--sampling_top_k', default=0)
@click.option('--sampling_top_p', default=0.)
@click.option('--use_predictive_sampling/--no_predictive_sampling',
              default=False)
@click.option('--device', type=click.Choice(['cuda', 'cpu'],
                                            case_sensitive=False),
              default='cuda')
@click.option('--port', default=5000,
              help='port to serve on')
def init_app(vqvae_model_parameters_path: pathlib.Path,
             vqvae_training_parameters_path: pathlib.Path,
             vqvae_weights_path: pathlib.Path,
             prediction_top_parameters_path: pathlib.Path,
             prediction_top_weights_path: pathlib.Path,
             prediction_bottom_parameters_path: pathlib.Path,
             prediction_bottom_weights_path: pathlib.Path,
             database_path_for_label_encoders: pathlib.Path,
             database_path_for_sampling: pathlib.Path,
             fs_hz: int,
             max_sound_duration_s: float,
             num_iterations: int,
             n_fft: int,
             hop_length: int,
             spectrograms_upsampling_factor: int,
             use_local_conditioning: bool,
             sampling_top_k: int,
             sampling_top_p: float,
             use_predictive_sampling: bool,
             device: str,
             port: int,
             ):
    global FS_HZ
    global HOP_LENGTH
    global MAX_SOUND_DURATION_S
    global DEVICE
    global SPECTROGRAMS_UPSAMPLING_FACTOR
    global USE_LOCAL_CONDITIONING
    global TOP_K
    global TOP_P
    global USE_PREDICTIVE_SAMPLING
    global partial_sample_model
    FS_HZ = fs_hz
    HOP_LENGTH = hop_length
    MAX_SOUND_DURATION_S = max_sound_duration_s
    DEVICE = device
    SPECTROGRAMS_UPSAMPLING_FACTOR = spectrograms_upsampling_factor
    USE_LOCAL_CONDITIONING = use_local_conditioning
    TOP_K = sampling_top_k
    TOP_P = sampling_top_p
    USE_PREDICTIVE_SAMPLING = use_predictive_sampling

    global vqvae
    print("Load VQ-VAE")
    vqvae = VQVAE.from_parameters_and_weights(
        expand_path(vqvae_model_parameters_path),
        expand_path(vqvae_weights_path),
        device=DEVICE
    )
    vqvae.eval().to(DEVICE)

    global spectrograms_helper
    VQVAE_TRAINING_PARAMETERS_PATH = expand_path(
        vqvae_training_parameters_path)
    # retrieve n_fft, hop length, window length parameters...
    with open(VQVAE_TRAINING_PARAMETERS_PATH, 'r') as f:
        vqvae_training_parameters = json.load(f)
    spectrograms_helper = get_spectrograms_helper(
        device=DEVICE, **vqvae_training_parameters)
    spectrograms_helper.to(DEVICE)

    global transformer_top
    print("Load top-layer Transformer")
    transformer_top = SelfAttentiveVQTransformer.from_parameters_and_weights(
        expand_path(prediction_top_parameters_path),
        expand_path(prediction_top_weights_path),
        device=DEVICE
    )
    transformer_top.eval().to(DEVICE)
    print("Load bottom-layer Transformer")
    global transformer_bottom
    transformer_bottom = UpsamplingVQTransformer.from_parameters_and_weights(
        expand_path(prediction_bottom_parameters_path),
        expand_path(prediction_bottom_weights_path),
        device=DEVICE
    )
    transformer_bottom.eval().to(DEVICE)

    global label_encoders_per_modality
    print("Retrieve label encoders")
    classes_for_conditioning = ['pitch', 'instrument_family_str']
    DATABASE_PATH = expand_path(database_path_for_label_encoders)
    dataset = LMDBDataset(
        DATABASE_PATH,
        classes_for_conditioning=list(classes_for_conditioning)
    )
    label_encoders_per_modality = dataset.label_encoders

    global codes_dataloader
    print("Load dataset for initial sounds sampling")
    classes_for_conditioning = ['pitch', 'instrument_family_str']
    SAMPLING_DATABASE_PATH = expand_path(database_path_for_label_encoders)
    codes_dataset = LMDBDataset(
        SAMPLING_DATABASE_PATH,
        classes_for_conditioning=list(classes_for_conditioning)
    )
    codes_dataloader = DataLoader(codes_dataset, shuffle=True,
                                  batch_size=1)

    partial_sample_model = functools.partial(
        sample_model,
        device=DEVICE,
        top_k_sampling_k=TOP_K,
        top_p_sampling_p=TOP_P,
        use_predictive_sampling=USE_PREDICTIVE_SAMPLING
    )

    os.makedirs('./uploads', exist_ok=True)
    # launch the script
    # use threaded=True to fix Chrome/Chromium engine hanging on requests
    # [https://stackoverflow.com/a/30670626]
    local_only = False
    if local_only:
        # accessible only locally:
        app.run(threaded=True)
    else:
        # accessible from outside:
        app.run(host='0.0.0.0', port=port, threaded=True)


def make_matrix(shape: Tuple[int, int],
                value: Union[str, int]
                ) -> ConditioningMap:
    return [[value] * shape[1]] * shape[0]


def masked_fill(array, mask, value):
    return [[value if mask_value else previous_value
             for previous_value, mask_value in zip(array_row, mask_row)]
            for array_row, mask_row in zip(array, mask)]


def resize_codemaps_repeat_last(
        top_code: torch.Tensor, bottom_code: torch.Tensor,
        duration_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
    upsampling_ratio_time = bottom_code.shape[-1] // top_code.shape[-1]
    duration_bottom = upsampling_ratio_time * duration_top

    def resize_codemap(codemap: torch.Tensor, duration: int) -> torch.Tensor:
        codemap = codemap[..., :duration]
        if codemap.shape[-1] < duration:
            codemap = torch.cat([codemap] + [codemap[..., -1:]] * (
                duration - codemap.shape[-1]),
                                dim=-1)
        return codemap

    return tuple(resize_codemap(codemap, duration)
                 for (codemap, duration) in zip((top_code, bottom_code),
                                                (duration_top, duration_bottom)))


def get_codemaps_from_database(
        duration_top: int,
        attribute_constraints: Dict[str, Union[any, List[any]]] = {}):
    global codes_dataloader
    assert codes_dataloader is not None
    global label_encoders_per_modality
    if 'pitch_class' or 'octave' in attribute_constraints.keys():
        assert (label_encoders_per_modality is not None
                and 'pitch' in label_encoders_per_modality)

    def decode_attributes(encoded_attributes):
        decoded_attributes = {
            key: label_encoders_per_modality[key].inverse_transform(
                [value.item()])[0]
            for key, value in encoded_attributes.items()}
        if 'pitch_class' in attribute_constraints.keys():
            decoded_attributes['pitch_class'] = (
                decoded_attributes['pitch'] % 12)
        if 'octave' in attribute_constraints.keys():
            decoded_attributes['pitch_class'] = (
                decoded_attributes['pitch'] // 12)
        return decoded_attributes

    def check_attributes(
            attributes: Dict[str, Union[any, List[any]]]):
        return all([
            attributes[key] == constraint
            for key, constraint in attribute_constraints.items()
        ])

    found_valid_sample = False
    while not found_valid_sample:
        top_code, bottom_code, encoded_sample_attributes = next(iter(
            codes_dataloader))
        sample_attributes = decode_attributes(encoded_sample_attributes)
        found_valid_sample = check_attributes(sample_attributes)
    return (resize_codemaps_repeat_last(top_code, bottom_code, duration_top),
            sample_attributes)


@torch.no_grad()
@app.route('/generate', methods=['GET', 'POST'])
def generate():
    """
    Return a new, generated sheet
    Usage:
        [GET/POST] /generate?pitch=XXX&instrument_family_str=XXX&temperature=XXX
        - Request: empty payload, a new sound is synthesized from scratch
        - Response: a new, generated sound
    """
    global transformer_top
    assert transformer_top is not None
    global transformer_bottom
    assert transformer_bottom is not None
    global label_encoders_per_modality
    assert label_encoders_per_modality is not None
    global DEVICE
    assert DEVICE is not None
    global partial_sample_model
    assert partial_sample_model is not None

    temperature = float(request.args.get('temperature'))
    pitch = int(request.args.get('pitch'))
    instrument_family_str = str(request.args.get('instrument_family_str'))

    class_conditioning_top = class_conditioning_bottom = {
        'pitch': pitch,
        'instrument_family_str': instrument_family_str
    }
    class_conditioning_tensors_top = make_conditioning_tensors(
        class_conditioning_top,
        label_encoders_per_modality)
    class_conditioning_tensors_bottom = make_conditioning_tensors(
        class_conditioning_bottom,
        label_encoders_per_modality)

    batch_size = 1
    top_code = partial_sample_model(
        model=transformer_top,
        batch_size=batch_size,
        codemap_size=transformer_top.shape,
        temperature=temperature,
        class_conditioning=class_conditioning_tensors_top
    )
    bottom_code = partial_sample_model(
        model=transformer_bottom,
        condition=top_code,
        batch_size=batch_size,
        codemap_size=transformer_bottom.shape,
        temperature=temperature,
        class_conditioning=class_conditioning_tensors_bottom,
    )

    class_conditioning_top_map = {
        modality: make_matrix(transformer_top.shape,
                              value)
        for modality, value in class_conditioning_top.items()
    }
    class_conditioning_bottom_map = {
        modality: make_matrix(transformer_bottom.shape,
                              value)
        for modality, value in class_conditioning_bottom.items()
    }

    response = make_response(top_code, bottom_code,
                             class_conditioning_top_map,
                             class_conditioning_bottom_map)
    return response


@torch.no_grad()
@app.route('/sample-from-dataset', methods=['GET', 'POST'])
def sample_from_dataset():
    global label_encoders_per_modality
    assert label_encoders_per_modality is not None

    duration_top = request.args.get('duration_top', type=int)

    # retrieve and check sampling constraints
    constraint_pitch = (
        request.args.get('pitch', type=int, default=None))
    assert (
        constraint_pitch is None
        or constraint_pitch in label_encoders_per_modality['pitch'].classes_
        )

    constraint_pitch_class = request.args.get('pitch_class', type=int,
                                              default=None)
    if (constraint_pitch_class is not None
        and (constraint_pitch_class < 0
             or constraint_pitch_class > 12)):
        constraint_pitch_class = None

    constraint_octave = request.args.get('octave', type=int, default=None)
    if (constraint_octave is not None and constraint_octave < 0):
        constraint_octave = None

    constraint_instrument_family_str = request.args.get(
        'instrument_family_str', type=str, default=None)
    assert (
        constraint_instrument_family_str is None
        or constraint_instrument_family_str in (
            label_encoders_per_modality['instrument_family_str'].classes_)
        )

    attribute_constraints = {}
    # TODO(theis, 2021_04_20): simplify this
    if constraint_pitch is not None:
        attribute_constraints['pitch'] = constraint_pitch
    if constraint_pitch_class is not None:
        attribute_constraints['pitch_class'] = constraint_pitch_class
    if constraint_octave is not None:
        attribute_constraints['octave'] = constraint_octave
    if constraint_instrument_family_str is not None:
        attribute_constraints['instrument_family_str'] = (
            constraint_instrument_family_str)
    (top_code, bottom_code), sampled_attributes = get_codemaps_from_database(
        duration_top, attribute_constraints)

    class_conditioning_top = class_conditioning_bottom = {
        'pitch': int(sampled_attributes['pitch']),
        'instrument_family_str': str(
            sampled_attributes['instrument_family_str'])
    }

    class_conditioning_top_map = {
        modality: make_matrix(top_code.shape,
                              value)
        for modality, value in class_conditioning_top.items()
    }
    class_conditioning_bottom_map = {
        modality: make_matrix(bottom_code.shape,
                              value)
        for modality, value in class_conditioning_bottom.items()
    }

    response = make_response(top_code, bottom_code,
                             class_conditioning_top_map,
                             class_conditioning_bottom_map)
    return response


@app.route('/test-generate', methods=['GET', 'POST'])
@torch.no_grad()
def test_generate():
    global transformer_top
    assert transformer_top is not None
    global transformer_bottom
    assert transformer_bottom is not None

    pitch = int(request.args.get('pitch'))
    instrument_family_str = str(request.args.get('instrument_family_str'))

    class_conditioning_top = class_conditioning_bottom = {
        'pitch': pitch,
        'instrument_family_str': instrument_family_str
    }

    top_code = torch.randint(size=transformer_top.shape, low=0,
                             high=vqvae.n_embed_t).unsqueeze(0)
    bottom_code = torch.randint(size=transformer_bottom.shape, low=0,
                                high=vqvae.n_embed_b).unsqueeze(0)

    class_conditioning_top_map = {
        modality: make_matrix(transformer_top.shape,
                              value)
        for modality, value in class_conditioning_top.items()
    }
    class_conditioning_bottom_map = {
        modality: make_matrix(transformer_bottom.shape,
                              value)
        for modality, value in class_conditioning_bottom.items()
    }

    response = make_response(top_code, bottom_code,
                             class_conditioning_top_map,
                             class_conditioning_bottom_map)
    return response


def get_duration_sox_n(audio_file_path: str) -> float:
    """Retrieve duration of a signal without loading it

    This uses the global sampling frequency of the loaded models
    """
    global FS_HZ
    assert FS_HZ is not None
    audiometadata = torchaudio.info(audio_file_path)
    num_frames = audiometadata.num_frames
    num_channels = audiometadata.num_channels
    original_fs_hz = audiometadata.sample_rate
    duration_n = num_frames // num_channels
    # TODO(theis): probably not exact value
    duration_n_resampled = int(duration_n * (FS_HZ / original_fs_hz))
    return duration_n_resampled


def get_duration_sox_s(audio_file_path: str) -> float:
    """Retrieve duration of a signal without loading it

    This uses the global sampling frequency of the loaded models
    """
    global FS_HZ
    assert FS_HZ is not None
    duration_n = get_duration_sox_n(audio_file_path)
    return duration_n / FS_HZ


def get_vqvae_top_resolution_n() -> int:
    """Return the duration in samples of one column of the VQVAE's top layer"""
    global vqvae
    assert vqvae is not None
    global transformer_top
    assert transformer_top is not None
    global spectrograms_helper
    assert spectrograms_helper is not None
    global DEVICE
    assert DEVICE is not None
    dummy_codes_top = torch.zeros(transformer_top.shape,
                                  dtype=torch.long).to(DEVICE).unsqueeze(0)
    dummy_codes_bottom = torch.zeros(transformer_bottom.shape,
                                     dtype=torch.long).to(DEVICE).unsqueeze(0)
    decoded_audio = spectrograms_helper.to_audio(
        vqvae.decode_code(dummy_codes_top, dummy_codes_bottom))
    _, duration_top = transformer_top.shape
    return decoded_audio.shape[-1] // duration_top


def adapt_duration(audio_file_path: str) -> float:
    """Adapt duration of a file for loading

    Accounts for both the max duration and the VQ-VAE's resolution
    """
    global MAX_SOUND_DURATION_S
    assert MAX_SOUND_DURATION_S is not None
    global FS_HZ
    assert FS_HZ is not None
    global transformer_top
    assert transformer_top is not None
    duration_n = get_duration_sox_n(audio_file_path)
    # trim to max duration
    duration_n = min(MAX_SOUND_DURATION_S * FS_HZ, duration_n)
    # round-up to the resolution of the VQVAE
    vqvae_top_resolution_n = get_vqvae_top_resolution_n()
    duration_n = vqvae_top_resolution_n * (max(
        transformer_top.shape[1],
        np.round(duration_n / vqvae_top_resolution_n)))
    return duration_n


@app.route('/analyze-audio', methods=['POST'])
@torch.no_grad()
def audio_to_codes():
    global vqvae
    assert vqvae is not None
    global spectrograms_helper
    assert spectrograms_helper is not None
    global DEVICE
    assert DEVICE is not None
    global FS_HZ
    assert FS_HZ is not None

    pitch = int(request.args.get('pitch'))
    instrument_family_str = str(request.args.get('instrument_family_str'))

    class_conditioning_top = class_conditioning_bottom = {
        'pitch': pitch,
        'instrument_family_str': instrument_family_str
    }

    with tempfile.NamedTemporaryFile(
            'w+b', suffix=request.files['audio'].filename) as f:
        request.files['audio'].save(f.name)
        duration_n = adapt_duration(f.name)
        spec_and_IF = spectrograms_helper.from_wavfile(
            f.name, duration_n=duration_n).to(DEVICE)

    _, _, _, top_code, bottom_code, *_ = vqvae.encode(spec_and_IF)

    class_conditioning_top_map = {
        modality: make_matrix(transformer_top.shape,
                              value)
        for modality, value in class_conditioning_top.items()
    }
    class_conditioning_bottom_map = {
        modality: make_matrix(transformer_bottom.shape,
                              value)
        for modality, value in class_conditioning_bottom.items()
    }

    response = make_response(top_code, bottom_code,
                             class_conditioning_top_map,
                             class_conditioning_bottom_map)
    return response


def make_time_indexes(start_index: int, codemap_duration: int,
                      transformer_duration: int) -> List[int]:
    time_indexes_full = [0]  # attack
    num_steps_to_repeat = transformer_duration - 2
    steps_repetitions = (codemap_duration - 2) // num_steps_to_repeat
    for i in range(num_steps_to_repeat - 1):
        time_indexes_full += [i+1] * steps_repetitions
    time_indexes_full += [num_steps_to_repeat] * (
        (codemap_duration - 2) - (len(time_indexes_full)-1))
    time_indexes_full += [transformer_duration-1]

    return time_indexes_full[start_index:
                             start_index+transformer_duration]


@app.route('/timerange-change', methods=['POST'])
@torch.no_grad()
def timerange_change():
    """
    Perform local re-generation on a sheet and return the updated sheet
    Usage:
        POST /timerange-change?TODO(theis)
        - Request:
        - Response:
    """
    global transformer_top
    assert transformer_top is not None
    global transformer_bottom
    assert transformer_bottom is not None
    global label_encoders_per_modality
    assert label_encoders_per_modality is not None
    global DEVICE
    assert DEVICE is not None
    global USE_LOCAL_CONDITIONING
    assert USE_LOCAL_CONDITIONING is not None
    global partial_sample_model
    assert partial_sample_model is not None

    layer = str(request.args.get('layer'))
    temperature = request.args.get('temperature', type=float)
    start_index_top = request.args.get('start_index_top', type=int)
    uniform_sampling = bool(strtobool(
        request.args.get('uniform_sampling', type=str,
                         default="False")))

    # try to retrieve local conditioning map in the request's JSON payload
    (class_conditioning_top_map, class_conditioning_bottom_map,
     input_conditioning_top, input_conditioning_bottom) = (
        parse_conditioning(request)
    )
    global_instrument_family_str = str(
        request.args.get('instrument_family_str'))
    global_pitch = request.args.get('pitch', type=int)
    global_class_conditioning = {
        'pitch': global_pitch,
        'instrument_family_str': global_instrument_family_str
    }
    if (not USE_LOCAL_CONDITIONING
            or not transformer_bottom.local_class_conditioning):
        class_conditioning_bottom = global_class_conditioning.copy()
        class_conditioning_tensors_bottom = make_conditioning_tensors(
            class_conditioning_bottom,
            label_encoders_per_modality)
        class_conditioning_bottom_map = None
    else:
        class_conditioning_bottom = class_conditioning_tensors_bottom = None

    top_code, bottom_code = parse_codes(request)

    # extract frame to operate on
    end_index_top = start_index_top + transformer_top.shape[1]
    top_code_frame = top_code[..., start_index_top:end_index_top]

    upsampling_ratio_time = (transformer_bottom.shape[1]
                             // transformer_top.shape[1])
    start_index_bottom = upsampling_ratio_time * start_index_top
    end_index_bottom = start_index_bottom + transformer_bottom.shape[1]
    bottom_code_frame = bottom_code[..., start_index_bottom:end_index_bottom]
    generation_mask_batched = parse_mask(request).to(DEVICE)

    time_indexes_top = make_time_indexes(start_index_top,
                                         top_code.shape[-1],
                                         transformer_top.shape[-1])
    time_indexes_bottom = make_time_indexes(start_index_bottom,
                                            bottom_code.shape[-1],
                                            transformer_bottom.shape[-1])

    if layer == 'bottom':
        if not uniform_sampling:
            bottom_code_resampled_frame = partial_sample_model(
                model=transformer_bottom,
                condition=top_code_frame,
                batch_size=1,
                codemap_size=transformer_bottom.shape,
                temperature=temperature,
                class_conditioning=class_conditioning_tensors_bottom,
                local_class_conditioning_map=class_conditioning_bottom_map,
                initial_code=bottom_code_frame,
                mask=generation_mask_batched,
                time_indexes_source=time_indexes_top,
                time_indexes_target=time_indexes_bottom,
            )
        else:
            bottom_code_resampled_frame = bottom_code_frame.masked_scatter(
                generation_mask_batched,
                torch.randint_like(bottom_code_frame,
                                   high=transformer_bottom.n_class_target)
            )

        bottom_code_resampled = bottom_code
        bottom_code_resampled[..., start_index_bottom:end_index_bottom] = (
            bottom_code_resampled_frame)

        # create JSON response
        response = make_response(top_code, bottom_code_resampled,
                                 input_conditioning_top,
                                 input_conditioning_bottom)
    elif layer == 'top':
        if (not USE_LOCAL_CONDITIONING
                or not transformer_top.local_class_conditioning):
            # try to retrieve conditioning from http arguments
            class_conditioning_top = global_class_conditioning.copy()
            class_conditioning_tensors_top = make_conditioning_tensors(
                class_conditioning_top,
                label_encoders_per_modality)
            class_conditioning_top_map = None
        else:
            class_conditioning_top = class_conditioning_tensors_top = None

        if not uniform_sampling:
            if transformer_top.self_conditional_model:
                condition = top_code_frame
            else:
                condition = None
            top_code_resampled_frame = partial_sample_model(
                model=transformer_top,
                condition=condition,
                device=DEVICE,
                batch_size=1,
                codemap_size=transformer_top.shape,
                temperature=temperature,
                class_conditioning=class_conditioning_tensors_top,
                local_class_conditioning_map=class_conditioning_top_map,
                initial_code=top_code_frame,
                mask=generation_mask_batched,
                time_indexes_source=time_indexes_top,
                time_indexes_target=time_indexes_top,
            )
        else:
            top_code_resampled_frame = top_code_frame.masked_scatter(
                generation_mask_batched,
                torch.randint_like(top_code_frame,
                                   high=transformer_top.n_class_target)
            )

        top_code_resampled = top_code
        top_code_resampled[..., start_index_top:end_index_top] = (
            top_code_resampled_frame)

        upsampling_ratio_frequency = (transformer_bottom.shape[0]
                                      // transformer_top.shape[0])
        generation_mask_bottom_batched = (
            generation_mask_batched
            .repeat_interleave(upsampling_ratio_frequency, -2)
            .repeat_interleave(upsampling_ratio_time, -1)
        )
        bottom_code_resampled_frame = partial_sample_model(
            model=transformer_bottom,
            condition=top_code_resampled_frame,
            device=DEVICE,
            batch_size=1,
            codemap_size=transformer_bottom.shape,
            temperature=temperature,
            class_conditioning=class_conditioning_tensors_bottom,
            local_class_conditioning_map=class_conditioning_bottom_map,
            initial_code=bottom_code_frame,
            mask=generation_mask_bottom_batched,
            time_indexes_source=time_indexes_top,
            time_indexes_target=time_indexes_bottom,
        )

        # update conditioning map
        bottom_mask = generation_mask_bottom_batched[0]
        new_conditioning_map_bottom = {
            modality: masked_fill(modality_conditioning,
                                  bottom_mask,
                                  class_conditioning_bottom[modality])
            for modality, modality_conditioning
            in input_conditioning_bottom.items()
        }

        bottom_code_resampled = bottom_code
        bottom_code_resampled[..., start_index_bottom:end_index_bottom] = (
            bottom_code_resampled_frame)

        # create JSON response
        response = make_response(top_code_resampled, bottom_code_resampled,
                                 input_conditioning_top,
                                 new_conditioning_map_bottom)

    return response


@app.route('/erase', methods=['POST'])
@torch.no_grad()
def erase():
    global transformer_top
    assert transformer_top is not None
    global transformer_bottom
    assert transformer_bottom is not None
    global label_encoders_per_modality
    assert label_encoders_per_modality is not None
    global DEVICE
    assert DEVICE is not None

    amplitude = float(request.args.get('eraser_amplitude'))
    start_index_top = int(request.args.get('start_index_top'))

    top_code_batched, bottom_code_batched = parse_codes(request)
    generation_mask = parse_mask(request).to(DEVICE)[0]

    logmelspectrogram, IF = vqvae.decode_code(top_code_batched,
                                              bottom_code_batched)[0]

    top_code = top_code_batched[0]
    upsampling_f = logmelspectrogram.shape[0] // top_code.shape[0]
    upsampling_t = logmelspectrogram.shape[1] // top_code.shape[1]

    upsampled_mask = (generation_mask.float().flip(0)
                      .repeat_interleave(upsampling_f, 0)
                      .repeat_interleave(upsampling_t, 1)
                      ).flip(0)
    amplitude_mask = 200 * amplitude * upsampled_mask

    # zero-pad the amplitude mask
    padding_before = torch.zeros(logmelspectrogram.shape[0],
                                 upsampling_t * start_index_top)
    padding_after = torch.zeros(logmelspectrogram.shape[0],
                                max(0,
                                    upsampling_t * (
                                        top_code.shape[1] - (
                                            start_index_top + generation_mask.shape[1]))))
    amplitude_mask = torch.cat([
        padding_before.to(DEVICE),
        amplitude_mask,
        padding_after.to(DEVICE)], dim=1)

    masked_logmelspectrogram_and_IF = torch.cat(
        [(logmelspectrogram - amplitude_mask).unsqueeze(0),
         IF.unsqueeze(0)],
        dim=0
    ).unsqueeze(0)

    _, _, _, new_top_code, new_bottom_code, *_ = vqvae.encode(
        masked_logmelspectrogram_and_IF)

    (_, _, input_conditioning_top, input_conditioning_bottom) = (
        parse_conditioning(request))
    return make_response(new_top_code, new_bottom_code,
                         input_conditioning_top,
                         input_conditioning_bottom)


@torch.no_grad()
def parse_codes(request) -> Tuple[torch.LongTensor,
                                  torch.LongTensor]:
    global transformer_top
    assert transformer_top is not None
    global transformer_bottom
    assert transformer_bottom is not None

    json_data = request.get_json(force=True)

    top_code_array = json_data['top_code']
    bottom_code_array = json_data['bottom_code']

    top_code = torch.LongTensor(top_code_array
                                ).unsqueeze(0).to(DEVICE)
    bottom_code = torch.LongTensor(bottom_code_array
                                   ).unsqueeze(0).to(DEVICE)

    return top_code, bottom_code


def parse_conditioning(request) -> Tuple[torch.LongTensor,
                                         torch.LongTensor,
                                         Mapping[str, ConditioningMap],
                                         Mapping[str, ConditioningMap],
                                         ]:
    global label_encoders_per_modality
    assert label_encoders_per_modality is not None

    json_data = request.get_json(force=True)

    if 'top_conditioning' not in json_data.keys():
        return None, None

    conditioning_top = json_data['top_conditioning']
    conditioning_bottom = json_data['bottom_conditioning']

    class_conditioning_top_map = make_conditioning_map(
        conditioning_top,
        label_encoders_per_modality)
    class_conditioning_bottom_map = make_conditioning_map(
        conditioning_bottom,
        label_encoders_per_modality)

    return (class_conditioning_top_map, class_conditioning_bottom_map,
            conditioning_top, conditioning_bottom)


def parse_mask(request) -> torch.BoolTensor:
    json_data = request.get_json(force=True)

    generation_mask_array = json_data['mask']
    generation_mask = torch.BoolTensor(generation_mask_array
                                       ).unsqueeze(0)

    return generation_mask


def make_response(top_code: torch.Tensor,
                  bottom_code: torch.Tensor,
                  class_conditioning_top_map: Mapping[str, ConditioningMap],
                  class_conditioning_bottom_map: Mapping[str, ConditioningMap],
                  send_files: bool = False):
    return flask.jsonify({'top_code': top_code[0].int().cpu().numpy().tolist(),
                          'bottom_code': bottom_code[0].int().cpu().numpy().tolist(),
                          'top_conditioning': class_conditioning_top_map,
                          'bottom_conditioning': class_conditioning_bottom_map,
                          })


@app.route('/get-audio', methods=['POST'])
@torch.no_grad()
def codes_to_audio_response():
    global vqvae
    assert vqvae is not None
    global spectrograms_helper
    assert spectrograms_helper is not None
    top_code, bottom_code = parse_codes(request)

    logmelspectrogram_and_IF = vqvae.decode_code(top_code,
                                                 bottom_code)

    # convert to audio and write to file
    audio = spectrograms_helper.to_audio(logmelspectrogram_and_IF)[0]
    audio_path = write_audio_to_file(audio)

    return flask.send_file(audio_path, mimetype="audio/wav",
                           cache_timeout=-1  # disable cache
                           )


@app.route('/get-spectrogram-image', methods=['POST'])
@torch.no_grad()
def codes_to_spectrogram_image_response():
    global SPECTROGRAMS_UPSAMPLING_FACTOR
    assert SPECTROGRAMS_UPSAMPLING_FACTOR is not None

    top_code, bottom_code = parse_codes(request)

    logmelspectrogram_and_IF = vqvae.decode_code(top_code,
                                                 bottom_code)

    # generate spectrogram PNG image
    spectrogram = logmelspectrogram_and_IF[0, 0]
    image_format = 'png'
    spectrogram_image_path = make_spectrogram_image(
        spectrogram,
        upsampling_factor=SPECTROGRAMS_UPSAMPLING_FACTOR,
        format=image_format)

    return flask.send_file(spectrogram_image_path,
                           mimetype=f"image/{image_format}",
                           cache_timeout=-1  # disable cache
                           )


@app.route('/top-conditioned-sample', methods=['POST'])
@torch.no_grad()
def top_conditioned_sample():
    """Sample from the bottom prior given the incoming top codemap"""
    global vqvae
    assert vqvae is not None
    global DEVICE
    assert DEVICE is not None
    global transformer_bottom
    assert transformer_bottom is not None
    global label_encoders_per_modality
    assert label_encoders_per_modality is not None
    global spectrograms_helper
    assert spectrograms_helper is not None

    BYPASS = False

    top_code, bottom_code = parse_codes(request)
    global_instrument_family_str = str(
        request.args.get('instrument_family_str'))
    min_pitch = int(request.args.get('min_pitch'))
    max_pitch = int(request.args.get('max_pitch'))

    if not BYPASS:
        temperature = float(request.args.get('temperature'))
        top_p = float(request.args.get('top_p') or 0.0)
        top_k = int(request.args.get('top_k') or 0)

        class_conditioning_tensors_bottom = make_conditioning_tensors(
            {'pitch': (min_pitch, max_pitch),
             'instrument_family_str': global_instrument_family_str},
            label_encoders_per_modality
        )

        # repeat the top codemap for all bottom samples
        num_samples = max_pitch - min_pitch
        top_code = top_code.expand(num_samples, -1, -1)

        bottom_code = sample_model(
            transformer_bottom, DEVICE, num_samples,
            transformer_bottom.shape,
            temperature, condition=top_code,
            class_conditioning=class_conditioning_tensors_bottom,
            top_p_sampling_p=top_p,
            top_k_sampling_k=top_k
        )
    else:
        import time
        num_samples = 1
        top_code = top_code.expand(num_samples, -1, -1)
        bottom_code = bottom_code.expand(num_samples, -1, -1)
        time.sleep(2)

    logmelspectrogram_and_IF = vqvae.decode_code(top_code,
                                                 bottom_code)

    zip_path = upload_directory + 'samples.zip'
    with ZipFile(zip_path, 'w') as zf:
        for pitch, sample in zip(range(min_pitch, max_pitch),
                                 logmelspectrogram_and_IF):
            # convert to audio and write to file
            audio = spectrograms_helper.to_audio(sample.unsqueeze(0))[0]
            audio_path = write_audio_to_file(
                audio,
                f'-{global_instrument_family_str}-{pitch}')
            zf.write(audio_path,
                     arcname=f'{global_instrument_family_str}-{pitch}.wav')

    return flask.send_file(zip_path, mimetype="application/zip",
                           cache_timeout=-1  # disable cache
                           )


def write_audio_to_file(audio: torch.Tensor, sample_id: str = ''):
    """Generate and send WAV file
    """
    global FS_HZ
    assert FS_HZ is not None
    audio_extension = '.wav'
    audio_path = upload_directory + 'sample' + sample_id + audio_extension
    audio_np = audio.cpu().numpy()
    with open(audio_path, 'wb') as f:
        soundfile.write(f,
                        audio_np,
                        samplerate=FS_HZ)
    return audio_path


if __name__ == '__main__':
    file_handler = logging_handlers.RotatingFileHandler(
        'app.log', maxBytes=10000, backupCount=5)

    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    init_app()
