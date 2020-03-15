from vqvae import VQVAE
from transformer import VQNSynthTransformer
from sample import (sample_model, make_conditioning_tensors,
                    ConditioningMap, make_conditioning_map)
from dataset import LMDBDataset
from GANsynth_pytorch.spectrograms_helper import SpectrogramsHelper
import utils as vqvae_utils
from utils import expand_path

import soundfile
from typing import Union, Tuple, Mapping, Optional
import click
import tempfile
import os
import json
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder

import torch

import flask
from flask import request
from flask_cors import CORS

import logging
from logging import handlers as logging_handlers

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
FS_HZ: Optional[int] = None
HOP_LENGTH: Optional[int] = None
DEVICE: Optional[str] = None
SOUND_DURATION_S: Optional[float] = None
SPECTROGRAMS_UPSAMPLING_FACTOR: Optional[int] = None
USE_LOCAL_CONDITIONING: Optional[bool] = None

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
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    return fig, ax


def make_spectrogram_image(spectrogram: torch.Tensor,
                           filename: str = 'spectrogram',
                           upsampling_factor: int = 1,
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
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.set_axis_off()
    fig, ax = full_frame(width=12, height=8)
    upsampled_spectrogram = (
        torch.nn.functional.interpolate(
            spectrogram.unsqueeze(0).unsqueeze(1),
            mode='bilinear',
            scale_factor=upsampling_factor)).squeeze(0).squeeze(0)
    spectrogram_np = upsampled_spectrogram.cpu().numpy()
    librosa.display.specshow(spectrogram_np,
                             #  y_axis='mel',
                             ax=ax,
                             sr=FS_HZ * upsampling_factor,
                             cmap='viridis',
                             hop_length=HOP_LENGTH)
    # ax.margins(0)
    # fig.tight_layout()

    image_format = 'png'
    # output_path = tempfile.mktemp() + '.' + image_format
    output_path = upload_directory + filename + '.' + image_format
    fig.savefig(output_path, format=image_format, dpi=200,
                pad_inches=0, bbox_inches=0)
    fig.clear()
    plt.close()
    return output_path


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
@click.option('--database_path_for_label_encoders', type=pathlib.Path,
              required=True)
@click.option('--fs_hz', default=16000)
@click.option('--sound_duration_s', default=4)
@click.option('--num_iterations', default=50,
              help='number of parallel pseudo-Gibbs sampling iterations (for a single update)')
@click.option('--n_fft', default=2048)
@click.option('--hop_length', default=512)
@click.option('--spectrograms_upsampling_factor', default=4)
@click.option('--use_local_conditioning/--ignore_local_conditioning',
              default=True)
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
             fs_hz: int,
             sound_duration_s: float,
             num_iterations: int,
             n_fft: int,
             hop_length: int,
             spectrograms_upsampling_factor: int,
             use_local_conditioning: bool,
             device: str,
             port: int,
             ):
    global FS_HZ
    global HOP_LENGTH
    global SOUND_DURATION_S
    global DEVICE
    global SPECTROGRAMS_UPSAMPLING_FACTOR
    global USE_LOCAL_CONDITIONING
    FS_HZ = fs_hz
    HOP_LENGTH = hop_length
    SOUND_DURATION_S = sound_duration_s
    DEVICE = device
    SPECTROGRAMS_UPSAMPLING_FACTOR = spectrograms_upsampling_factor
    USE_LOCAL_CONDITIONING = use_local_conditioning

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
    spectrograms_helper = vqvae_utils.get_spectrograms_helper(
        device=device, **vqvae_training_parameters)
    spectrograms_helper.to(device)

    global transformer_top
    print("Load top-layer Transformer")
    transformer_top = VQNSynthTransformer.from_parameters_and_weights(
        expand_path(prediction_top_parameters_path),
        expand_path(prediction_top_weights_path),
        device=DEVICE
    )
    transformer_top.eval().to(DEVICE)
    print("Load bottom-layer Transformer")
    global transformer_bottom
    transformer_bottom = VQNSynthTransformer.from_parameters_and_weights(
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
        app.run(host='0.0.0.0', port=port, threaded=False)


def make_matrix(shape: Tuple[int, int],
                value: Union[str, int]
                ) -> ConditioningMap:
    return [[value] * shape[1]] * shape[0]


def masked_fill(array, mask, value):
    return [[value if mask_value else previous_value
             for previous_value, mask_value in zip(array_row, mask_row)]
            for array_row, mask_row in zip(array, mask)]


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
    top_code = sample_model(
        model=transformer_top,
        device=DEVICE,
        batch_size=batch_size,
        codemap_size=transformer_top.shape,
        temperature=temperature,
        class_conditioning=class_conditioning_tensors_top
    )
    bottom_code = sample_model(
        model=transformer_bottom,
        condition=top_code,
        device=DEVICE,
        batch_size=batch_size,
        codemap_size=transformer_bottom.shape,
        temperature=temperature,
        class_conditioning=class_conditioning_tensors_bottom
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
@app.route('/test-generate', methods=['GET', 'POST'])
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


@app.route('/analyze-audio', methods=['POST'])
def audio_to_codes():
    global vqvae
    assert vqvae is not None
    global spectrograms_helper
    global DEVICE
    assert DEVICE is not None
    global FS_HZ
    assert FS_HZ is not None
    global SOUND_DURATION_S
    assert SOUND_DURATION_S is not None

    pitch = int(request.args.get('pitch'))
    instrument_family_str = str(request.args.get('instrument_family_str'))

    class_conditioning_top = class_conditioning_bottom = {
        'pitch': pitch,
        'instrument_family_str': instrument_family_str
    }

    with tempfile.NamedTemporaryFile(
            'w+b', suffix=request.files['audio'].filename) as f:
        request.files['audio'].save(f)
        spec_and_IF = spectrograms_helper.from_wavfile(
            f.name, duration_s=SOUND_DURATION_S).to(DEVICE)

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


@app.route('/timerange-change', methods=['POST'])
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

    layer = str(request.args.get('layer'))
    temperature = float(request.args.get('temperature'))

    # try to retrieve local conditioning map in the request's JSON payload
    (class_conditioning_top_map, class_conditioning_bottom_map,
     input_conditioning_top, input_conditioning_bottom) = (
        parse_conditioning(request)
    )
    global_instrument_family_str = str(
        request.args.get('instrument_family_str'))
    global_pitch = int(request.args.get('pitch'))
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
    generation_mask_batched = parse_mask(request).to(DEVICE)

    if layer == 'bottom':
        bottom_code_resampled = sample_model(
            model=transformer_bottom,
            condition=top_code,
            device=DEVICE,
            batch_size=1,
            codemap_size=transformer_bottom.shape,
            temperature=temperature,
            class_conditioning=class_conditioning_tensors_bottom,
            local_class_conditioning_map=class_conditioning_bottom_map,
            initial_code=bottom_code,
            mask=generation_mask_batched
        )

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

        if transformer_top.self_conditional_model:
            condition = top_code
        else:
            condition = None
        top_code_resampled = sample_model(
            model=transformer_top,
            condition=condition,
            device=DEVICE,
            batch_size=1,
            codemap_size=transformer_top.shape,
            temperature=temperature,
            class_conditioning=class_conditioning_tensors_top,
            local_class_conditioning_map=class_conditioning_top_map,
            initial_code=top_code,
            mask=generation_mask_batched
        )

        generation_mask_bottom_batched = (
            generation_mask_batched.repeat_interleave(4, 1)
            .repeat_interleave(4, 2)
        )
        bottom_code_resampled = sample_model(
            model=transformer_bottom,
            condition=top_code_resampled,
            device=DEVICE,
            batch_size=1,
            codemap_size=transformer_bottom.shape,
            temperature=temperature,
            class_conditioning=class_conditioning_tensors_bottom,
            local_class_conditioning_map=class_conditioning_bottom_map,
            initial_code=bottom_code,
            mask=generation_mask_bottom_batched
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

        # create JSON response
        response = make_response(top_code_resampled, bottom_code_resampled,
                                 input_conditioning_top,
                                 new_conditioning_map_bottom)

    return response


@app.route('/erase', methods=['POST'])
def erase():
    global transformer_top
    assert transformer_top is not None
    global transformer_bottom
    assert transformer_bottom is not None
    global label_encoders_per_modality
    assert label_encoders_per_modality is not None
    global DEVICE
    assert DEVICE is not None

    layer = str(request.args.get('layer'))
    amplitude = float(request.args.get('eraser_amplitude'))

    top_code, bottom_code = parse_codes(request)
    generation_mask = parse_mask(request).to(DEVICE)[0]

    logmelspectrogram, IF = vqvae.decode_code(top_code,
                                              bottom_code)[0]

    upsampling_f = logmelspectrogram.shape[0] // generation_mask.shape[0]
    upsampling_t = logmelspectrogram.shape[1] // generation_mask.shape[1]

    upsampled_mask = (generation_mask.float().flip(0)
                      .repeat_interleave(upsampling_f, 0)
                      .repeat_interleave(upsampling_t, 1)
                      ).flip(0)
    amplitude_mask = 200 * amplitude * upsampled_mask
    # amplitude_mask = 1 - upsampled_mask

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


def parse_codes(request) -> Tuple[torch.LongTensor,
                                  torch.LongTensor]:
    global transformer_top
    assert transformer_top is not None
    global transformer_bottom
    assert transformer_bottom is not None

    json_data = request.get_json(force=True)

    top_code_flattened_array = json_data['top_code']
    bottom_code_flattened_array = json_data['bottom_code']

    top_code_flattened = torch.LongTensor(top_code_flattened_array
                                          ).unsqueeze(0)
    bottom_code_flattened = torch.LongTensor(bottom_code_flattened_array
                                             ).unsqueeze(0)

    top_code = transformer_top.to_time_frequency_map(
        top_code_flattened, kind='source').to(DEVICE)
    bottom_code = transformer_bottom.to_time_frequency_map(
        bottom_code_flattened, kind='target').to(DEVICE)

    return top_code, bottom_code


def parse_conditioning(request) -> Tuple[torch.LongTensor,
                                         torch.LongTensor,
                                         Mapping[str, ConditioningMap],
                                         Mapping[str, ConditioningMap],
                                         ]:
    global transformer_top
    assert transformer_top is not None
    global transformer_bottom
    assert transformer_bottom is not None
    global label_encoders_per_modality
    assert label_encoders_per_modality is not None

    json_data = request.get_json(force=True)

    if 'top_conditioning' not in json_data.keys():
        return None, None

    conditioning_top = json_data['top_conditioning']
    conditioning_bottom = json_data['bottom_conditioning']

    class_conditioning_top_map = make_conditioning_map(
        json_data['top_conditioning'],
        label_encoders_per_modality)
    class_conditioning_bottom_map = make_conditioning_map(
        json_data['bottom_conditioning'],
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
    global transformer_top
    assert transformer_top is not None
    global transformer_bottom
    assert transformer_bottom is not None

    # flatten codes for sending as lists
    top_code_flattened = transformer_top.flatten_map(
        top_code, kind='source')[0].int().cpu().numpy().tolist()
    bottom_code_flattened = transformer_bottom.flatten_map(
        bottom_code, kind='target')[0].int().cpu().numpy().tolist()

    return flask.jsonify({'top_code': top_code_flattened,
                          'bottom_code': bottom_code_flattened,
                          'top_conditioning': class_conditioning_top_map,
                          'bottom_conditioning': class_conditioning_bottom_map,
                          })


@app.route('/get-audio', methods=['POST'])
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
def codes_to_spectrogram_image_response():
    global SPECTROGRAMS_UPSAMPLING_FACTOR
    assert SPECTROGRAMS_UPSAMPLING_FACTOR is not None

    top_code, bottom_code = parse_codes(request)

    logmelspectrogram_and_IF = vqvae.decode_code(top_code,
                                                 bottom_code)

    # generate spectrogram PNG image
    spectrogram = logmelspectrogram_and_IF[0, 0]
    spectrogram_image_path = make_spectrogram_image(
        spectrogram,
        upsampling_factor=SPECTROGRAMS_UPSAMPLING_FACTOR)

    return flask.send_file(spectrogram_image_path,
                           mimetype="image/png",
                           cache_timeout=-1  # disable cache
                           )


def write_audio_to_file(audio: torch.Tensor):
    """Generate and send MP3 file
    """
    global FS_HZ
    assert FS_HZ is not None
    audio_extension = '.wav'
    # audio_path = tempfile.mktemp() + audio_extension
    audio_path = upload_directory + 'audio.wav'
    audio_np = audio.cpu().numpy()
    with open(audio_path, 'wb') as f:
        soundfile.write(f,
                        audio_np,
                        samplerate=FS_HZ)
    return audio_path
    # return flask.send_file(audio_sample_path, mimetype="audio/wav",
    #                        cache_timeout=-1  # disable cache
    #                        )


if __name__ == '__main__':
    file_handler = logging_handlers.RotatingFileHandler(
        'app.log', maxBytes=10000, backupCount=5)

    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    init_app()
