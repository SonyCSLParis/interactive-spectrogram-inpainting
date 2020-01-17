import argparse
import pathlib
import json
import os

from sklearn.model_selection import train_test_split

from GANsynth_pytorch.pytorch_nsynth_lib.nsynth import (
    NSynth, WavToSpectrogramDataLoader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_paths', type=str, nargs='+')
    parser.add_argument('--output_directory', type=str, required=True)
    parser.add_argument('--valid_pitch_range', type=int, nargs=2,
                        default=None)
    parser.add_argument('--train_size', type=float, default=0.8)
    args = parser.parse_args()

    print(args)

    MAIN_OUTPUT_DIRECTORY = pathlib.Path(args.output_directory)
    os.makedirs(MAIN_OUTPUT_DIRECTORY, exist_ok=False)
    with open(MAIN_OUTPUT_DIRECTORY / 'command_line_parameters.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # collect all filenames and associated metadata
    filenames = []
    all_json_data = {}
    for dataset_path in args.dataset_paths:
        dataset_path = pathlib.Path(dataset_path).expanduser().absolute()
        audio_directory_path = dataset_path / 'audio/'
        json_data_path = dataset_path / 'examples.json'
        dataset = NSynth(audio_directory_paths=audio_directory_path,
                         json_data_path=json_data_path,
                         valid_pitch_range=args.valid_pitch_range)
        # filenames.extend(dataset.filenames)
        all_json_data.update(dataset.json_data)

    # create json_data split
    print('Create json_data split')
    json_data_splits_as_lists = train_test_split(
        list(all_json_data.items()),
        train_size=args.train_size
    )

    json_data_train_split, json_data_valid_split = [
        {key: value
         for key, value in split_as_list}
        for split_as_list in json_data_splits_as_lists
    ]

    print('Dump splits')
    split_names = ['train', 'valid']
    for split_name, json_data_split in zip(
            split_names,
            [json_data_train_split, json_data_valid_split]):
        file_name = 'examples.json'
        os.makedirs(MAIN_OUTPUT_DIRECTORY / split_name)
        file_path = MAIN_OUTPUT_DIRECTORY / split_name / file_name
        with open(file_path, 'w') as f:
            json.dump(json_data_split, f, indent=4)
