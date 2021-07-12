from typing import Mapping, Iterable
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pathlib
import json


def dump_label_encoders(label_encoders: Mapping[str, LabelEncoder],
                        savedir_path: pathlib.Path):
    label_encoders_classes: Mapping[str, int] = {}
    for field, label_encoder in label_encoders.items():
        classes_as_list = np.asarray(label_encoder.classes_).tolist()
        label_encoders_classes[field] = classes_as_list  # type: ignore
    with open(savedir_path / 'label_encoders.json', 'w') as f:
        json.dump(label_encoders_classes, f)


def load_label_encoders(path: pathlib.Path):
    label_encoders_classes: Mapping[str, Iterable[int]]
    with open(path, 'r') as f:
        label_encoders_classes = json.load(f)
    label_encoders: Mapping[str, LabelEncoder] = {}
    for field, classes in label_encoders_classes.items():
        le = LabelEncoder().fit(classes)
        label_encoders[field] = le
    return label_encoders
