from typing import Iterable, Mapping, Optional
import os
import pickle
import numpy as np
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'attributes', 'filename'])


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename


class LMDBDataset(Dataset):
    """Dataset based on a LMDB database

    Arguments:
        * path, str:
            The path to the directory containing the database
        * active_class_labels, optional, Iterable[str], default=[]:
            If provided,
    """
    def __init__(self, path, classes_for_conditioning: Optional[Iterable[str]] = None):
        self.env = lmdb.open(
            str(path),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.classes_for_conditioning = classes_for_conditioning

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

            if (self.classes_for_conditioning is None
                    or len(self.classes_for_conditioning) == 0):
                self.label_encoders = []
            else:
                self.label_encoders = pickle.loads(
                    txn.get('label_encoders'.encode('utf-8')))
                self.label_encoders = self._filter_classes_labels(
                    self.label_encoders)

    def __len__(self):
        return self.length

    def _filter_classes_labels(self, class_labels: Mapping[str, any]
                               ) -> Mapping[str, any]:
        return {class_name: class_label
                for class_name, class_label in class_labels.items()
                if class_name in self.classes_for_conditioning}

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        attributes = {
            attribute_name: attribute_value.view(1)
            for attribute_name, attribute_value in row.attributes.items()}

        return (torch.from_numpy(row.top), torch.from_numpy(row.bottom),
                *attributes.values())
