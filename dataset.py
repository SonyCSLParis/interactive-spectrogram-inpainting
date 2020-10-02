# TODO(theis): DELETE THIS from project root!!!!
# only copied here to make previous pickled datasets load...

from typing import Iterable, Mapping, Optional, Sequence
import os
import pickle
import numpy as np
from collections import namedtuple, OrderedDict

from sklearn.preprocessing import LabelEncoder
import lmdb

import torch
from torch.utils.data import Dataset
from torchvision import datasets


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'attributes', 'filename'])


class LMDBDataset(Dataset):
    """Dataset based on a LMDB database

    Arguments:
        * path, str:
            The path to the directory containing the database
        * active_class_labels, optional, Iterable[str], default=[]:
            If provided,
    """
    def __init__(self, path, classes_for_conditioning: Sequence[str] = []):
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

        self.label_encoders: Mapping[str, LabelEncoder]
        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

            if (self.classes_for_conditioning is None
                    or len(self.classes_for_conditioning) == 0):
                self.label_encoders = {}
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

        attributes = OrderedDict()
        for class_name in self.classes_for_conditioning:
            attributes[class_name] = row.attributes[class_name].view(1)

        return (torch.from_numpy(row.top), torch.from_numpy(row.bottom),
                attributes)
