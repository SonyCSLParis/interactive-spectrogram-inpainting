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
    _keys: Sequence[bytes]

    """Dataset based on a LMDB database

    Arguments:
        * path, str:
            The path to the directory containing the database
        * active_class_labels, optional, Iterable[str], default=[]:
            If provided,
    """
    def __init__(self, path, classes_for_conditioning: Sequence[str] = [],
                 dataset_db_name: str = 'dataset'):
        self.env = lmdb.open(
            str(path),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=2
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        self.dataset_db = self.env.open_db(dataset_db_name.encode('utf-8'))
        self.__init_indexes()

        self.classes_for_conditioning = classes_for_conditioning

        self.label_encoders: Mapping[str, LabelEncoder]
        if (self.classes_for_conditioning is None
                or len(self.classes_for_conditioning) == 0):
            self.label_encoders = {}
        else:
            with self.env.begin() as txn:
                self.label_encoders = pickle.loads(
                    txn.get('label_encoders'.encode('utf-8')))
                self.label_encoders = self._filter_classes_labels(
                    self.label_encoders)

    def __init_indexes(self):
        """Initialize index-to-database-key mapping"""
        self._keys = []
        with self.env.begin(db=self.dataset_db) as txn:
            c = txn.cursor()
            c.first()
            for key in c.iternext(values=False):
                self._keys.append(key)

    def __len__(self):
        with self.env.begin() as txn:
            length = txn.stat(self.dataset_db)['entries']
        return length

    def _filter_classes_labels(self, class_labels: Mapping[str, any]
                               ) -> Mapping[str, any]:
        return {class_name: class_label
                for class_name, class_label in class_labels.items()
                if class_name in self.classes_for_conditioning}

    def __getitem__(self, index):
        key = self._keys[index]
        with self.env.begin(db=self.dataset_db, write=False) as txn:
            row = pickle.loads(txn.get(key))

        attributes = OrderedDict()
        for class_name in self.classes_for_conditioning:
            attributes[class_name] = row.attributes[class_name].view(1)

        return (torch.from_numpy(row.top), torch.from_numpy(row.bottom),
                attributes)
