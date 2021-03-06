from interactive_spectrogram_inpainting.utils.datasets.label_encoders import (
    load_label_encoders)

import pathlib
from typing import Mapping, Sequence, Union
import pickle
from collections import namedtuple, OrderedDict
from sklearn.preprocessing import LabelEncoder
import lmdb

import torch
from torch.utils.data import Dataset


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'attributes', 'filename'])


class LMDBDataset(Dataset):
    _keys: Sequence[bytes]
    label_encoders: Mapping[str, LabelEncoder]

    """Dataset based on a LMDB database

    Arguments:
        * path, str:
            The path to the directory containing the database
        * active_class_labels, optional, Iterable[str], default=[]:
            If provided,
    """
    def __init__(self, path: Union[str, pathlib.Path],
                 classes_for_conditioning: Sequence[str] = [],
                 dataset_db_name: str = 'codes'):
        map_size = 100 * 1024 * 1024 * 1024
        self.env = lmdb.open(
            str(path),
            max_readers=32,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=2,
            map_size=map_size
            )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        self.dataset_db = self.env.open_db(dataset_db_name.encode('utf-8'))
        self.__init_indexes()

        self.classes_for_conditioning = classes_for_conditioning

        if (self.classes_for_conditioning is None
                or len(self.classes_for_conditioning) == 0):
            self.label_encoders = {}
        else:
            self.label_encoders = self._filter_classes_labels(
                load_label_encoders(
                    pathlib.Path(path) / 'label_encoders.json')
            )

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
