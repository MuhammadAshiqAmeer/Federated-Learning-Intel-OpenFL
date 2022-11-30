#Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""
from typing import List, Tuple
import logging
import os
from typing import List
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)

class LemMelShardDataset(ShardDataset):
    """Lem-Mel Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1):
        """Pick rank-specific subset of (x, y)"""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x[self.rank - 1::self.worldsize]
        self.y = y[self.rank - 1::self.worldsize]

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)
    
class LemMelShardDescriptor(ShardDescriptor):
    """Lem-Mel Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize Lem-Mel ShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        (x_train, y_train), (x_test, y_test) = self.load_prepare_data(rank=self.rank, worldsize=self.worldsize)

        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }


    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return LemMelShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['120', '120', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'LemMel dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
    @staticmethod
    def load_prepare_data(rank: int, worldsize: int) -> Tuple[tf.data.Dataset]:
        """Load and prepare dataset."""
        local_file_path = './dataset/'


        X = []
        y = []

        source_path='./dataset/'
        for child in os.listdir(source_path):
            sub_path = os.path.join(source_path, child)
            if os.path.isdir(sub_path):
                for data_file in os.listdir(sub_path):
                    X_i = Image.open(os.path.join(sub_path, data_file))
                    X_i = np.array(X_i.resize((120,120))) / 255.0
                    X.append(X_i)
                    y.append(child)
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(np.array(X), np.array(y),test_size=0.2, random_state=42)
        y_train=np.asarray(y_train).reshape((-1,1))
        y_test=np.asarray(y_test).reshape((-1,1))
        print('Lem-Mel data was loaded!')
        return (x_train, y_train), (x_test, y_test)
        
