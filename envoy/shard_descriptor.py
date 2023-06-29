#Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""
import struct,gzip
from tensorflow.keras.utils import to_categorical
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

class mnistShardDataset(ShardDataset):
    """MNIST Shard dataset class."""

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
    
class mnistShardDescriptor(ShardDescriptor):
    """MNIST Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MNIST ShardDescriptor."""
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
        return mnistShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['28', '28', '1']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['10']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'MNIST dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    

    
    def read_idx(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
            


    @staticmethod
    def load_prepare_data(rank: int, worldsize: int) -> Tuple[tf.data.Dataset]:
        """Load and prepare dataset."""


        X = []
        y = []
        train_images = mnistShardDescriptor.read_idx("./dataset/train-images.idx3-ubyte")
        train_labels = mnistShardDescriptor.read_idx("./dataset/train-labels.idx1-ubyte")
        test_images = mnistShardDescriptor.read_idx("./dataset/test-images.idx3-ubyte")
        test_labels = mnistShardDescriptor.read_idx("./dataset/test-labels.idx1-ubyte")

        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        print('MNIST data loaded!')
        return (train_images, train_labels), (test_images, test_labels)






