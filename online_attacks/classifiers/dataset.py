# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from dataclasses import dataclass


class DatasetType(Enum):
    MNIST = "mnist"
    CIFAR = "cifar"


@dataclass
class DatasetParams:
    data_dir: str = "./data"
    batch_size: int = 256
    test_batch_size: int = 512
    num_workers: int = 4
    shuffle: bool = True
    download: bool = True
