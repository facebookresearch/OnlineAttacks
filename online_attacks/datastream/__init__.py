# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .toy_data import *
from .datastream import *
import torchvision
from torchvision import transforms


def load_dataset(dataset_name, args):
    if dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST(
            "./data", train=False, transform=transform, download=True
        )
    else:
        raise ValueError()

    return dataset
