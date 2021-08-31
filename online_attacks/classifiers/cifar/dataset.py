# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from online_attacks.classifiers.dataset import DatasetParams


def load_cifar_dataset(
    params: DatasetParams = DatasetParams(), train: bool = True
) -> CIFAR10:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return CIFAR10(
        root=params.data_dir, train=train, transform=transform, download=params.download
    )


def create_cifar_loaders(
    params: DatasetParams = DatasetParams()
) -> (DataLoader, DataLoader):
    trainset = load_cifar_dataset(params, True)
    testset = load_cifar_dataset(params, False)

    train_loader = DataLoader(
        trainset,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        num_workers=params.num_workers,
    )
    test_loader = DataLoader(
        testset, batch_size=params.test_batch_size, num_workers=params.num_workers
    )

    return train_loader, test_loader
