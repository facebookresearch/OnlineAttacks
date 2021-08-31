# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .mnist import load_mnist_classifier, MnistModel, load_mnist_dataset
from .cifar import load_cifar_classifier, CifarModel, load_cifar_dataset
from .dataset import DatasetType
from torch.nn import Module
from dataclasses import dataclass
from omegaconf import MISSING, OmegaConf
from typing import Any
import torch
from torch.nn import CrossEntropyLoss

from online_attacks.attacks import (
    Attacker,
    AttackerParams,
    create_attacker,
    compute_attack_success_rate,
)
from online_attacks import datastream


@dataclass
class ModelParams:
    model_type: Any = MISSING
    model_name: str = MISSING


@dataclass
class AttackerConfig:
    attacker: Attacker = Attacker.NONE
    model: ModelParams = ModelParams()
    params: AttackerParams = AttackerParams()


def eval_classifier(
    dataset: DatasetType,
    target_model: ModelParams,
    attacker: AttackerConfig = AttackerConfig(),
    batch_size=1000,
    model_dir="/checkpoint/hberard/OnlineAttack/pretained_models/",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_classifier = load_classifier(
        dataset,
        target_model.model_type,
        name=target_model.model_name,
        model_dir=model_dir,
        device=device,
        eval=True,
    )
    source_classifier = load_classifier(
        dataset,
        attacker.model.model_type,
        name=attacker.model.model_name,
        model_dir=model_dir,
        device=device,
        eval=True,
    )

    attacker = OmegaConf.structured(attacker)
    attacker = create_attacker(source_classifier, attacker.attacker, attacker.params)

    dataset = load_dataset(dataset, train=False)
    transform = datastream.Compose(
        [
            datastream.ToDevice(device),
            datastream.AttackerTransform(attacker),
            datastream.ClassifierTransform(target_classifier),
        ]
    )
    stream = datastream.BatchDataStream(
        dataset, batch_size=batch_size, transform=transform, return_target=True
    )
    fool_rate, knapsack = compute_attack_success_rate(
        stream, CrossEntropyLoss(reduction="sum")
    )
    print(fool_rate / len(dataset) * 100)


def load_dataset(dataset: DatasetType, train: bool = False):
    if dataset == DatasetType.MNIST:
        return load_mnist_dataset(train=train)
    elif dataset == DatasetType.CIFAR:
        return load_cifar_dataset(train=train)
    else:
        raise ValueError()


def load_classifier(
    dataset: DatasetType,
    model_type,
    name: str = None,
    model_dir: str = None,
    device=None,
    eval=False,
) -> Module:
    if dataset == DatasetType.MNIST:
        assert isinstance(model_type, MnistModel)
        return load_mnist_classifier(
            model_type, name=name, model_dir=model_dir, device=device, eval=eval
        )
    elif dataset == DatasetType.CIFAR:
        assert isinstance(model_type, CifarModel)
        return load_cifar_classifier(
            model_type, name=name, model_dir=model_dir, device=device, eval=eval
        )
    else:
        raise ValueError()
