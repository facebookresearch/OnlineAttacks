# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
from typing import Optional
import torch.optim as optim
import os
from .params import MnistTrainingParams
from .dataset import create_mnist_loaders
from .models import load_mnist_classifier
from online_attacks.classifiers.trainer import Trainer
from online_attacks.attacks import create_attacker
from online_attacks.utils.optimizer import create_optimizer


def train_mnist(
    params: MnistTrainingParams = MnistTrainingParams(),
    device: Optional[torch.device] = None,
) -> nn.Module:
    train_loader, test_loader = create_mnist_loaders(params.dataset_params)
    model = load_mnist_classifier(params.model_type)
    optimizer = create_optimizer(model.parameters(), params.optimizer_params)
    if params.train_on_test:
        train_loader, test_loader = test_loader, train_loader

    # TODO: Implement Ensemble Adversarial Training, where a list of attacker is provided
    model_attacker = model
    if params.model_attacker is not None:
        model_attacker = load_mnist_classifier(
            params.model_type,
            name=params.model_attacker,
            model_dir=params.model_dir,
            device=device,
        )

    attacker = create_attacker(model_attacker, params.attacker, params.attacker_params)
    trainer = Trainer(
        model, train_loader, test_loader, optimizer, attacker=attacker, device=device
    )

    filename = None
    if params.save_model:
        filename = os.path.join(
            params.save_dir, "mnist", params.model_type.value, "%s.pth" % (params.name)
        )
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    for epoch in range(1, params.num_epochs):
        trainer.train(epoch)
        trainer.test(epoch)
        if params.save_model:
            torch.save(model.state_dict(), filename)

    return model
