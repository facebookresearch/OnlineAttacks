# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable
from torch.nn import CrossEntropyLoss, Module
import torch
from .attacks_registry import Attacker, AttackerParams, create_attacker, NoAttacker


def compute_attack_success_rate(
    datastream: Iterable, loss: Module = CrossEntropyLoss(reduction="sum")
):
    adv_correct = 0
    num_samples = 0
    total_loss = 0
    for x, target in datastream:
        with torch.no_grad():
            total_loss += loss(x, target)
            pred = x.max(1, keepdim=True)[1]
            adv_correct += pred.eq(target.view_as(pred)).sum().item()
            num_samples += len(x)
    fool_rate = num_samples - adv_correct

    return fool_rate, float(total_loss)
