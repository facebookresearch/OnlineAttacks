# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from advertorch.attacks import GradientSignAttack
from torch.nn import Module
from .params import AttackerParams


def make_fgsm_attacker(
    classifier: Module, params: AttackerParams = AttackerParams()
) -> GradientSignAttack:
    attacker = GradientSignAttack(classifier, **params)
    return attacker

