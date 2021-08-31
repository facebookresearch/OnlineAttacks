# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from advertorch.attacks import CarliniWagnerL2Attack
from torch.nn import Module
from dataclasses import dataclass
from .params import AttackerParams


@dataclass
class CWParams(AttackerParams):
    confidence: float = 0
    learning_rate: float = 0.01
    binary_search_steps: int = 9
    max_iterations: int = 10000
    abort_early: bool = True
    initial_const: float = 0.001


def make_cw_attacker(
    classifier: Module, params: CWParams = CWParams()
) -> CarliniWagnerL2Attack:
    attacker = CarliniWagnerL2Attack(
        classifier,
        classifier.num_classes,
        confidence=params.confidence,
        targeted=params.targeted,
        learning_rate=params.learning_rate,
        binary_search_steps=params.binary_search_steps,
        max_iterations=params.max_iterations,
        abort_early=params.abort_early,
        initial_const=params.initial_const,
        clip_min=params.clip_min,
        clip_max=params.clip_max,
    )
    return attacker

