# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from advertorch.attacks import PGDAttack, L1PGDAttack, L2PGDAttack, LinfPGDAttack
from dataclasses import dataclass
from torch.nn import Module
from .params import AttackerParams


@dataclass
class PGDParams(AttackerParams):
    nb_iter: int = 40
    eps_iter: float = 0.01
    rand_init: bool = True
    norm: str = "Linf"


def make_pgd_attacker(classifier: Module, params: PGDParams = PGDParams()) -> PGDAttack:

    if params.norm == "Linf":
        attacker = LinfPGDAttack(
            classifier,
            eps=params.eps,
            nb_iter=params.nb_iter,
            eps_iter=params.eps_iter,
            rand_init=params.rand_init,
            clip_min=params.clip_min,
            clip_max=params.clip_max,
            targeted=params.targeted,
        )
    elif params.norm == "L2":
        attacker = L2PGDAttack(
            classifier,
            eps=params.eps,
            nb_iter=params.nb_iter,
            eps_iter=params.eps_iter,
            rand_init=params.rand_init,
            clip_min=params.clip_min,
            clip_max=params.clip_max,
            targeted=params.targeted,
        )
    elif params.norm == "L1":
        attacker = L1PGDAttack(
            classifier,
            eps=params.eps,
            nb_iter=params.nb_iter,
            eps_iter=params.eps_iter,
            rand_init=params.rand_init,
            clip_min=params.clip_min,
            clip_max=params.clip_max,
            targeted=params.targeted,
        )

    return attacker

