# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from advertorch.attacks import Attack
from torch.nn import Module
from .params import AttackerParams


class Attacker(Enum):
    NONE = "none"  # Utility for training classifier without any attacker
    PGD_ATTACK = "pgd"
    FGSM_ATTACK = "fgsm"
    CW_ATTACK = "cw"


# Defines a dummy attacker
class NoAttacker(Attack):
    def __init__(self):
        pass

    def perturb(self, x, y=None):
        return x


def create_attacker(
    classifier: Module,
    attacker_type: Attacker,
    params: AttackerParams = AttackerParams(),
) -> Attack:
    if attacker_type == Attacker.PGD_ATTACK:
        from .pgd import make_pgd_attacker, PGDParams

        params = PGDParams(**params)
        attacker = make_pgd_attacker(classifier, params)
    elif attacker_type == Attacker.FGSM_ATTACK:
        from .fgsm import make_fgsm_attacker

        attacker = make_fgsm_attacker(classifier, params)
    elif attacker_type == Attacker.CW_ATTACK:
        from .cw_pgd import make_cw_attacker, CWParams

        params = CWParams(**params)
        attacker = make_cw_attacker(classifier, params)
    elif attacker_type == Attacker.NONE:
        attacker = NoAttacker()
    else:
        raise ValueError()

    return attacker
