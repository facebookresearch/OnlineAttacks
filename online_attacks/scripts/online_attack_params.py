# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import MISSING
from dataclasses import dataclass
from typing import Any

from online_attacks.classifiers import DatasetType, MnistModel, CifarModel
from online_attacks.attacks import Attacker, AttackerParams
from online_attacks.online_algorithms import OnlineParams


@dataclass
class OnlineAttackParams:
    name: str = "default"
    dataset: DatasetType = MISSING
    model_name: str = MISSING
    model_type: Any = MISSING
    model_dir: str = "/checkpoint/hberard/OnlineAttack/pretained_models/"
    attacker_type: Attacker = MISSING
    attacker_params: AttackerParams = AttackerParams()
    online_params: OnlineParams = OnlineParams(exhaust=True)
    save_dir: str = "/checkpoint/hberard/OnlineAttack/results_icml/${name}"
    seed: int = 1234
    batch_size: int = 1000


@dataclass
class CifarParams(OnlineAttackParams):
    model_type: CifarModel = MISSING


@dataclass
class MnistParams(OnlineAttackParams):
    model_type: MnistModel = MISSING
