# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from omegaconf import MISSING
from .models import CifarModel
from online_attacks.classifiers.dataset import DatasetParams
from online_attacks.attacks import Attacker, AttackerParams
from online_attacks.utils.optimizer import OptimizerParams, OptimizerType
from typing import Optional


@dataclass
class CifarTrainingParams:
    name: str = "cifar"
    model_type: CifarModel = MISSING
    num_epochs: int = 100
    optimizer_params: OptimizerParams = OptimizerParams(
        optimizer_type=OptimizerType.SLS
    )
    dataset_params: DatasetParams = DatasetParams()
    save_model: bool = True
    save_dir: str = "./pretained_models"
    train_on_test: bool = False
    attacker: Attacker = Attacker.NONE
    attacker_params: AttackerParams = AttackerParams()
    model_dir: str = "./pretrained_models/"
    model_attacker: Optional[str] = None
