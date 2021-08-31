# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from omegaconf import MISSING
from .models import MnistModel
from online_attacks.classifiers.dataset import DatasetParams
from online_attacks.attacks import Attacker, AttackerParams
from online_attacks.utils.optimizer import OptimizerParams
from typing import Optional


@dataclass
class MnistTrainingParams:
    name: str = "mnist"
    model_type: MnistModel = MISSING
    num_epochs: int = 100
    dataset_params: DatasetParams = DatasetParams()
    optimizer_params: OptimizerParams = OptimizerParams(lr=1e-3)
    save_model: bool = True
    save_dir: str = "./pretained_models"
    train_on_test: bool = False
    attacker: Attacker = Attacker.NONE
    attacker_params: AttackerParams = AttackerParams()
    model_dir: str = "./pretrained_models/"
    model_attacker: Optional[str] = None
