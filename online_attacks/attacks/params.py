# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class AttackerParams:
    eps: float = MISSING
    clip_min: float = 0.0
    clip_max: float = 1.0
    targeted: bool = False
    # TODO: Add loss_fn
