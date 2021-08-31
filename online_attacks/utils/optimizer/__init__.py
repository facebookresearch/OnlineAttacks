# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from .sls import Sls


class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"
    SLS = "sls"


@dataclass
class OptimizerParams:
    optimizer_type: OptimizerType = OptimizerType.ADAM
    lr: float = 1e-3


def create_optimizer(
    params, optimizer_params: OptimizerParams, n_batches_per_epoch: Optional[int] = None
):
    if optimizer_params.optimizer_type == OptimizerType.SGD:
        from torch.optim import SGD

        return SGD(params, lr=optimizer_params.lr)

    elif optimizer_params.optimizer_type == OptimizerType.ADAM:
        from torch.optim import Adam

        return Adam(params, lr=optimizer_params.lr)

    elif optimizer_params.optimizer_type == OptimizerType.SLS:
        if n_batches_per_epoch is None:
            return Sls(params)
        return Sls(params, n_batches_per_epoch=n_batches_per_epoch)
