# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import optim
import copy


class Extragradient(optim.Optimizer):
    def __init__(self, optimizer, params):
        super(Extragradient, self).__init__(params, optimizer.defaults)
        self.params_copy = []
        self.optimizer = optimizer
        self.extrapolation_flag = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.extrapolation_flag is False:
            for group in self.param_groups:
                group["params_copy"] = copy.deepcopy(group["params"])
            self.optimizer.step()
            self.extrapolation_flag = True

        else:
            for group in self.param_groups:
                for p, p_copy in zip(group["params"], group["params_copy"]):
                    p.data = p_copy.data
            self.optimizer.step()
            self.extrapolation_flag = False

        return loss
