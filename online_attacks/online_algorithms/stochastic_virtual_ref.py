# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Algorithm, AlgorithmType
from typing import Optional
import numpy as np


class StochasticVirtualRef(Algorithm):
    DEFAULT_RANK = 1

    def __init__(
        self,
        N: int,
        k: int,
        l: Optional[int] = None,
        threshold: Optional[int] = None,
        exhaust: bool = False,
    ):
        """ Construct Stochastic Virtual
        Parameters:
            N (int)                 -- number of data points
            k (int)                 -- number of attacks
            reference rank (int)    -- threshold for t
            threshold (int)         -- threshold for t
            exhaust (bool)          -- whether to exhaust K
        """
        super().__init__(k)
        self.N = N
        self.l = l
        if self.l is None:
            self.l = self.DEFAULT_RANK

        if threshold is None:
            threshold = np.floor(N / np.e)
        self.threshold = threshold

        self.R = []
        self.sampling_phase = True
        self.exhaust = False  # exhaust
        self.name = AlgorithmType.STOCHASTIC_VIRTUAL_REF.name

    def reset(self):
        super().reset()
        self.R = []
        self.sampling_phase = True

    def action(self, value: float, index: int):
        if self.sampling_phase:
            self.R.append([value, index])
            self.R.sort(key=lambda tup: tup[0], reverse=True)  # sorts in place
            # self.R = self.R[:self.l]
            self.R = self.R[: self.k]

            if index >= self.threshold:
                self.sampling_phase = False
        else:
            l_value, l_index = self.R[self.l]
            num_picked = len(self.S)
            num_left_to_pick = self.k - num_picked
            num_samples_left = self.N - index
            if (
                num_samples_left <= num_left_to_pick
                and self.exhaust
                and num_left_to_pick > 0
            ):
                # Just Pick the last samples to exhaust K
                self.S.append([value, index])
            elif value > l_value and num_left_to_pick > 0:
                # Update and pick
                self.S.append([value, index])
                self.R.append([value, index])
                self.R.sort(key=lambda tup: tup[0], reverse=True)  # sorts in place
                self.R = self.R[: self.k]
                # Update L
                self.l = min(self.k - 1, self.l + 1)


if __name__ == "__main__":
    algorithm = StochasticModifiedVirtual(10, 1, 5)
    algorithm.reset()
    algorithm.action(1, 1)
