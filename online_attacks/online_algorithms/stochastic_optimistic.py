# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Algorithm, AlgorithmType
from typing import Optional
import numpy as np


class StochasticOptimistic(Algorithm):
    def __init__(
        self, N: int, k: int, threshold: Optional[int] = None, exhaust: bool = False
    ):
        """ Construct Stochastic Optimistic
        Parameters:
            N (int)           -- number of data points
            k (int)           -- number of attacks
            threshold (int)   -- threshold for t
            exhaust (bool)    -- whether to exhaust K
        """
        super().__init__(k)
        self.N = N

        if threshold is None:
            threshold = np.floor(N / np.e)
        self.threshold = threshold

        self.R = []
        self.sampling_phase = True
        self.exhaust = exhaust
        self.name = AlgorithmType.STOCHASTIC_OPTIMISTIC.name

    def reset(self):
        super().reset()
        self.R = []
        self.sampling_phase = True

    def action(self, value: float, index: int):
        if self.sampling_phase:
            self.R.append([value, index])
            self.R.sort(key=lambda tup: tup[0], reverse=True)  # sorts in place
            self.R = self.R[: self.k]

            if index >= self.threshold:
                self.sampling_phase = False
        else:
            num_picked = len(self.S)
            num_left_to_pick = self.k - num_picked
            num_samples_left = self.N - index
            if len(self.R) > 0:
                self.k_value, self.k_index = self.R[-1]
            if num_left_to_pick > 0:
                if num_samples_left <= num_left_to_pick and self.exhaust:
                    # Just Pick the last samples to exhaust K
                    self.S.append([value, index])
                elif value > self.k_value:
                    # Update and pick
                    self.S.append([value, index])
                    self.R = self.R[:-2]


if __name__ == "__main__":
    algorithm = StochasticOptimistic(10, 1, 5)
    algorithm.reset()
    algorithm.action(1, 1)
