# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Algorithm, AlgorithmType
import numpy as np
import math


def create_r_default(x):
    # Take a list of (start, end, value) and return a dict with all keys between start and end having the value `value`.
    data = {}
    for start, end, value in x:
        data.update({i: value for i in range(start, end + 1)})
    return data


class StochasticSingleRef(Algorithm):
    # Default values from this paper: https://drops.dagstuhl.de/opus/volltexte/2019/11514/pdf/LIPIcs-ISAAC-2019-18.pdf
    C_DEFAULT = {
        "default": 0.3678,
        1: 0.3678,
        2: 0.2545,
        3: 0.3475,
        4: 0.2928,
        5: 0.2525,
        6: 0.2217,
        7: 0.2800,
        8: 0.2549,
        9: 0.2338,
        10: 0.2159,
        11: 0.2570,
        12: 0.2410,
        13: 0.2267,
        14: 0.2140,
        15: 0.2026,
        16: 0.1924,
        17: 0.2231,
        18: 0.2133,
        19: 0.2042,
        20: 0.1959,
        100: 0.1331,
    }

    # Take a list of (start, end, value)
    R_DEFAULT = create_r_default(
        [
            (1, 2, 1),
            (3, 6, 2),
            (7, 10, 3),
            (11, 16, 4),
            (17, 22, 5),
            (23, 28, 6),
            (29, 35, 7),
            (36, 42, 8),
            (43, 50, 9),
            (51, 58, 10),
            (59, 67, 11),
            (68, 76, 12),
            (77, 85, 13),
            (86, 95, 14),
            (96, 100, 15),
        ]
    )

    @classmethod
    def get_default_c(cls, k: int) -> float:
        if k in cls.C_DEFAULT:
            return cls.C_DEFAULT[k]
        elif k > 100:
            return 0.13  # This is a guess based on the values ffrom the paper
        else:
            return cls.C_DEFAULT["default"]

    @classmethod
    def get_default_r(cls, k: int) -> int:
        if k in cls.R_DEFAULT:
            return cls.R_DEFAULT[k]
        elif k > 100:
            return int(
                10 + 0.5 * (math.sqrt(1 + 4 * (k - 96) - 1))
            )  # This is an heuristic somewhat interpolating the other values for k > 100
        else:
            return 1

    def __init__(self, N: int, k: int, r: int, threshold: int, exhaust: bool):
        """ Construct Stochastic Optimistic
        Parameters:
            N (int)           -- number of data points
            k (int)           -- number of attacks
            r (int)           -- reference rank
            threshold (int)   -- threshold for t
            exhaust (bool)    -- whether to exhaust K
        """
        super().__init__(k)
        self.N = N
        self.r = self.get_default_r(k)

        if threshold is None:
            threshold = np.floor(self.get_default_c(k) * N + 1)

        self.threshold = threshold
        self.R = []
        self.sampling_phase = True
        self.exhaust = exhaust
        self.name = AlgorithmType.STOCHASTIC_SINGLE_REF.name

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
            if num_left_to_pick > 0:
                r_value, r_index = self.R[self.r - 1]  # 0-based indexing.
                if num_samples_left <= num_left_to_pick and self.exhaust:
                    # Just Pick the last samples to exhaust K
                    self.S.append([value, index])
                elif value > r_value:
                    # Pick
                    self.S.append([value, index])


if __name__ == "__main__":
    algorithm = StochasticSingleRef(10, 1, 5)
    algorithm.reset()
    algorithm.action(1, 1)
