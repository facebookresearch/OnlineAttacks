# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from enum import Enum


class AlgorithmType(Enum):
    OFFLINE = "offline_algorithm"
    STOCHASTIC_VIRTUAL = "stochastic_virtual"
    STOCHASTIC_OPTIMISTIC = "stochastic_optimistic"
    STOCHASTIC_MODIFIED_VIRTUAL = "stochastic_modified_virtual"
    STOCHASTIC_VIRTUAL_REF = "stochastic_virtual_ref"
    STOCHASTIC_SINGLE_REF = "stochastic_single_ref"
    RANDOM = "random"


class Algorithm:
    def __init__(self, k: int):
        self.k = k
        self.S = []

    def action(self, value: float, index: int):
        raise NotImplementedError()

    def reset(self):
        self.S = []

    def get_indices(self):
        assert len(self.S) <= self.k
        return self.S


class RandomAlgorithm(Algorithm):
    def __init__(self, N: int, k: int):
        super().__init__(k)
        self.N = N
        self.random_permutation = random.sample(range(self.N), self.k)
        self.name = AlgorithmType.RANDOM.name

    def action(self, value: float, index: int):
        if index in self.random_permutation:
            self.S.append([value, index])

    def reset(self):
        self.S = []
        self.random_permutation = random.sample(range(self.N), self.k)

