# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Algorithm, AlgorithmType


class OfflineAlgorithm(Algorithm):
    def __init__(self, k: int):
        """ Construct Offline Algorithm
        Parameters:
            k (int)     -- number of attacks
        """
        super().__init__(k)
        self.name = AlgorithmType.OFFLINE.name

    def action(self, value: float, index: int):
        self.S.append([value, index])
        self.S.sort(key=lambda tup: tup[0], reverse=True)  # sorts in place
        self.S = self.S[: self.k]


if __name__ == "__main__":
    algorithm = OfflineAlgorithm(1)
    algorithm.reset()
    algorithm.action(1, 1)
