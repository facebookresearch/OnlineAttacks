# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch.utils.data import Dataset
from torch.distributions import Normal
import torch


class ToyDatastream:
    def __init__(self, N: int, max_perms: int = 120):
        self.perms = []
        total_perms = np.math.factorial(N)
        max_perms = np.minimum(max_perms, total_perms)
        for i in range(
            max_perms
        ):  # (1) Draw N samples from permutations Universe U (#U = k!)
            while True:  # (2) Endless loop
                perm = np.random.permutation(
                    N
                )  # (3) Generate a random permutation form U
                key = tuple(perm)
                if (
                    key not in self.perms
                ):  # (4) Check if permutation already has been drawn (hash table)
                    self.perms.append(key)  # (5) Insert into set
                    break
                pass

    def __getitem__(self, index: int) -> Dataset:
        self.perms[index]
        return self.perms[index]

    def __len__(self) -> int:
        return len(self.perms)


class ToyDatastream_Stochastic:
    def __init__(self, N: int, max_perms: int = 120, eps: float = 1.0):
        self.perms = []
        self.noisy_perms = []
        total_perms = np.math.factorial(N)
        max_perms = np.minimum(max_perms, total_perms)
        mean = torch.zeros(N)
        std = eps * torch.ones(N)
        normal = Normal(mean, std)
        for i in range(
            max_perms
        ):  # (1) Draw N samples from permutations Universe U (#U = k!)
            while True:  # (2) Endless loop
                perm = np.random.permutation(
                    N
                )  # (3) Generate a random permutation form U
                key = tuple(perm)
                if (
                    key not in self.perms
                ):  # (4) Check if permutation already has been drawn (hash table)
                    self.perms.append(key)  # (5) Insert into set
                    noise = normal.rsample()
                    noisy_key = tuple(perm + noise.numpy())
                    self.noisy_perms.append(noisy_key)
                    break
                pass

    def __getitem__(self, index: int) -> Dataset:
        return [self.perms[index], self.noisy_perms[index]]

    def __len__(self) -> int:
        return len(self.perms)
