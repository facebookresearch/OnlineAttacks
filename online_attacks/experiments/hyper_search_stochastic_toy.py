# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
from torch.utils.data import Dataset

sys.path.append("..")  # Adds higher directory to python modules path.
from online_attacks.online_algorithms import (
    create_online_algorithm,
    compute_knapsack_online_value,
    compute_competitive_ratio,
    AlgorithmType,
    OnlineParams,
    compute_indices,
)
from online_attacks.utils.utils import seed_everything
from online_attacks.datastream import ToyDatastream, ToyDatastream_Stochastic
from online_attacks.utils.parser import ArgumentParser

K = 100
N = 1000
max_perms = 1000
eps = 0.0
knapsack = False
train_loader = ToyDatastream_Stochastic(N, max_perms, eps)


def objective(trial):
    online_params = OnlineParams
    online_params.online_type = [AlgorithmType.STOCHASTIC_VIRTUAL_REF]
    online_params.exhaust = True
    online_params.K = K
    online_params.N = N
    online_params.reference_rank = trial.suggest_int("L", 1, K - 1)
    online_params.threshold = None
    # online_params.threshold = N*trial.suggest_float("threshold", 0.05, 1)
    online_params.threshold = N * trial.suggest_float("threshold", 0.1, 1)
    # train_loader = ToyDatastream_Stochastic(online_params.N, max_perms, eps)
    comp_ratio = run_experiment(online_params, train_loader, knapsack)
    return comp_ratio


def run_experiment(params: OnlineParams, train_loader: Dataset, knapsack: bool):
    offline_algorithm, online_algorithm = create_online_algorithm(params)
    num_perms = len(train_loader)
    comp_ratio_list, online_knapsack_list = [], []
    for i, dataset in enumerate(train_loader):
        offline_dataset, online_dataset = dataset[0], dataset[1]
        indices = compute_indices(online_dataset, [online_algorithm, offline_algorithm])
        comp_ratio_list.append(
            compute_competitive_ratio(
                indices[online_algorithm.name], indices[offline_algorithm.name]
            )
        )

        if knapsack:
            offline_value = sum([x[0] for x in indices[offline_algorithm.name]])
            online_knapsack_list.append(
                compute_knapsack_online_value(indices[online_algorithm.name])
            )

    # Indicator Competitive Ratio
    comp_ratio = np.sum(comp_ratio_list) / (params.K * num_perms)
    print(
        "Competitive Ratio for %s with K = %d is %f "
        % (online_algorithm.name, params.K, comp_ratio)
    )

    # Knapsack Competitive Ratio
    if knapsack:
        comp_ratio = np.sum(online_knapsack_list) / (offline_value * num_perms)
        print(
            "Knapsack Competitive Ratio for %s with K = %d is %f "
            % (online_algorithm.name, params.K, comp_ratio)
        )

    return comp_ratio


def main():
    online_params = OnlineParams
    online_params.online_type = [AlgorithmType.STOCHASTIC_VIRTUAL_REF]
    online_params.exhaust = True
    online_params.K = K
    online_params.N = N
    online_params.reference_rank = 25
    online_params.threshold = 0.2 * N
    train_loader = ToyDatastream_Stochastic(online_params.N, max_perms, eps)
    comp_ratio = run_experiment(online_params, train_loader, knapsack)
    return comp_ratio


if __name__ == "__main__":
    main()
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=1000)

    # # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    # print("    {}: {}".format(key, value))
