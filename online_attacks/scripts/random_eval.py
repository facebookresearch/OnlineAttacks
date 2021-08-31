# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from omegaconf import OmegaConf
import random
import glob
import os
import copy
import submitit

from online_attacks.classifiers import DatasetType, MnistModel, CifarModel
from online_attacks.scripts.online_attacks_sweep import OnlineAttackExp, create_params
from online_attacks.scripts.online_attack_params import OnlineAttackParams as Params
from online_attacks.scripts.eval_all import eval_comp_ratio
from online_attacks.online_algorithms import AlgorithmType
from online_attacks.launcher import Launcher
from online_attacks.attacks import Attacker


def random_model(model_dir, dataset, pattern="train_*.pth"):
    while True:
        if dataset == DatasetType.MNIST:
            model_type = random.choice(list(MnistModel))
        elif dataset == DatasetType.CIFAR:
            model_type = random.choice(list(CifarModel))
        else:
            raise ValueError("%s not in DatasetType" % dataset)

        list_models = glob.glob(
            os.path.join(model_dir, dataset.value, model_type.value, pattern)
        )
        if len(list_models) > 0:
            break

    list_models = [
        os.path.splitext(os.path.basename(model))[0] for model in list_models
    ]
    model_name = random.choice(list_models)

    return model_type, model_name


class Run:
    def __init__(self, k, num_runs=1, same_model_type=False, pattern="train_*.pth"):
        self.same_model_type = same_model_type
        self.pattern = pattern
        self.num_runs = num_runs
        self.k = k

    def create_random_config(self, params):
        params.model_type, params.model_name = random_model(
            params.model_dir, params.dataset, self.pattern
        )

        while True:
            model_type, model_name = random_model(
                params.model_dir, params.dataset, self.pattern
            )
            if self.same_model_type:
                model_type = params.model_type

            if model_type != params.model_type or model_name != params.model_name:
                break

        params.seed = random.randrange(10000)

        return params, model_type, model_name

    def __call__(self, params):
        for i in range(self.num_runs):
            params, model_type, model_name = self.create_random_config(params)
            params = create_params(params)
            for k in self.k:
                params.online_params.K = k
                params.online_params.online_type = list(AlgorithmType)
                logger, list_records = OnlineAttackExp()(params, 1, max_num_runs=False)
                eval_comp_ratio(logger, model_type, model_name, list_records)
        print("done")

    def checkpoint(self, params) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(params)  # submits to requeuing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType
    )
    parser.add_argument(
        "--attacker_type", default=Attacker.FGSM_ATTACK, type=Attacker, choices=Attacker
    )
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--name", default="default", type=str)
    args, _ = parser.parse_known_args()

    params = OmegaConf.structured(Params(**vars(args)))

    # Hack to be able to parse num_runs without affecting params
    parser.add_argument("--num_runs", default=1000, type=int)
    parser.add_argument("--slurm", type=str, default="")
    parser.add_argument("--pattern", type=str, default="train_*.pth")
    parser.add_argument("--same_model_type", action="store_true")
    parser.add_argument("--k", nargs="+", type=int, default=[2, 5, 10, 100, 1000])

    args = parser.parse_args()

    if args.dataset == DatasetType.MNIST:
        num_runs = 10
        args.num_runs = int(args.num_runs / num_runs) + 1
    elif args.dataset == DatasetType.CIFAR:
        num_runs = 5
        args.num_runs = int(args.num_runs / num_runs) + 1

    run = Run(args.k, num_runs, args.same_model_type, args.pattern)
    launcher = Launcher(run, args.slurm)
    args = [copy.deepcopy(params)] * args.num_runs
    launcher.batch_launch(args)
