# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from omegaconf import OmegaConf
import os

from online_attacks.classifiers import MnistModel, DatasetType, CifarModel
from online_attacks.attacks import Attacker
from online_attacks.online_algorithms import AlgorithmType
from online_attacks.launcher import Launcher
from online_attacks.scripts.online_attacks_sweep import OnlineAttackExp, create_params
from online_attacks.scripts.online_attack_params import Params


if __name__ == "__main__":
    # Initialize submitit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType
    )
    parser.add_argument(
        "--attacker_type", default=Attacker.FGSM_ATTACK, type=Attacker, choices=Attacker
    )
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--name", default="default", type=str)

    # Hack to be able to parse either MnistModel or CifarModel and to add slurm config
    args, _ = parser.parse_known_args()
    params = OmegaConf.structured(Params(**vars(args)))

    parser.add_argument("--slurm", type=str, default="")
    if args.dataset == DatasetType.MNIST:
        parser.add_argument(
            "--model_type",
            nargs="+",
            default=MnistModel,
            type=MnistModel,
            choices=MnistModel,
        )
    elif args.dataset == DatasetType.CIFAR:
        parser.add_argument(
            "--model_type",
            nargs="+",
            default=CifarModel,
            type=CifarModel,
            choices=CifarModel,
        )
    parser.add_argument("--num_runs", default=100, type=int)

    args = parser.parse_args()

    launcher = Launcher(OnlineAttackExp.run, slurm=args.slurm, checkpointing=True)

    for model_type in args.model_type:
        params.model_type = model_type
        list_models = os.listdir(
            os.path.join(
                params.model_dir, params.dataset.value, params.model_type.value
            )
        )
        for model_name in list_models:
            model_name = os.path.splitext(model_name)[0]
            params.model_name = model_name
            conf = create_params(params)
            for k in [10, 100, 1000]:
                conf.online_params.K = k
                conf.online_params.online_type = list(AlgorithmType)
                launcher.launch(params=conf, num_runs=args.num_runs)
