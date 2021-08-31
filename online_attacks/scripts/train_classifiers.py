# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from omegaconf import OmegaConf
import copy
from typing import Union

# TODO: Fix the import to be more clean !
import sys
import os

path = os.path.realpath(os.path.join(os.getcwd(), ".."))
sys.path.append(path)
from online_attacks.classifiers.dataset import DatasetType
from online_attacks.launcher import Launcher
from online_attacks.attacks import Attacker
from online_attacks.classifiers.cifar.models import CifarModel
from online_attacks.classifiers.mnist.models import MnistModel


class TrainClassifier(Launcher):
    @classmethod
    def run(cls, args):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if args.dataset == DatasetType.MNIST:
            import online_attacks.classifiers.mnist as mnist

            params = OmegaConf.structured(mnist.TrainingParams)
            params.model_type = args.model_type
            params.train_on_test = args.train_on_test
            params.name = ""
            if args.robust:
                params.attacker = Attacker.PGD_ATTACK
                params.name = "%s_" % params.attacker.name
            params.name += "test_" if params.train_on_test else "train_"
            params.name += str(args.name)

            if args.model_attacker is not None:
                params.model_attacker = args.model_attacker
            mnist.train(params, device=device)

        elif args.dataset == DatasetType.CIFAR:
            import online_attacks.classifiers.cifar as cifar

            params = OmegaConf.structured(cifar.TrainingParams)
            params.model_type = args.model_type
            if params.model_type in [CifarModel.GOOGLENET, CifarModel.WIDE_RESNET]:
                params.dataset_params.batch_size = 64
                params.dataset_params.test_batch_size = 256
            elif params.model_type == CifarModel.DENSE_121:
                params.dataset_params.batch_size = 128

            params.train_on_test = args.train_on_test
            params.name = ""
            if args.robust:
                params.attacker = Attacker.PGD_ATTACK
                params.name = "%s_" % params.attacker.name
            params.name += "test_" if params.train_on_test else "train_"
            params.name += str(args.name)

            if args.model_attacker is not None:
                params.model_attacker = args.model_attacker
            cifar.train(params, device=device)

        else:
            raise ValueError()

    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--dataset",
            default=DatasetType.MNIST,
            type=DatasetType,
            choices=DatasetType,
        )

        # Hack to be able to parse either MnistModel or CifarModel
        args, _ = parser.parse_known_args()
        if args.dataset == DatasetType.MNIST:
            parser.add_argument(
                "--model_type",
                nargs="+",
                default=MnistModel.MODEL_A,
                type=MnistModel,
                choices=MnistModel,
            )
        elif args.dataset == DatasetType.CIFAR:
            parser.add_argument(
                "--model_type",
                nargs="+",
                default=CifarModel.VGG_16,
                type=CifarModel,
                choices=CifarModel,
            )

        parser.add_argument("--train_on_test", action="store_true")
        parser.add_argument("--num_models", default=1, type=int)
        parser.add_argument("--slurm", type=str, default="")
        parser.add_argument("--robust", action="store_true")
        parser.add_argument("--model_attacker", default=None, type=str)

        return parser


if __name__ == "__main__":
    launcher = TrainClassifier()
    parser = TrainClassifier.create_argument_parser()
    args = parser.parse_args()

    for model_type in args.model_type:
        for i in range(args.num_models):
            config = copy.deepcopy(args)
            config.model_type = model_type
            config.name = str(i)
            launcher.launch(config, slurm=config.slurm)
