# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from online_attacks.classifiers import (
    AttackerConfig,
    ModelParams,
    CifarModel,
    MnistModel,
    eval_classifier,
    DatasetType,
)
from online_attacks.attacks import Attacker
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType
)
parser.add_argument("--source_model_name", default="0", type=str)
parser.add_argument("--target_model_name", default=None, type=str)
parser.add_argument(
    "--attacker", default=Attacker.NONE, type=Attacker, choices=Attacker
)
parser.add_argument("--batch_size", default=1000, type=int)
args, _ = parser.parse_known_args()

if args.dataset == DatasetType.MNIST:
    parser.add_argument(
        "--source_model_type",
        default=MnistModel.MODEL_A,
        type=MnistModel,
        choices=MnistModel,
    )
    parser.add_argument(
        "--target_model_type", default=None, type=MnistModel, choices=MnistModel
    )
elif args.dataset == DatasetType.CIFAR:
    parser.add_argument(
        "--source_model_type",
        default=CifarModel.VGG_16,
        type=CifarModel,
        choices=CifarModel,
    )
    parser.add_argument(
        "--target_model_type", default=None, type=CifarModel, choices=CifarModel
    )

args = parser.parse_args()

if args.target_model_name is None:
    args.target_model_name = args.source_model_name
if args.target_model_type is None:
    args.target_model_type = args.source_model_type

attacker = AttackerConfig(
    attacker=args.attacker,
    model=ModelParams(
        model_type=args.source_model_type, model_name=args.source_model_name
    ),
)

if args.dataset == DatasetType.MNIST:
    attacker.params.eps = 0.3
elif args.dataset == DatasetType.CIFAR:
    attacker.params.eps = 0.03125

model = ModelParams(
    model_type=args.target_model_type, model_name=args.target_model_name
)
eval_classifier(args.dataset, model, attacker, batch_size=args.batch_size)
