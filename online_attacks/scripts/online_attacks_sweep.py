# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import CrossEntropyLoss
from omegaconf import OmegaConf
import tqdm
import os
import submitit

from online_attacks.classifiers import (
    load_dataset,
    DatasetType,
    load_classifier,
    MnistModel,
    CifarModel,
)
from online_attacks.attacks import create_attacker, Attacker
from online_attacks.datastream import datastream
from online_attacks.online_algorithms import (
    create_algorithm,
    compute_indices,
    AlgorithmType,
)
from online_attacks.utils.logger import Logger
from online_attacks.utils import seed_everything
from online_attacks.launcher import Launcher
from online_attacks.scripts.online_attack_params import (
    OnlineAttackParams,
    CifarParams,
    MnistParams,
)


def config_exists(config):
    if os.path.exists(config.save_dir):
        list_paths = os.listdir(config.save_dir)
    else:
        return None

    for exp_id in list_paths:
        logger = Logger(save_dir=config.save_dir, exp_id=exp_id)
        other_config = logger.load_hparams()
        params = OmegaConf.structured(OnlineAttackParams)
        other_config = OmegaConf.merge(params, other_config)
        other_config = create_params(other_config)
        if config == other_config:
            print("Found existing config with id=%s" % logger.exp_id)
            return logger.exp_id
    return None


def create_params(params):
    if params.dataset == DatasetType.CIFAR:
        params = OmegaConf.structured(CifarParams(**params))
        params.attacker_params.eps = 0.03125
    elif params.dataset == DatasetType.MNIST:
        params = OmegaConf.structured(MnistParams(**params))
        params.attacker_params.eps = 0.3
    return params


class OnlineAttackExp:
    def __call__(
        self,
        params: OnlineAttackParams,
        num_runs: int = 1,
        max_num_runs: bool = True,
        overwrite=False,
    ):
        seed_everything(params.seed)

        exp_id = config_exists(params)
        logger = Logger(params.save_dir, exp_id=exp_id)
        if exp_id is None:
            logger.save_hparams(params)

        if overwrite:
            logger.reset()

        if max_num_runs:
            num_runs -= len(logger)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = load_dataset(params.dataset, train=False)
        permutation_gen = datastream.PermutationGenerator(
            len(dataset), seed=params.seed
        )

        source_classifier = load_classifier(
            params.dataset,
            params.model_type,
            name=params.model_name,
            model_dir=params.model_dir,
            device=device,
            eval=True,
        )
        attacker = create_attacker(
            source_classifier, params.attacker_type, params.attacker_params
        )

        transform = datastream.Compose(
            [
                datastream.ToDevice(device),
                datastream.AttackerTransform(attacker),
                datastream.ClassifierTransform(source_classifier),
                datastream.LossTransform(CrossEntropyLoss(reduction="none")),
            ]
        )

        algorithm = create_algorithm(
            params.online_params.online_type, params.online_params, N=len(dataset)
        )

        list_records = []
        for i in tqdm.tqdm(range(num_runs)):
            permutation = permutation_gen.sample()
            source_stream = datastream.BatchDataStream(
                dataset,
                batch_size=params.batch_size,
                transform=transform,
                permutation=permutation,
            )
            indices = compute_indices(source_stream, algorithm, pbar_flag=False)
            record = {"permutation": permutation.tolist(), "indices": indices}
            record_id = logger.save_record(record)
            list_records.append(record_id)

        return logger, list_records

    def checkpoint(
        self,
        params: OnlineAttackParams,
        num_runs: int = 1,
        max_num_runs: bool = True,
        overwrite=False,
    ) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            params, num_runs, max_num_runs, overwrite=False
        )  # submits to requeuing


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType
    )
    parser.add_argument("--model_name", default="train_0", type=str)
    parser.add_argument(
        "--attacker_type", default=Attacker.FGSM_ATTACK, type=Attacker, choices=Attacker
    )
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--name", default="default", type=str)

    # Hack to be able to parse either MnistModel or CifarModel
    args, _ = parser.parse_known_args()
    if args.dataset == DatasetType.MNIST:
        parser.add_argument(
            "--model_type",
            default=MnistModel.MODEL_A,
            type=MnistModel,
            choices=MnistModel,
        )
    elif args.dataset == DatasetType.CIFAR:
        parser.add_argument(
            "--model_type",
            default=CifarModel.VGG_16,
            type=CifarModel,
            choices=CifarModel,
        )
    args, _ = parser.parse_known_args()

    params = OmegaConf.structured(OnlineAttackParams(**vars(args)))
    params = create_params(params)

    # Hack to be able to parse num_runs without affecting params
    parser.add_argument("--num_runs", default=100, type=int)
    parser.add_argument("--slurm", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--k", nargs="+", default=[2, 5, 10, 100, 1000], type=int)
    args = parser.parse_args()

    launcher = Launcher(OnlineAttackExp(), args.slurm)

    for k in args.k:
        params.online_params.K = k
        params.online_params.online_type = list(AlgorithmType)
        launcher.launch(params, args.num_runs, overwrite=args.overwrite)

