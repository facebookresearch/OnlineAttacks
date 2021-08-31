# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from online_attacks.utils.logger import Logger
from online_attacks.classifiers import (
    load_dataset,
    load_classifier,
    DatasetType,
    MnistModel,
    CifarModel,
)
from online_attacks.datastream import datastream
from online_attacks.attacks import create_attacker, compute_attack_success_rate
from online_attacks.online_algorithms import (
    AlgorithmType,
    create_algorithm,
    compute_indices,
    compute_competitive_ratio,
)
from online_attacks.scripts.online_attacks_sweep import create_params
from online_attacks.scripts.online_attack_params import OnlineAttackParams
from online_attacks.launcher import Launcher, SlurmLauncher
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from collections import defaultdict
import argparse
import os
import torch
import tqdm


def eval_comp_ratio(logger, model_type, model_name, list_records=None):
    dir_name = os.path.join(model_type.value, model_name)

    params = logger.load_hparams()
    params = OmegaConf.structured(OnlineAttackParams(**params))
    params = create_params(params)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset(params.dataset, train=False)
    target_classifier = load_classifier(
        params.dataset,
        model_type,
        name=model_name,
        model_dir=params.model_dir,
        device=device,
        eval=True,
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
            datastream.ClassifierTransform(target_classifier),
        ]
    )

    target_transform = datastream.Compose(
        [
            datastream.ToDevice(device),
            datastream.AttackerTransform(attacker),
            datastream.ClassifierTransform(target_classifier),
            datastream.LossTransform(CrossEntropyLoss(reduction="none")),
        ]
    )
    algorithm = create_algorithm(
        AlgorithmType.OFFLINE, params.online_params, N=len(dataset)
    )
    target_stream = datastream.BatchDataStream(
        dataset, batch_size=params.batch_size, transform=target_transform
    )
    offline_indices = compute_indices(target_stream, algorithm, pbar_flag=False)[
        algorithm[0].name
    ]

    target_stream = datastream.BatchDataStream(
        dataset, batch_size=params.batch_size, transform=target_transform
    )
    offline_indices = compute_indices(target_stream, algorithm, pbar_flag=False)[
        algorithm[0].name
    ]

    indices = [x[1] for x in offline_indices]
    target_stream = datastream.BatchDataStream(
        dataset, batch_size=params.batch_size, transform=transform
    )
    stream = target_stream.subset(indices)
    offline_fool_rate, knapsack_offline = compute_attack_success_rate(
        stream, CrossEntropyLoss(reduction="sum")
    )

    if list_records is None:
        list_records = logger.list_all_records()
    for record_name in tqdm.tqdm(list_records):
        if logger.check_eval_results_exist(dir_name, record_name):
            # print("Ignoring %s/%s, already exists."%(dir_name, record_name))
            continue
        record = logger.load_record(record_name)
        permutation = record["permutation"]
        eval_results = {}
        eval_results["metrics"] = {
            name: defaultdict(list) for name in record["indices"]
        }
        eval_results["offline_fool_rate"] = offline_fool_rate
        for name, indices in record["indices"].items():
            permuted_indices = [(x, permutation[index]) for x, index in indices]
            comp_ratio = compute_competitive_ratio(
                permuted_indices, offline_indices
            ) / len(offline_indices)

            indices = [x[1] for x in indices]
            target_stream = datastream.BatchDataStream(
                dataset,
                batch_size=params.batch_size,
                transform=transform,
                permutation=permutation,
            )
            stream = target_stream.subset(indices)
            fool_rate, knapsack = compute_attack_success_rate(
                stream, CrossEntropyLoss(reduction="sum")
            )

            knapsack_ratio = knapsack / knapsack_offline

            eval_results["metrics"][name]["num_indices"].append(len(indices))
            eval_results["metrics"][name]["fool_rate"].append(fool_rate)
            eval_results["metrics"][name]["knapsack_ratio"].append(knapsack_ratio)
            eval_results["metrics"][name]["comp_ratio"].append(comp_ratio)

        logger.save_eval_results(eval_results, dir_name, record_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument(
        "--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType
    )
    args, _ = parser.parse_known_args()

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
    parser.add_argument("--model_name", nargs="+", default=["all"], type=str)
    args = parser.parse_args()

    launcher = Launcher(
        SlurmLauncher(eval_comp_ratio, checkpointing=True), slurm=args.slurm
    )
    list_exp_id = Logger.list_all_logger(args.path)
    for model_type in args.model_type:
        model_name_list = args.model_name
        if "all" in args.model_name:
            list_files = os.listdir(
                os.path.join(
                    OnlineAttackParams().model_dir, args.dataset.value, model_type.value
                )
            )
            model_name_list = [os.path.splitext(filename)[0] for filename in list_files]
        for model_name in model_name_list:
            dir_name = os.path.join(model_type.value, model_name)
            print("Evaluating on %s" % dir_name)
            for exp_id in list_exp_id:
                logger = Logger(args.path, exp_id)
                if logger.check_eval_done(dir_name):
                    continue
                else:
                    launcher.launch(logger, model_type, model_name)
