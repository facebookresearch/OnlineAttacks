# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
import tqdm

from online_attacks.classifiers import (
    DatasetType,
    MnistModel,
    CifarModel,
    load_classifier,
    load_dataset,
)
from online_attacks.launcher import Launcher, SlurmLauncher
from online_attacks.utils.logger import Logger
from online_attacks.scripts.online_attack_params import OnlineAttackParams
from online_attacks.scripts.online_attacks_sweep import create_params
from online_attacks.attacks import create_attacker
from online_attacks import datastream


def compute_loss_values(datastream):
    list_loss = []
    for x, target in datastream:
        list_loss += x.tolist()
    return list_loss


def compute_hist(logger, model_type, model_name, list_records=None):
    dir_name = os.path.join(model_type, model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = logger.load_hparams()
    params = OmegaConf.structured(OnlineAttackParams(**params))
    params = create_params(params)

    if params.dataset == DatasetType.MNIST:
        model_type = MnistModel(model_type)
    elif params.dataset == DatasetType.CIFAR:
        model_type = CifarModel(model_type)

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

    target_transform = datastream.Compose(
        [
            datastream.ToDevice(device),
            datastream.AttackerTransform(attacker),
            datastream.ClassifierTransform(target_classifier),
            datastream.LossTransform(CrossEntropyLoss(reduction="none")),
        ]
    )

    if list_records is None:
        list_records = logger.list_all_records()
    for record_name in tqdm.tqdm(list_records):
        if logger.check_hist_exist(dir_name, record_name):
            # print("Ignoring %s/%s, already exists."%(dir_name, record_name))
            continue
        record = logger.load_record(record_name)
        permutation = record["permutation"]
        eval_results = {}
        for name, indices in record["indices"].items():
            indices = [x[1] for x in indices]
            target_stream = datastream.BatchDataStream(
                dataset,
                batch_size=params.batch_size,
                transform=target_transform,
                permutation=permutation,
            )
            stream = target_stream.subset(indices)
            loss_values = compute_loss_values(stream)
            eval_results[name] = loss_values
        logger.save_hist(eval_results, dir_name, record_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument(
        "--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType
    )
    parser.add_argument("--random_eval", action="store_true")
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
        SlurmLauncher(compute_hist, checkpointing=True), slurm=args.slurm
    )
    list_exp_id = Logger.list_all_logger(args.path)
    for exp_id in list_exp_id:
        logger = Logger(args.path, exp_id)
        if args.random_eval:
            args.model_type = os.listdir(os.path.join(logger.path, "eval"))
        for model_type in args.model_type:
            if args.random_eval:
                model_name_list = os.listdir(
                    os.path.join(logger.path, "eval", model_type)
                )
            else:
                model_type = model_type.value
                model_name_list = args.model_name
                if "all" in args.model_name:
                    list_files = os.listdir(
                        os.path.join(
                            OnlineAttackParams().model_dir,
                            args.dataset.value,
                            model_type,
                        )
                    )
                    model_name_list = [
                        os.path.splitext(filename)[0] for filename in list_files
                    ]
            for model_name in model_name_list:
                dir_name = os.path.join(model_type, model_name)
                if logger.check_hist_done(dir_name):
                    continue
                else:
                    launcher.launch(logger, model_type, model_name)
