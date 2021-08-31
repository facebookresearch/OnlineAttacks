# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from online_attacks.classifiers import (
    CifarModel,
    MnistModel,
    DatasetType,
    load_classifier,
    load_dataset,
)
from online_attacks.attacks import Attacker, AttackerParams, create_attacker
from online_attacks import datastream
import argparse
import torch
from torch.nn import CrossEntropyLoss
import tqdm
from omegaconf import OmegaConf
import os
import uuid
import json


class Logger:
    def __init__(self, path, config_id=None):
        self.config_id = config_id
        if self.config_id is None:
            self.config_id = str(uuid.uuid4())
        self.path = os.path.join(path, self.config_id)

    def save_config(self, config):
        os.makedirs(self.path, exist_ok=True)
        filename = os.path.join(self.path, "config.yaml")
        OmegaConf.save(config=config, f=filename)

    def save_hist(self, hist):
        os.makedirs(self.path, exist_ok=True)
        filename = os.path.join(self.path, "hist.json")
        with open(filename, "w+") as f:
            json.dump(hist, f)
            f.flush()

    def load_hist(self):
        filename = os.path.join(self.path, "hist.json")
        with open(filename, "r") as f:
            hist = json.load(f)
        return hist

    def load_config(self):
        filename = os.path.join(self.path, "config.yaml")
        return OmegaConf.load(filename)

    @staticmethod
    def list_logger(path):
        list_logger = []
        for config_id in os.listdir(path):
            list_logger.append(Logger(path, config_id))
        return list_logger

    @staticmethod
    def check_config_exists(path, config):
        config = OmegaConf.to_yaml(config)
        if os.path.exists(path):
            list_config = os.listdir(path)
        else:
            return False

        for config_id in list_config:
            logger = Logger(path, config_id=config_id)
            other_config = logger.load_config()
            other_config = OmegaConf.to_yaml(other_config)
            if config == other_config:
                print("Found existing config with id=%s" % logger.config_id)
                return True
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    parser.add_argument(
        "--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType
    )
    parser.add_argument("--source_model_name", default="train_0", type=str)
    parser.add_argument("--target_model_name", default=None, type=str)
    parser.add_argument(
        "--attacker", default=Attacker.NONE, type=Attacker, choices=Attacker
    )
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument(
        "--model_dir",
        default="/checkpoint/hberard/OnlineAttack/pretained_models/",
        type=str,
    )
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

    print(args.target_model_type, args.source_model_type)

    if args.target_model_name is None:
        args.target_model_name = args.source_model_name
    if args.target_model_type is None:
        args.target_model_type = args.source_model_type

    attacker_params = OmegaConf.structured(AttackerParams)
    if args.dataset == DatasetType.MNIST:
        attacker_params.eps = 0.3
    elif args.dataset == DatasetType.CIFAR:
        attacker_params.eps = 0.03125

    config = OmegaConf.create(vars(args))
    config.attacker_params = attacker_params

    if Logger.check_config_exists(args.output_path, config):
        exit()

    logger = Logger(args.output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_classifier = load_classifier(
        args.dataset,
        args.target_model_type,
        name=args.target_model_name,
        model_dir=args.model_dir,
        device=device,
        eval=True,
    )
    source_classifier = load_classifier(
        args.dataset,
        args.source_model_type,
        name=args.source_model_name,
        model_dir=args.model_dir,
        device=device,
        eval=True,
    )
    attacker = create_attacker(source_classifier, args.attacker, attacker_params)

    transform = datastream.Compose(
        [datastream.ToDevice(device), datastream.AttackerTransform(attacker)]
    )

    dataset = load_dataset(args.dataset, train=False)
    datastream = datastream.BatchDataStream(
        dataset, batch_size=args.batch_size, transform=transform, return_target=True
    )

    loss = CrossEntropyLoss(reduction="none")

    hist = {
        "source": {"loss": [], "correct": []},
        "target": {"loss": [], "correct": []},
    }
    for x, target in tqdm.tqdm(datastream):
        target_x = target_classifier(x)
        pred = target_x.max(1, keepdim=True)[1]
        hist["target"]["correct"] += pred.eq(target.view_as(pred)).tolist()
        hist["target"]["loss"] += loss(target_x, target).tolist()

        source_x = source_classifier(x)
        pred = source_x.max(1, keepdim=True)[1]
        hist["source"]["correct"] += pred.eq(target.view_as(pred)).tolist()
        hist["source"]["loss"] += loss(source_x, target).tolist()

    logger.save_config(config)
    logger.save_hist(hist)
