# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from online_attacks.classifiers import load_classifier, DatasetType, MnistModel, CifarModel


def test_madry_model_mnist():
    load_classifier(DatasetType.MNIST, MnistModel.MADRY_MODEL, "natural", "/checkpoint/hberard/OnlineAttack/pretained_models/")
    load_classifier(DatasetType.MNIST, MnistModel.MADRY_MODEL, "adv_trained", "/checkpoint/hberard/OnlineAttack/pretained_models/")
    load_classifier(DatasetType.MNIST, MnistModel.MADRY_MODEL, "secret", "/checkpoint/hberard/OnlineAttack/pretained_models/")


def test_madry_model_cifar():
    load_classifier(DatasetType.CIFAR, CifarModel.MADRY_MODEL, "natural", "/checkpoint/hberard/OnlineAttack/pretained_models/")
    load_classifier(DatasetType.CIFAR, CifarModel.MADRY_MODEL, "adv_trained", "/checkpoint/hberard/OnlineAttack/pretained_models/")
    load_classifier(DatasetType.CIFAR, CifarModel.MADRY_MODEL, "secret", "/checkpoint/hberard/OnlineAttack/pretained_models/")


if __name__ == "__main__":
    test_madry_model_mnist()
    test_madry_model_cifar()
