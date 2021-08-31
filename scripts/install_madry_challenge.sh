#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

MODELS_PATH=$1
CURRENT_PATH=$(pwd)
MADRY_PATH=online_attacks/classifiers/madry

### INSTALLING MADRY MNIST CHALLENGE ###
git clone https://github.com/MadryLab/mnist_challenge $MADRY_PATH/madry_mnist
mkdir /tmp/madry_mnist
cd /tmp/madry_mnist
python -m online_attacks.classifiers.madry.madry_mnist.fetch_model secret
python -m online_attacks.classifiers.madry.madry_mnist.fetch_model natural
python -m online_attacks.classifiers.madry.madry_mnist.fetch_model adv_trained
mkdir $MODELS_PATH/mnist/madry
mv models/* $MODELS_PATH/mnist/madry
cd $CURRENT_PATH
rm -rf /tmp/madry_mnist

### INSTALLING MADRY CIFAR CHALLENGE ###
git clone https://github.com/MadryLab/cifar10_challenge $MADRY_PATH/madry_cifar
mkdir /tmp/madry_cifar
cd /tmp/madry_cifar
python online_attacks.classifiers.madry.madry_cifar.fetch_model secret
python online_attacks.classifiers.madry.madry_cifar.fetch_model natural
python online_attacks.classifiers.madry.madry_cifar.fetch_model adv_trained

mv models/model_0 models/secret
mv models/naturally_trained models/natural

mkdir $MODELS_PATH/cifar/madry
mv models/* $MODELS_PATH/cifar/madry
cd $CURRENT_PATH
rm -rf /tmp/madry_cifar
