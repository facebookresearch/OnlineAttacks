from enum import Enum
import torch
import torch.nn as nn
import os

from .vgg import VGG
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .resnet import ResNet18
from .densenet import densenet_cifar
from .googlenet import GoogLeNet
from .wide_resnet import Wide_ResNet


class CifarModel(Enum):
    VGG_16 = "VGG16"
    RESNET_18 = "res18"
    DENSE_121 = "dense121"
    GOOGLENET = "googlenet"
    WIDE_RESNET = "wide_resnet"
    MADRY_MODEL = "madry"

    def __str__(self):
        return self.value


__cifar_model_dict__ = {
    CifarModel.VGG_16: VGG,
    CifarModel.RESNET_18: ResNet18,
    CifarModel.DENSE_121: densenet_cifar,
    CifarModel.GOOGLENET: GoogLeNet,
    CifarModel.WIDE_RESNET: Wide_ResNet,
}


def make_cifar_model(model: CifarModel) -> nn.Module:
    return __cifar_model_dict__[model]()


def load_cifar_classifier(
    model_type: CifarModel,
    name: str = None,
    model_dir: str = None,
    device=None,
    eval=False,
) -> nn.Module:
    if model_type == CifarModel.MADRY_MODEL:
        from online_attacks.classifiers.madry import load_madry_model

        filename = os.path.join(model_dir, "cifar", model_type.value, "%s" % name)
        if os.path.exists(filename):
            model = load_madry_model("cifar", filename)
        else:
            raise OSError("File %s not found !" % filename)

    elif model_type in __cifar_model_dict__:
        model = make_cifar_model(model_type)
        if name is not None:
            filename = os.path.join(
                model_dir, "cifar", model_type.value, "%s.pth" % name
            )
            if os.path.exists(filename):
                state_dict = torch.load(filename, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
            else:
                raise OSError("File not found !")

    else:
        raise ValueError()

    if eval:
        model.eval()

    # Hack to be able to use some attacker class
    model.num_classes = 10

    return model.to(device)
