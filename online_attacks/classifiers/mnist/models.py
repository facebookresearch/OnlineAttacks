# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
import torch
import os


class MnistModel(Enum):
    MODEL_A = "modelA"
    MODEL_B = "modelB"
    MODEL_C = "modelC"
    MODEL_D = "modelD"
    MADRY_MODEL = "madry"

    def __str__(self):
        return self.value


class modelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10

        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 20 * 20, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class modelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10

        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 64, 8)
        self.conv2 = nn.Conv2d(64, 128, 6)
        self.conv3 = nn.Conv2d(128, 128, 5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 12 * 12, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class modelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10

        self.conv1 = nn.Conv2d(1, 128, 3)
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class modelD(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10

        self.fc1 = nn.Linear(1 * 28 * 28, 300)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 300)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(300, 300)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(300, 300)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(300, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x


__mnist_model_dict__ = {
    MnistModel.MODEL_A: modelA,
    MnistModel.MODEL_B: modelB,
    MnistModel.MODEL_C: modelC,
    MnistModel.MODEL_D: modelD,
}


def make_mnist_model(model: MnistModel) -> nn.Module:
    return __mnist_model_dict__[model]()


def load_mnist_classifier(
    model_type: MnistModel,
    name: str = None,
    model_dir: str = None,
    device=None,
    eval=False,
) -> nn.Module:
    if model_type == MnistModel.MADRY_MODEL:
        from online_attacks.classifiers.madry import load_madry_model

        filename = os.path.join(model_dir, "mnist", model_type.value, "%s" % name)
        if os.path.exists(filename):
            model = load_madry_model("mnist", filename)
        else:
            raise OSError("File %s not found !" % filename)

        # Hack to be able to use some attacker class
        model.num_classes = 10

    elif model_type in __mnist_model_dict__:
        model = make_mnist_model(model_type)
        if name is not None:
            filename = os.path.join(
                model_dir, "mnist", model_type.value, "%s.pth" % name
            )
            if os.path.exists(filename):
                state_dict = torch.load(filename, map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
            else:
                raise OSError("File %s not found !" % filename)

    else:
        raise ValueError()

    if eval:
        model.eval()

    return model.to(device)
