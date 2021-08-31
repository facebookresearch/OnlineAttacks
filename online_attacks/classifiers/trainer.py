# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import Module
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch
from online_attacks.utils.optimizer import Sls
from online_attacks.attacks import NoAttacker
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import Attack
from typing import Optional


class Trainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Module = nn.CrossEntropyLoss(),
        attacker: Optional[Attack] = None,
        device=None,
        logger=None,
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.attacker = attacker
        self.logger = logger

        self.device = device
        self.model.to(device)

    def train(self, epoch: int):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            if not isinstance(self.attacker, NoAttacker):
                with ctx_noparamgrad_and_eval(self.attacker.predict):
                    data = self.attacker.perturb(data, target)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            if isinstance(self.optimizer, Sls):

                def closure():
                    output = self.model(data)
                    loss = self.criterion(output, target).mean()
                    return loss

                self.optimizer.step(closure)
            else:
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if batch_idx % 10 == 0:
                print(
                    f"Train Epoch: {epoch:d} [{batch_idx * len(data):d}/{len(self.train_loader.dataset):d} "
                    f"{100. * batch_idx / len(self.train_loader):.0f}] \tLoss: {loss.item():.6f} | "
                    f"Acc: {100. * correct / total:.3f}"
                )

        if self.logger is not None:
            self.logger.write(
                dict(train_accuracy=100.0 * correct / total, loss=loss.item()), epoch
            )

    def test(self, epoch: int) -> float:
        self.model.eval()
        if not isinstance(self.attacker, NoAttacker):
            self.attacker.predict.eval()
        test_loss = 0
        correct = 0
        adv_correct = 0
        for data, target in self.test_loader:
            with torch.no_grad():
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                # sum up batch loss
                test_loss += loss.item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            if not isinstance(self.attacker, NoAttacker):
                data = self.attacker.perturb(data, target)
                with torch.no_grad():
                    output = self.model(data)
                    pred = output.max(1, keepdim=True)[1]
                    adv_correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        acc = 100.0 * correct / len(self.test_loader.dataset)
        adv_acc = 100.0 * adv_correct / len(self.test_loader.dataset)

        if self.logger is None:
            log_output = "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss, correct, len(self.test_loader.dataset), acc
            )
            if self.attacker is not None:
                log_output += ", Adv Accuracy: {}/{} ({:.0f}".format(
                    adv_correct, len(self.test_loader.dataset), adv_acc
                )
            print(log_output)
        else:
            results = dict(test_loss=test_loss, test_accuracy=acc)
            if self.attacker is not None:
                results["adv_accuracy"] = adv_acc
            self.logger.write(results, epoch)

        return acc
