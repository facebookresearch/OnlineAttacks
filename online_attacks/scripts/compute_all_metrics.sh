#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



python -m online_attacks.scripts.eval_all /checkpoint/hberard/OnlineAttack/results_icml/madry-icml/  --dataset mnist --model_type madry --model_name secret --slurm configs/learnfair.yaml
python -m online_attacks.scripts.eval_all /checkpoint/hberard/OnlineAttack/results_icml/madry-icml/  --dataset cifar --model_type madry --model_name secret --slurm configs/learnfair.yaml
