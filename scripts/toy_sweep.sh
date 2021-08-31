#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-10 --online_params.threshold 10
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-15 --online_params.threshold 15
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-20 --online_params.threshold 20
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-25 --online_params.threshold 25
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-30 --online_params.threshold 30
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-35 --online_params.threshold 35
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-40 --online_params.threshold 40
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-45 --online_params.threshold 45
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-50 --online_params.threshold 50
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-55 --online_params.threshold 55
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-60 --online_params.threshold 60
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-65 --online_params.threshold 65
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-70 --online_params.threshold 70
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-75 --online_params.threshold 75
python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Optimistic-80 --online_params.threshold 80

# Virtual
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-10 --online_params.threshold 10
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-15 --online_params.threshold 15
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-20 --online_params.threshold 20
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-25 --online_params.threshold 25
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-30 --online_params.threshold 30
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-35 --online_params.threshold 35
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-40 --online_params.threshold 40
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-45 --online_params.threshold 45
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-50 --online_params.threshold 50
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-55 --online_params.threshold 55
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-60 --online_params.threshold 60
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-65 --online_params.threshold 65
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-70 --online_params.threshold 70
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-75 --online_params.threshold 75
python -m online_attacks.experiments.toy --online_params.online_type stochastic_virtual --wandb --online_params.N 100 --max_perms 10000 --K 20 --namestr Toy-Virtual-80 --online_params.threshold 80
