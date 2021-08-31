#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_optimistic --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 1.0  --namestr Exhaust-Var1-Toy-Stochastic-Optimistic
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_optimistic --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 5.0  --namestr Exhaust-Var5-Toy-Stochastic-Optimistic
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_optimistic --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 10.0  --namestr Exhaust-Var10-Toy-Stochastic-Optimistic

# Virtual
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 1.0  --namestr Exhaust-Var1-Toy-Stochastic-Virtual
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 5.0  --namestr Exhaust-Var5-Toy-Stochastic-Virtual
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 10.0  --namestr Exhaust-Var10-Toy-Stochastic-Virtual

# Virtual Plus
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_modified_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 1.0  --namestr Exhaust-Var1-Toy-Stochastic-Virtual-Plus
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_modified_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 5.0  --namestr Exhaust-Var5-Toy-Stochastic-Virtual-Plus
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_modified_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 10.0  --namestr Exhaust-Var10-Toy-Stochastic-Virtual-Plus

# Single-Ref
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_single_ref --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 1.0  --namestr Exhaust-Var1-Toy-Stochastic-Single-Ref
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_single_ref --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 5.0  --namestr Exhaust-Var5-Toy-Stochastic-Single-Ref
python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_single_ref --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 10.0  --namestr Exhaust-Var10-Toy-Stochastic-Single-Ref
