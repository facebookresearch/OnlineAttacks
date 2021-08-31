#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

NAME="madry-icml"
SLURM="configs/learnfair.yaml"
NUM_RUNS="1000"
K="1 3 4"

# MNIST
DATASET="mnist"
#python -m online_attacks.scripts.online_attacks_sweep --k $K --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type none --name $NAME/mnist/none --num_runs $NUM_RUNS --slurm $SLURM
python -m online_attacks.scripts.online_attacks_sweep --k $K --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type fgsm --name $NAME/mnist/fgsm --num_runs $NUM_RUNS --slurm $SLURM
python -m online_attacks.scripts.online_attacks_sweep --k $K --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type pgd --name $NAME/mnist/pgd --num_runs $NUM_RUNS --slurm $SLURM
#python -m online_attacks.scripts.online_attacks_sweep --k $K --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type cw --name $NAME/mnist/cw --num_runs $NUM_RUNS --slurm $SLURM

# MNIST
DATASET="cifar"
BATCH_SIZE="256"
#python -m online_attacks.scripts.online_attacks_sweep --k $K --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type none --name $NAME/cifar/none --num_runs $NUM_RUNS --slurm $SLURM --batch_size $BATCH_SIZE
python -m online_attacks.scripts.online_attacks_sweep --k $K --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type fgsm --name $NAME/cifar/fgsm --num_runs $NUM_RUNS --slurm $SLURM --batch_size $BATCH_SIZE
python -m online_attacks.scripts.online_attacks_sweep --k $K --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type pgd --name $NAME/cifar/pgd --num_runs $NUM_RUNS --slurm $SLURM --batch_size $BATCH_SIZE
#python -m online_attacks.scripts.online_attacks_sweep --k $K --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type cw --name $NAME/cifar/cw --num_runs $NUM_RUNS --slurm $SLURM --batch_size $BATCH_SIZE





