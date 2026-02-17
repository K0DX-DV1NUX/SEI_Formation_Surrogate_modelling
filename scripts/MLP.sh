#!/bin/sh

model="MLP"
hidden_sizes="128 64"


python run_exp.py \
    --mode train \
    --model $model \
    --mlp_hidden_sizes $hidden_sizes \
    --train_dir "Experiments/train" \
    --vali_dir "Experiments/vali" \
    --test_dir "Experiments/test" \
    --plots_dir "plots" \
    --checkpoints_dir "checkpoints" \
    --results_dir "results" \
    --epochs 100 \
    --patience 20 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --loss mse \
    --seed 42
