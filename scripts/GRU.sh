#!/bin/sh

model="GRU"

python run_exp.py \
    --mode train \
    --model $model \
    --train_dir "Experiments/train" \
    --vali_dir "Experiments/vali" \
    --test_dir "Experiments/test" \
    --plots_dir "plots" \
    --checkpoints_dir "checkpoints" \
    --results_dir "results" \
    --epochs 20 \
    --patience 5 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --loss mse \
    --num_workers 0 \
    --window_size 100 \
    --stride 10 \
    --seed 42
