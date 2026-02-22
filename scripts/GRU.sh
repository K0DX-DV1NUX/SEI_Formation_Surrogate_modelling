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
    --epochs 50 \
    --patience 10 \
    --batch_size 512 \
    --learning_rate 1e-3 \
    --loss mse \
    --num_workers 0 \
    --window_size 100 \
    --stride 2 \
    --seed 43
        #--test_model_path "checkpoints/GRU_43/best_model_GRU.pt" \
    #--test_standardize_path "checkpoints/GRU_43/std_values.json" \
