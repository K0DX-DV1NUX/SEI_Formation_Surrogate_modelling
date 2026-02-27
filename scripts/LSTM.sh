#!/bin/sh

run_experiment() {
    seed=$3
    model="LSTM"
    mode=$1
    test_dir=$2

    python run_exp.py \
        --mode $mode \
        --model $model \
        --train_dir "Experiments/train" \
        --vali_dir "Experiments/vali" \
        --test_dir "Experiments/$test_dir" \
        --plots_dir "plots" \
        --checkpoints_dir "checkpoints" \
        --results_dir "results" \
        --epochs 50 \
        --patience 10 \
        --batch_size 512 \
        --learning_rate 1e-3 \
        --loss mse \
        --num_workers 0 \
        --window_size 50 \
        --stride 10 \
        --seed $seed \
        --test_model_folder "checkpoints/LSTM_$seed"
        #--test_standardize_path "checkpoints/GRU_43/std_values.json"
}

# Example usage:
run_experiment train test1 2021
run_experiment test test2 2021
run_experiment test test3 2021


run_experiment train test1 2022
run_experiment test test2 2022
run_experiment test test3 2022

run_experiment train test1 2023
run_experiment test test2 2023
run_experiment test test3 2023

run_experiment train test1 2024
run_experiment test test2 2024
run_experiment test test3 2024

run_experiment train test1 2025
run_experiment test test2 2025
run_experiment test test3 2025

# run_experiment train test2 2022
# run_experiment train test3 2023
# run_experiment train test4 2024
# run_experiment train test5 2025