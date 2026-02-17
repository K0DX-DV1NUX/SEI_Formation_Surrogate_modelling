#!/usr/bin/env python
# run_exp.py

import argparse
import torch
from exp import Exp, BuildDataframes, Standardizer, BatteryDataset, plot_predictions

def main(args):

    # ---------------------------
    # Build configs object
    # ---------------------------
    class Config:
        pass

    configs = Config()
    configs.epochs = args.epochs
    configs.batch_size = args.batch_size
    configs.learning_rate = args.learning_rate
    configs.loss = args.loss
    configs.model = args.model
    configs.window_size = args.window_size
    configs.stride = args.stride
    configs.patience = args.patience
    configs.device = args.device

    # ---------------------------
    # Initialize experiment
    # ---------------------------
    exp = Exp(configs)

    # ---------------------------
    # Run training or testing
    # ---------------------------
    if args.mode == "train":
        print("Starting training...")
        exp.train()

        # Evaluate on test set after training
        preds, targets = exp.test()
        plot_predictions(preds, targets, plots_dir="plots")

    elif args.mode == "test":
        print("Running test...")
        if args.checkpoint is None:
            print("No checkpoint provided. Using best model from training.")
            preds, targets = exp.test()
        else:
            preds, targets = exp.test(checkpoint_path=args.checkpoint)

        plot_predictions(preds, targets, plots_dir="plots")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Battery Surrogate Experiment Runner")

    # ---------------------------
    # Mode
    # ---------------------------
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test"],
                        help="Run mode: train or test")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load for testing")

    # ---------------------------
    # Hyperparameters
    # ---------------------------
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae"], help="Loss function")
    parser.add_argument("--model", type=str, default="GRU", choices=["GRU", "LSTM", "MLP"], help="Model type")
    parser.add_argument("--window_size", type=int, default=30, help="LSTM/GRU window size")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sliding window")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu")

    args = parser.parse_args()
    main(args)
