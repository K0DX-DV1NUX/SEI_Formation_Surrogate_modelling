import os

def check_and_prepare_dirs(args):

    # ----------------------------
    # Required input directories
    # ----------------------------
    required_dirs = {
        "train_dir": args.train_dir,
        "vali_dir": args.vali_dir,
        "test_dir": args.test_dir,
    }

    for name, path in required_dirs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required directory '{name}' not found at: {path}"
            )

    # ----------------------------
    # Output directories (create if missing)
    # ----------------------------
    output_dirs = {
        "plots_dir": args.plots_dir,
        "checkpoints_dir": args.checkpoints_dir,
        "results_dir": args.results_dir,
    }

    for name, path in output_dirs.items():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")


import os
import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(preds, targets, plots_dir="plots"):
    """
    Create 4 separate plots:
    - Temperature
    - Voltage
    - SEI Rate
    - Lithium Capacity Rate

    Each saved as an individual PNG.
    """

    #os.makedirs(plots_dir, exist_ok=True)

    feature_names = [
        "Temperature [K]",
        "Voltage [V]",
        "SEI Rate [nm/s]",
        "Lithium Capacity Rate [A.h/s]"
    ]

    file_names = [
        "temperature.png",
        "voltage.png",
        "sei_rate.png",
        "lithium_capacity_rate.png"
    ]

    for i in range(4):

        plt.figure(figsize=(8, 4))

        plt.plot(targets[:, i], label="True", linewidth=1)
        plt.plot(preds[:, i], label="Predicted", linewidth=1)

        plt.xlabel("Timestep")
        plt.ylabel(feature_names[i])
        plt.title(f"{feature_names[i]} – True vs Predicted")

        plt.grid(alpha=0.3)
        plt.legend()

        save_path = os.path.join(plots_dir, file_names[i])
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
