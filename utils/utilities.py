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
    Plot 4 subfigures (Temperature, Voltage, SEI Rate, Li Rate)
    in a single figure and save it.

    Parameters
    ----------
    preds : np.ndarray (N, 4)
    targets : np.ndarray (N, 4)
    plots_dir : str
    """

    #os.makedirs(plots_dir, exist_ok=True)

    preds = np.asarray(preds)
    targets = np.asarray(targets)

    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    feature_names = [
        "Temperature [K]",
        "Voltage [V]",
        "SEI Rate [nm/s]",
        "Lithium Capacity Rate [A.h/s]"
    ]

    fig, axes = plt.subplots(4, 1, figsize=(8, 5), sharex=True)

    for i in range(4):
        axes[i].plot(targets[:, i], label="True", linewidth=1)
        axes[i].plot(preds[:, i], label="Predicted", linewidth=1)

        axes[i].set_ylabel(feature_names[i])
        axes[i].grid(alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel("Timestep")

    fig.suptitle("Battery Surrogate Model – True vs Predicted", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(plots_dir, "prediction_results.png")
    plt.savefig(save_path, dpi=300)
    plt.close()