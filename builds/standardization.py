import numpy as np
import json

class Standardizer:

    def __init__(self):
        self.stats = {}

    def fit(self, dataframes):

        current = np.concatenate([df["Current [A]"].values for df in dataframes])
        temp    = np.concatenate([df["Cell temperature [K]"].values for df in dataframes])
        voltage = np.concatenate([df["Terminal voltage [V]"].values for df in dataframes])
        sei_r   = np.concatenate([df["SEI Rate"].values for df in dataframes])
        li_r    = np.concatenate([df["Lithium Capacity Rate"].values for df in dataframes])

        self.stats = {
            "current_mean": current.mean(),
            "current_std": current.std(),

            "temp_mean": temp.mean(),
            "temp_std": temp.std(),

            "voltage_mean": voltage.mean(),
            "voltage_std": voltage.std(),

            "sei_mean": sei_r.mean(),
            "sei_std": sei_r.std(),

            "li_mean": li_r.mean(),
            "li_std": li_r.std(),
        }

    def transform(self, df):

        df = df.copy()

        df["Current [A]"] = (
            (df["Current [A]"] - self.stats["current_mean"]) /
            (self.stats["current_std"] + 1e-8)
        )

        df["Cell temperature [K]"] = (
            (df["Cell temperature [K]"] - self.stats["temp_mean"]) /
            (self.stats["temp_std"] + 1e-8)
        )

        df["Terminal voltage [V]"] = (
            (df["Terminal voltage [V]"] - self.stats["voltage_mean"]) /
            (self.stats["voltage_std"] + 1e-8)
        )

        df["SEI Rate"] = (
            (df["SEI Rate"] - self.stats["sei_mean"]) /
            (self.stats["sei_std"] + 1e-8)
        )

        df["Lithium Capacity Rate"] = (
            (df["Lithium Capacity Rate"] - self.stats["li_mean"]) /
            (self.stats["li_std"] + 1e-8)
        )

        return df
    
    def inverse_transform_targets(self, y_pred):
        """
        y_pred shape: (batch, 4)
        Order:
        [temperature, voltage, sei_rate, lithium_rate]
        """

        y_pred = y_pred.copy()

        y_pred[:, 2] = (
            y_pred[:, 2] * self.stats["temp_std"]
            + self.stats["temp_mean"]
        )

        y_pred[:, 0] = (
            y_pred[:, 0] * self.stats["sei_std"]
            + self.stats["sei_mean"]
        )

        y_pred[:, 1] = (
            y_pred[:, 1] * self.stats["li_std"]
            + self.stats["li_mean"]
        )

        return y_pred
    
    # -------------------------
    # SAVE
    # -------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=4)

    # -------------------------
    # LOAD
    # -------------------------
    def load(self, path):
        with open(path, "r") as f:
            self.stats = json.load(f)