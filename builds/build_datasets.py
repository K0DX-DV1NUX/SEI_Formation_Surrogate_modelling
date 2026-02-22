import torch
from torch.utils.data import Dataset
import numpy as np


class BatteryDataset(Dataset):
    def __init__(self, dataframes, window_size=30, stride=1):
        """
        dataframes: list of standardized pandas DataFrames
        window_size: sequence length
        stride: step size between windows
        """

        self.dataframes = dataframes
        self.window_size = window_size
        self.stride = stride

        self.samples = []

        for df_id, df in enumerate(dataframes):
            seq_len = len(df)

            for start in range(0, seq_len - window_size + 1, stride):
                end = start + window_size - 1
                self.samples.append((df_id, start, end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        df_id, start, end = self.samples[idx]
        df = self.dataframes[df_id]

        current = df["Current [A]"].values
        voltage = df["Terminal voltage [V]"].values
        temp    = df["Cell temperature [K]"].values
        sei_r   = df["SEI Rate"].values
        q_cum   = df["Q_cum"].values

        # ---- INPUT: current + voltage ----
        input_seq = np.stack([
            current[start:end+1],
            voltage[start:end+1],
            q_cum[start:end+1],
        ], axis=1).astype(np.float32)   # shape: (window_size, 2)

        # ---- TARGET: sei_r, li_r, temperature ----
        target = np.array([
            sei_r[end],
            temp[end]
        ], dtype=np.float32)

        return (
            torch.tensor(input_seq, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )