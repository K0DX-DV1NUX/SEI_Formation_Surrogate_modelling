import torch
from torch.utils.data import Dataset
import numpy as np


class BatteryDataset(Dataset):
    def __init__(self, dataframes, window_size=30, stride=1):
        """
        dataframes: list of standardized pandas DataFrames
        window_size: length of input sequence
        stride: step between consecutive windows
        """

        self.window_size = window_size
        self.stride = stride

        self.inputs = []
        self.targets = []

        for df in dataframes:

            current = df["Current [A]"].values
            temp    = df["Cell temperature [K]"].values
            voltage = df["Terminal voltage [V]"].values
            sei_r   = df["SEI Rate"].values
            li_r    = df["Lithium Capacity Rate"].values

            seq_len = len(df)

            for start in range(0, seq_len - window_size + 1, stride):

                end = start + window_size

                # ---- Input sequence (current only) ----
                input_seq = current[start:end].reshape(-1, 1)

                # ---- Target at final timestep ----
                target = np.array([
                    temp[end - 1],
                    voltage[end - 1],
                    sei_r[end - 1],
                    li_r[end - 1]
                ], dtype=np.float32)

                self.inputs.append(input_seq.astype(np.float32))
                self.targets.append(target)

        # Convert once to tensors (faster at runtime)
        self.inputs = torch.tensor(np.array(self.inputs))
        self.targets = torch.tensor(np.array(self.targets))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]