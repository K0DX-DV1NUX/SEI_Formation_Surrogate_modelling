import torch
from torch.utils.data import Dataset
import numpy as np


class BatteryDataset(Dataset):
    def __init__(self, dataframes, window_size=30, stride=1):
        """
        dataframes: list of standardized pandas DataFrames
        window_size: LSTM window length
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
        temp    = df["Cell temperature [K]"].values
        voltage = df["Terminal voltage [V]"].values
        sei_r   = df["SEI Rate"].values
        li_r    = df["Lithium Capacity Rate"].values

        input_seq = current[start:end+1].reshape(-1, 1)

        target = np.array([
            temp[end],
            voltage[end],
            sei_r[end],
            li_r[end]
        ], dtype=np.float32)

        return torch.tensor(input_seq, dtype=torch.float32), \
               torch.tensor(target, dtype=torch.float32)