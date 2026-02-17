import torch
from torch.utils.data import Dataset
import numpy as np


class BatteryDataset(Dataset):
    def __init__(self, dataframes, window_size=30):
        """
        dataframes: list of standardized pandas DataFrames
        window_size: LSTM window length
        """

        self.dataframes = dataframes
        self.window_size = window_size

        # Build global index mapping
        # Each sample = (df_id, time_index)
        self.index_map = []

        for df_id, df in enumerate(self.dataframes):
            seq_len = len(df)
            for t in range(seq_len):
                self.index_map.append((df_id, t))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        df_id, t = self.index_map[idx]
        df = self.dataframes[df_id]

        current = df["Current [A]"].values
        temp    = df["Cell temperature [K]"].values
        voltage = df["Terminal voltage [V]"].values
        sei_r   = df["SEI Rate"].values
        li_r    = df["Lithium Capacity Rate"].values

        start = max(0, t - self.window_size + 1)

        input_current = current[start:t+1]

        # reshape to (seq_len, 1)
        input_seq = input_current.reshape(-1, 1)

        # left padding if needed
        if len(input_seq) < self.window_size:
            pad_len = self.window_size - len(input_seq)
            pad = np.zeros((pad_len, 1))
            input_seq = np.vstack([pad, input_seq])

        target = np.array([
            temp[t],
            voltage[t],
            sei_r[t],
            li_r[t]
        ], dtype=np.float32)

        return torch.tensor(input_seq, dtype=torch.float32), \
               torch.tensor(target, dtype=torch.float32)



