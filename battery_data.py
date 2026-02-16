import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BatterySequenceDataset(Dataset):
    def __init__(self, datasets, window_size=100):
        self.window_size = window_size
        self.samples = []

        for data in datasets:

            current = data["current"]
            voltage = data["voltage"]
            rate = data["sei_rate"]

            seq_len = len(current)

            for t in range(seq_len):

                start = max(0, t - window_size + 1)
                input_seq = current[start:t+1]

                # left padding
                if len(input_seq) < window_size:
                    pad_len = window_size - len(input_seq)
                    input_seq = np.pad(
                        input_seq,
                        (pad_len, 0),
                        mode='constant'
                    )

                target = np.array([
                    voltage[t],
                    rate[t]
                ])

                self.samples.append(
                    (input_seq.astype(np.float32),
                     target.astype(np.float32))
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]

        return (
            torch.tensor(x).unsqueeze(-1),  # (W,1)
            torch.tensor(y)                # (2,)
        )
    
    

class SequenceStandardizer:
    def __init__(self):
        self.stats = {}

    def fit(self, datasets):

        all_current = np.concatenate([d["current"] for d in datasets])
        all_voltage = np.concatenate([d["voltage"] for d in datasets])
        all_rate = np.concatenate([d["sei_rate"] for d in datasets])

        self.stats["current_mean"] = all_current.mean()
        self.stats["current_std"] = all_current.std()

        self.stats["voltage_mean"] = all_voltage.mean()
        self.stats["voltage_std"] = all_voltage.std()

        self.stats["rate_mean"] = all_rate.mean()
        self.stats["rate_std"] = all_rate.std()

    def transform(self, dataset):
        dataset = dataset.copy()

        dataset["current"] = (
            (dataset["current"] - self.stats["current_mean"]) /
            (self.stats["current_std"] + 1e-8)
        )

        dataset["voltage"] = (
            (dataset["voltage"] - self.stats["voltage_mean"]) /
            (self.stats["voltage_std"] + 1e-8)
        )

        dataset["sei_rate"] = (
            (dataset["sei_rate"] - self.stats["rate_mean"]) /
            (self.stats["rate_std"] + 1e-8)
        )

        return dataset