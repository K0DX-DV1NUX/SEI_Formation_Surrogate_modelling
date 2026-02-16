import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BatteryMultiDataset(Dataset):
    def __init__(self, list_of_dfs, window_size=60, feature_col="Current [A]", target_cols=None):
        self.x_frames = []
        self.y_frames = []
        
        if target_cols is None:
            target_cols = ["Terminal voltage [V]", "Cell temperature [K]", "Negative SEI thickness [nm]"]

        for df in list_of_dfs:
            # Extract underlying numpy arrays
            inputs = df[feature_col].values.astype(np.float32)
            targets = df[target_cols].values.astype(np.float32)

            # Create sliding windows for this specific file
            # If length is 12000, it creates 11940 windows
            # If length is 70000, it creates 69940 windows
            for i in range(len(df) - window_size):
                self.x_frames.append(inputs[i : i + window_size])
                self.y_frames.append(targets[i + window_size])

        # Convert list to large numpy arrays for memory efficiency
        self.x_frames = np.array(self.x_frames)
        self.y_frames = np.array(self.y_frames)

    def __len__(self):
        return len(self.x_frames)

    def __getitem__(self, idx):
        # x: [window_size, 1] | y: [num_targets]
        x = torch.from_numpy(self.x_frames[idx]).unsqueeze(-1)
        y = torch.from_numpy(self.y_frames[idx])
        return x, y