import sys
sys.path.append("..")

from build.build_dataframes import BuildDataframes
from build.build_datasets import BatteryDataset
from build.standardization import Standardizer

from torch.utils.data import DataLoader



class Exp:

    def __init__(self):
        window_size= 30
        stride=1


        # Load raw data
        train_dataframes = BuildDataframes("Experiments/train").get_dataframes()
        test_dataframes  = BuildDataframes("Experiments/test").get_dataframes()
        vali_dataframes  = BuildDataframes("Experiments/vali").get_dataframes()

        # Standardize
        self.scaler = Standardizer()
        self.scaler.fit(self.train_dataframes)

        train_std = [self.scaler.transform(df) for df in self.train_dataframes]
        vali_std  = [self.scaler.transform(df) for df in self.vali_dataframes]
        test_std  = [self.scaler.transform(df) for df in self.test_dataframes]

        # Build datasets
        self.train_dataset = BatteryDataset(self.train_std, window_size, stride=stride)
        self.vali_dataset  = BatteryDataset(self.vali_std, window_size, stride=stride)
        self.test_dataset  = BatteryDataset(self.test_std, window_size, stride=1)

        self.train_loader = DataLoader(self.train_dataset, batch_size=128, drop_last=False, shuffle=True)


# window_size = 200

# train_dataset = BatterySequenceDataset(train_std, window_size)
# val_dataset   = BatterySequenceDataset(val_std, window_size)
# test_dataset  = BatterySequenceDataset(test_std, window_size)


# from torch.utils.data import DataLoader

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=256,
#     shuffle=True
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=256,
#     shuffle=True
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=256,
#     shuffle=False
# )