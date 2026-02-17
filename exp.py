import sys
sys.path.append("..")

from loading_data.build_dataframes import BuildDataset



class Exp:

    def __init__(self):
        self.dataframes = BuildDataset("Experiments").get_dataframes()






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