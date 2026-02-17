import torch
import torch.nn as nn
import torch.functional as F



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        input_size = configs.window_size  # or configs.in_features depending on your data format
        out_features = configs.out_features
        
        self.network = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_features)
                )

    def forward(self, x):
        return self.network(x.permute(0,2,1)).squeeze(1)