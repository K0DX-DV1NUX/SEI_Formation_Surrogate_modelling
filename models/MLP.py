import torch
import torch.nn as nn
import torch.functional as F



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        input_size = configs.window_size * configs.in_features # or configs.in_features depending on your data format
        out_features = configs.out_features
        
        self.network = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.Tanh(),
                    nn.Linear(256, 128),
                    nn.Tanh(),
                    nn.Linear(128, out_features)
                )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.network(x).squeeze(1)