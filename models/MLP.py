import torch
import torch.nn as nn
import torch.functional as F



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        layers = []
        input_size = configs.window_size  # Number of timesteps in input sequence
        out_features = configs.out_features
        hidden_sizes = [128, 64]  # Default hidden layer sizes

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
            
        layers.append(nn.Linear(input_size, out_features))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.permute(0,2,1))