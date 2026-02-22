import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.input_size = configs.in_features                  # current only
        self.hidden_size = 64
        self.num_layers = 5
        self.out_features = configs.out_features
        self.dropout = 0.2

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.out_features),
            #nn.Tanh(),
            #nn.Linear(128, self.out_features)
        )

    def forward(self, x):
        """
        x: (batch, window_size, 1)
        """

        # GRU output:
        # out -> (batch, seq_len, hidden_size)
        # h_n -> (num_layers, batch, hidden_size)
        out, h_n = self.gru(x)

        # Use final hidden state of last layer
        final_hidden = h_n[-1]   # (batch, hidden_size)

        output = self.fc(final_hidden)

        return output