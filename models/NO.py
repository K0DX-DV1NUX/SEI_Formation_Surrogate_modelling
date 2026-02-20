import torch
import torch.nn as nn
from neuralop.models import FNO


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.window_size = configs.window_size
        self.out_features = 3

        self.fno = FNO(
            n_modes=(16,),        # number of Fourier modes in time dimension
            hidden_channels=64,
            in_channels=2,        # current + voltage
            out_channels=3        # sei_r, li_r, temperature
        )

        # Since you want final prediction per window
        # we aggregate over time
        self.readout = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: (batch, window_size, 2)
        """

        # Convert to (batch, channels, window_size)
        x = x.permute(0, 2, 1)

        out = self.fno(x)   # (batch, 3, window_size)

        # Reduce over time dimension
        out = self.readout(out)  # (batch, 3, 1)

        out = out.squeeze(-1)    # (batch, 3)

        return out