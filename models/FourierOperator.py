import torch
import torch.nn as nn
import torch.fft


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of low-frequency modes to keep

        # Complex weights for Fourier coefficients
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, modes, dtype=torch.cfloat
            )
        )

    def forward(self, x):
        """
        x: (batch, channels, length)
        """

        batchsize = x.shape[0]
        length = x.shape[-1]

        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)

        # Allocate output FFT tensor
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device
        )

        # Apply learned weights to low modes only
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix, iox -> box",
            x_ft[:, :, :self.modes],
            self.weights
        )

        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=length, dim=-1)

        return x
    

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.modes = 16
        self.width = 64  # hidden channels

        self.fc0 = nn.Linear(configs.in_features, self.width)

        self.conv_layers = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes)
            for _ in range(4)
        ])

        self.w_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=1)
            for _ in range(4)
        ])

        self.activation = nn.GELU()

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, configs.out_features)

    def forward(self, x):
        """
        x: (batch, window_size, in_features)
        """

        # Lift to higher dimension
        x = self.fc0(x)  # (batch, window, width)

        # Convert to (batch, channels, window)
        x = x.permute(0, 2, 1)

        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = self.activation(x1 + x2)

        # Global average pooling over time
        x = x.mean(dim=-1)

        # Final projection
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x