import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
    ):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, stride=12):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=stride * 2,
                stride=stride,
                padding=stride // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, stride=12):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=stride * 2,
            stride=stride,
            padding=stride // 2,
        )

    def forward(self, x):
        x = self.up(x)
        return x


class DilatedConv1d(nn.Module):
    def __init__(self, hidden_size=32, kernel_size=2, dilation=0):
        super(DilatedConv1d, self).__init__()

        self.conv_dilated = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            bias=False,
        )

    def forward(self, x):
        output = self.conv_dilated(x)
        return output


class ResiduaConvBlock(nn.Module):
    def __init__(self, hidden_size=32, kernel_size=2, dilation=0):
        super(ResiduaConvBlock, self).__init__()

        self.dilated_conv1d = DilatedConv1d(hidden_size, kernel_size, dilation)
        self.gate_tanh = nn.Tanh()
        self.gate_sigmoid = nn.Sigmoid()
        self.conv_res = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=1, padding="same"
        )
        self.conv_skip = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=1, padding="same"
        )

    def forward(self, x):
        output = self.dilated_conv1d(x)

        tanh = self.gate_tanh(output)
        sigmoid = self.gate_sigmoid(output)
        gated = tanh * sigmoid

        output = self.conv_res(gated)
        output = output + x
        skip = self.conv_skip(gated)

        return output, skip


class WaveNetBlock(nn.Module):
    def __init__(self, hidden_size=32, kernel_size=2, dilations=10):
        super(WaveNetBlock, self).__init__()

        self.residual_blocks = nn.ModuleList(
            [
                ResiduaConvBlock(hidden_size, kernel_size, 2**i)
                for i in range(0, dilations)
            ]
        )

    def forward(self, x):
        skip_connections = []
        for residual_block in self.residual_blocks:
            x, skip = residual_block(x)
            skip_connections.append(skip)
        return sum(skip_connections)

