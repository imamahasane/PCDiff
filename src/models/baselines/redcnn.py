from __future__ import annotations

import torch
import torch.nn as nn


class REDCNN(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 1, channels: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 5, padding=2)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv3 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv4 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv5 = nn.Conv2d(channels, channels, 5, padding=2)

        self.deconv1 = nn.ConvTranspose2d(channels, channels, 5, padding=2)
        self.deconv2 = nn.ConvTranspose2d(channels, channels, 5, padding=2)
        self.deconv3 = nn.ConvTranspose2d(channels, channels, 5, padding=2)
        self.deconv4 = nn.ConvTranspose2d(channels, channels, 5, padding=2)
        self.deconv5 = nn.ConvTranspose2d(channels, out_channels, 5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        shortcut_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        shortcut_4 = out
        out = self.relu(self.conv5(out))

        out = self.relu(self.deconv1(out) + shortcut_4)
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out) + shortcut_2)
        out = self.relu(self.deconv4(out))
        out = self.deconv5(out)

        return out + residual_input
