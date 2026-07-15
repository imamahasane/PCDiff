from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DUGANGenerator(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.enc1 = _ConvBlock(in_channels, c)
        self.enc2 = _ConvBlock(c, c * 2)
        self.enc3 = _ConvBlock(c * 2, c * 4)
        self.enc4 = _ConvBlock(c * 4, c * 8)
        self.bottleneck = _ConvBlock(c * 8, c * 16)

        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = _ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = _ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = _ConvBlock(c * 2, c)

        self.out_conv = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return x + self.out_conv(d1)  


class _PatchDiscriminator(nn.Module):

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def image_gradient(x: torch.Tensor) -> torch.Tensor:
    
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag_sq = gx**2 + gy**2
    
    return torch.sqrt(torch.clamp(mag_sq, min=1e-12))


class DUGANDiscriminators(nn.Module):

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.image_disc = _PatchDiscriminator(in_channels, base_channels)
        self.grad_disc = _PatchDiscriminator(in_channels, base_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image_disc(x), self.grad_disc(image_gradient(x))
