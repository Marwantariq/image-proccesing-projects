

"""RAW-to-RGB U-Net used by all three RAW models (baseline / RAW-HSI / RAW-HSI-Edge).

Extracted from raw-hvi/final.ipynb cell 16.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RawToRGBUNet(nn.Module):
    """4-channel packed Bayer in -> 3-channel RGB out."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(4, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 4, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)
        self.out_conv = nn.Conv2d(c, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        u3 = self.up3(b); d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))

