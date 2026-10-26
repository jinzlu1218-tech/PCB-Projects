from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    """A small U-Net for one-class chip body segmentation."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.down1 = DoubleConv(in_channels, b)
        self.down2 = DoubleConv(b, b * 2)
        self.down3 = DoubleConv(b * 2, b * 4)
        self.down4 = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.conv3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.conv2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.conv1 = DoubleConv(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        u3 = self.conv3(torch.cat([self.up3(d4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))
        return self.head(u1)


def load_unet(path: str | Path, device: str | torch.device = "cpu") -> SmallUNet:
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    base_channels = int(checkpoint.get("base_channels", 32)) if isinstance(checkpoint, dict) else 32
    model = SmallUNet(base_channels=base_channels)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_chip_mask(
    model: SmallUNet,
    image: np.ndarray,
    size: tuple[int, int] = (512, 512),
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Return a full-resolution uint8 chip mask (0/255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    original_h, original_w = gray.shape[:2]
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(tensor)
    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = cv2.resize(probability, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return ((mask >= 0.5) * 255).astype(np.uint8)
