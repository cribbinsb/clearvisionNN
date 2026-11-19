"""The two-scale body shared by every layout: half resolution and quarter resolution."""

from __future__ import annotations

import torch
from torch import nn

from clearvisionnn.models.blocks import conv_relu, stack
from clearvisionnn.models.sizes import NetSize

UPSCALE = 2


class Trunk(nn.Module):
    """Features in and out at half the input resolution, with a quarter-resolution core.

    Half of the half-resolution blocks run before the quarter-resolution stage
    and half after, joined by a skip. Downsampling is a stride-2 convolution
    and upsampling a convolution plus depth-to-space: both are in every
    runtime's fast path.
    """

    def __init__(self, size: NetSize) -> None:
        super().__init__()
        half, quarter = size.half_channels, size.quarter_channels
        before = size.half_blocks // 2
        self.encode = stack(half, before)
        self.down = conv_relu(half, quarter, stride=2)
        self.core = stack(quarter, size.quarter_blocks)
        self.up = nn.Sequential(
            conv_relu(quarter, half * UPSCALE * UPSCALE), nn.PixelShuffle(UPSCALE)
        )
        self.decode = stack(half, size.half_blocks - before)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, half_channels, h/2, w/2) -> the same shape."""
        skip = self.encode(x)
        return self.decode(skip + self.up(self.core(self.down(skip))))
