"""How the trunk is connected to pixels: the layouts compared in DESIGN.md section 8.2.

A layout maps input-resolution features to two outputs: `fine`, at twice the
input resolution (image detail, and per-pixel maps such as the blend weight),
and `coarse`, at input resolution (hidden state).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from torch import nn

from clearvisionnn.contract import SR_SCALE
from clearvisionnn.models.blocks import RepConv, conv_relu, stack, zero_init
from clearvisionnn.models.sizes import NetSize
from clearvisionnn.models.trunk import Trunk
from clearvisionnn.registry import Registry

PACK = 2  # the packed layout folds this many samples per side into channels


class CoreOutput(NamedTuple):
    """What a layout produces."""

    fine: torch.Tensor  # (batch, fine_channels, 2h, 2w)
    coarse: torch.Tensor  # (batch, coarse_channels, h, w); zero channels if none wanted


class Layout(nn.Module):
    """Base of the layouts: fixes the constructor every layout shares."""

    def __init__(
        self, in_channels: int, fine_channels: int, coarse_channels: int, size: NetSize
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.fine_channels, self.coarse_channels = fine_channels, coarse_channels


LAYOUTS: Registry[type[Layout]] = Registry("layout")


@LAYOUTS.register("unet")
class UNetLayout(Layout):
    """A light stage at input resolution around the trunk, with a skip into the head.

    The input-resolution skip means fine detail does not depend on
    depth-to-space alone.
    """

    def __init__(
        self, in_channels: int, fine_channels: int, coarse_channels: int, size: NetSize
    ) -> None:
        super().__init__(in_channels, fine_channels, coarse_channels, size)
        stem, half = size.stem_channels, size.half_channels
        self.trunk = Trunk(size)
        self.stem = conv_relu(in_channels, stem)
        self.down = conv_relu(stem, half, stride=2)
        self.up = nn.Sequential(conv_relu(half, stem * PACK * PACK), nn.PixelShuffle(PACK))
        self.merge = conv_relu(stem, stem)
        # No ReLU after the head: it is the only narrow layer, and a ReLU on a
        # narrow layer can die and leave the network outputting nothing.
        self.head = zero_init(RepConv(stem, fine_channels * SR_SCALE**2 + coarse_channels))

    def forward(self, x: torch.Tensor) -> CoreOutput:
        """Input-resolution features to fine and coarse outputs."""
        skip = self.stem(x)
        out = self.head(self.merge(skip + self.up(self.trunk(self.down(skip)))))
        fine = out[:, : self.fine_channels * SR_SCALE**2]
        return CoreOutput(F.pixel_shuffle(fine, SR_SCALE), out[:, fine.shape[1] :])


@LAYOUTS.register("flat")
class FlatLayout(Layout):
    """Every layer at input resolution: the textbook super-resolution shape.

    Costs the most per layer, but it is the shape whose quality keeps rising
    with size; the two-scale layouts stop improving once most of their layers
    sit at a quarter of the input resolution.
    """

    def __init__(
        self, in_channels: int, fine_channels: int, coarse_channels: int, size: NetSize
    ) -> None:
        super().__init__(in_channels, fine_channels, coarse_channels, size)
        channels = size.flat_channels
        self.stem = conv_relu(in_channels, channels)
        self.body = stack(channels, size.flat_blocks)
        self.head = zero_init(RepConv(channels, fine_channels * SR_SCALE**2 + coarse_channels))

    def forward(self, x: torch.Tensor) -> CoreOutput:
        """Input-resolution features to fine and coarse outputs."""
        features = self.stem(x)
        out = self.head(features + self.body(features))
        fine = out[:, : self.fine_channels * SR_SCALE**2]
        return CoreOutput(F.pixel_shuffle(fine, SR_SCALE), out[:, fine.shape[1] :])


@LAYOUTS.register("packed")
class PackedLayout(Layout):
    """Nothing runs at input resolution: every layer works on wide, low-resolution tensors.

    Each 2x2 block of input samples is packed into channels, and the head
    emits each block's 4x4 output samples as channels. `core` is the part that
    goes on an accelerator; the packing and the final reorder to pixels are
    free in the surrounding code (DESIGN.md section 7).
    """

    def __init__(
        self, in_channels: int, fine_channels: int, coarse_channels: int, size: NetSize
    ) -> None:
        super().__init__(in_channels, fine_channels, coarse_channels, size)
        half = size.half_channels
        self.trunk = Trunk(size)
        self.stem = conv_relu(in_channels * PACK**2, half)
        self._fine_packed = fine_channels * (PACK * SR_SCALE) ** 2
        self.head = zero_init(RepConv(half, self._fine_packed + coarse_channels * PACK**2))

    def core(self, packed: torch.Tensor) -> torch.Tensor:
        """Packed input to packed outputs, all at half the input resolution."""
        return self.head(self.trunk(self.stem(packed)))

    def forward(self, x: torch.Tensor) -> CoreOutput:
        """Pack, run the core, and reorder its channels to pixels."""
        out = self.core(F.pixel_unshuffle(x, PACK))
        fine = F.pixel_shuffle(out[:, : self._fine_packed], PACK * SR_SCALE)
        return CoreOutput(fine, F.pixel_shuffle(out[:, self._fine_packed :], PACK))
