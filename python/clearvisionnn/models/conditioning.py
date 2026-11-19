"""Bringing the block-grid side planes to the network's input resolution."""

from __future__ import annotations

import torch

from clearvisionnn.contract import SIDE_CHANNELS
from clearvisionnn.models.resample import block_grid_to_frame


def side_planes(side: torch.Tensor, height: int, width: int, enabled: bool) -> torch.Tensor:
    """(batch, SIDE_CHANNELS, height, width): the planes, or zeros when the model takes none."""
    if not enabled:
        return side.new_zeros(side.shape[0], SIDE_CHANNELS, height, width)
    return block_grid_to_frame(side, height, width, "nearest")
