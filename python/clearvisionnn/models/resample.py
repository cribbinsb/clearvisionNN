"""Fixed (untrained) resampling: the conventional upscale, and warping by a motion field."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from clearvisionnn.contract import BLOCK_SIZE, SR_SCALE


def upscale(frame: torch.Tensor) -> torch.Tensor:
    """The conventional 2x upscale the network's detail is added to (bicubic)."""
    out = F.interpolate(frame, scale_factor=SR_SCALE, mode="bicubic", align_corners=False)
    return out.clamp(0.0, 1.0)


def block_grid_to_frame(grid: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
    """A block-grid tensor (batch, channels, rows, cols) resampled to one value per sample.

    Blocks on the right and bottom edges may overhang the frame, so the grid
    is resampled over the area it covers and then cropped to the frame.
    """
    covered = (grid.shape[-2] * BLOCK_SIZE, grid.shape[-1] * BLOCK_SIZE)
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(grid, size=covered, mode=mode, **kwargs)[..., :height, :width]


def block_field_to_flow(motion: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """A per-sample flow (batch, 2, height, width) from the per-block field.

    Vectors are interpolated between block centres, and stay in input samples.
    """
    return block_grid_to_frame(motion, height, width, "bilinear")


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample `image` at each position displaced by `flow` (dx, dy in samples).

    With the pre-process's convention (the matching content of the previous
    frame is at position + vector), this brings the previous frame onto the
    current one. Positions outside the image take the nearest edge sample.
    """
    _, _, height, width = image.shape
    ys = torch.arange(height, device=image.device, dtype=image.dtype)
    xs = torch.arange(width, device=image.device, dtype=image.dtype)
    grid_x = (xs[None, None, :] + flow[:, 0] + 0.5) / width * 2 - 1
    grid_y = (ys[None, :, None] + flow[:, 1] + 0.5) / height * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
