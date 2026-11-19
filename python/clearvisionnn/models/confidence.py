"""How well the motion field predicts a frame from the one before it."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from clearvisionnn.contract import BLOCK_SIZE, TRUSTED_ERROR, UNTRUSTED_ERROR
from clearvisionnn.models.resample import block_grid_to_frame


def confidence(luma: torch.Tensor, carried: torch.Tensor) -> torch.Tensor:
    """How far to trust motion-compensated history: (batch, 1, h, w) in [0, 1].

    It is 1 where `carried` (the previous luma brought onto this frame)
    matches `luma`, and falls to 0 as the block's mean error grows. Judged per
    analysis block and interpolated between block centres, so that noise in
    single samples does not register and no block outline is drawn.
    A recurrent network uses it to distrust its history where motion
    estimation fails: water, occlusions, a cut the analysis missed.
    """
    height, width = luma.shape[-2:]
    # Blocks overhanging the right and bottom edges average the samples they have.
    error = F.avg_pool2d((luma - carried).abs(), BLOCK_SIZE, ceil_mode=True)
    spread = block_grid_to_frame(error, height, width, "bilinear")
    return (1.0 - (spread - TRUSTED_ERROR) / (UNTRUSTED_ERROR - TRUSTED_ERROR)).clamp(0.0, 1.0)
