"""Temporal stability that does not depend on matching the original.

The frame-difference error counts any detail that differs from the original
as flicker once it moves. These two measures ask only whether the output's
own detail is steady: where the scene is still, and along its motion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from clearvisionnn.data.features import analyse
from clearvisionnn.models.resample import block_field_to_flow, warp

MAX_SAMPLE = 255.0
# The original changes by less than this (0-255) where the scene is still:
# above sensor noise on clean sources, below any visible motion.
STILL_THRESHOLD = 2.0
# A motion vector is trusted where it brings the original's previous frame
# within this (0-255) of the current one; elsewhere (occlusions, poor matches)
# no output could be steady.
MATCHED_THRESHOLD = 3.0

# Stillness and match quality are judged over this neighbourhood: judged per
# sample, the masks would pick the samples where the original's own noise
# happened to be small and understate its floor.
NEIGHBOURHOOD = 15

STILL, STILL_ORIGINAL = "still_flicker", "still_flicker_original"
MOVING, MOVING_ORIGINAL = "motion_flicker", "motion_flicker_original"


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of `values` where `mask`; zero if the mask is empty."""
    return float((values * mask).sum() / mask.sum().clamp_min(1))


def _local_mean(values: torch.Tensor) -> torch.Tensor:
    """Mean of (frames, height, width) `values` over each sample's neighbourhood."""
    return F.avg_pool2d(values[:, None], NEIGHBOURHOOD, stride=1, padding=NEIGHBOURHOOD // 2)[:, 0]


def _along_motion(luma: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """|frame - previous frame brought onto it| for frames 1.., in 0-255 units."""
    carried = warp(luma[:-1, None], flow[1:])[:, 0]
    return (luma[1:] - carried).abs() * MAX_SAMPLE


@torch.no_grad()
def stability(output: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Stability of (frames, height, width) luma in [0, 1], beside the original's own.

    `still_flicker`: mean frame-to-frame change where the original is still.
    `motion_flicker`: mean change along the original's motion, where that
    motion is trusted. The original's values are the floor real footage sets:
    its noise, and what resampling along motion costs any sharp picture.
    """
    change = lambda x: (x[1:] - x[:-1]).abs() * MAX_SAMPLE  # noqa: E731 - a local shorthand
    still = _local_mean(change(target)) < STILL_THRESHOLD
    features = analyse((target * MAX_SAMPLE).round().byte().cpu().numpy())
    height, width = target.shape[-2:]
    flow = block_field_to_flow(features.motion.to(target.device), height, width)
    original_residual = _along_motion(target, flow)
    cut = features.reset.to(target.device)[1:, None, None]
    matched = (_local_mean(original_residual) < MATCHED_THRESHOLD) & ~cut
    return {
        STILL: _masked_mean(change(output), still),
        STILL_ORIGINAL: _masked_mean(change(target), still),
        MOVING: _masked_mean(_along_motion(output, flow), matched),
        MOVING_ORIGINAL: _masked_mean(original_residual, matched),
    }
