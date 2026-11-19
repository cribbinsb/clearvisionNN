"""Temporal difference: the output should change between frames exactly as the target does.

Needs no optical flow. Flicker, shimmer and lag all show up as a mismatch
between the output's frame-to-frame change and the target's.
"""

from __future__ import annotations

import torch

from clearvisionnn.losses.base import LOSSES, Rollout, charbonnier


@LOSSES.register("temporal_difference")
def temporal_difference_loss(rollout: Rollout) -> torch.Tensor:
    """Charbonnier distance between the frame differences of output and target."""
    if rollout.output.shape[1] < 2:
        return rollout.output.new_zeros(())
    output_change = rollout.output[:, 1:] - rollout.output[:, :-1]
    target_change = rollout.target[:, 1:] - rollout.target[:, :-1]
    return charbonnier(output_change - target_change)
