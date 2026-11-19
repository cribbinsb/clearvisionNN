"""Fidelity: the output should equal the target."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from clearvisionnn.contract import SR_SCALE
from clearvisionnn.losses.base import LOSSES, Rollout, charbonnier


@LOSSES.register("charbonnier")
def charbonnier_loss(rollout: Rollout) -> torch.Tensor:
    """Charbonnier distance between output and target."""
    return charbonnier(rollout.output - rollout.target)


def downscaled(frames: torch.Tensor) -> torch.Tensor:
    """`frames` (batch, frames, channels, height, width) averaged back to input resolution."""
    return F.avg_pool2d(frames.flatten(0, 1), SR_SCALE).unflatten(0, frames.shape[:2])


@LOSSES.register("downscale_match")
def downscale_match_loss(rollout: Rollout) -> torch.Tensor:
    """Charbonnier distance between output and target once both are downscaled.

    Whatever detail the network adds, shrinking its output must give back the
    picture that shrinking the original gives: detail may be plausible rather
    than exact, the low frequencies may not.
    """
    return charbonnier(downscaled(rollout.output) - downscaled(rollout.target))
