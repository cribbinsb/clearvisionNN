"""Degradation consistency: what should not matter must not change the output.

The batch's alternate view shows the same content with a different noise
realisation (or different encoder settings). Any difference between the two
outputs is noise or coding artefacts turned into picture, which is seen as
flicker.
"""

from __future__ import annotations

import torch

from clearvisionnn.losses.base import LOSSES, Rollout, charbonnier


@LOSSES.register("consistency")
def consistency_loss(rollout: Rollout) -> torch.Tensor:
    """Charbonnier distance between the outputs for the two views."""
    if rollout.alternate is None:
        raise ValueError("the consistency loss needs train.data.noise_pair: true")
    return charbonnier(rollout.output - rollout.alternate)
