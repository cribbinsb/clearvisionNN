"""Temporal metrics: flicker, shimmer and lag."""

from __future__ import annotations

import torch

from clearvisionnn.eval.metrics.base import METRICS, Family, Metric

_MAX_SAMPLE = 255.0


@METRICS.register("frame_diff_error")
class FrameDifferenceError(Metric):
    """Mismatch between the output's frame-to-frame luma change and the target's.

    Mean absolute value in 0-255 units, one per frame pair; lower is better.
    The target's own change is the floor set by real motion, so this measures
    only the temporal error added on top of it.
    """

    family = Family.TEMPORAL
    higher_is_better = False

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-frame-pair temporal error."""
        output_change = output[1:, 0] - output[:-1, 0]
        target_change = target[1:, 0] - target[:-1, 0]
        return (output_change - target_change).abs().mean(dim=(1, 2)) * _MAX_SAMPLE
