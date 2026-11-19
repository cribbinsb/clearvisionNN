"""Structural metrics: whether structure is kept, whatever the exact pixel values."""

from __future__ import annotations

import torch
from pytorch_msssim import ms_ssim

from clearvisionnn.eval.metrics.base import METRICS, Family, Metric


@METRICS.register("ms_ssim")
class MsSsim(Metric):
    """Multi-scale SSIM of luma, in [0, 1]."""

    family = Family.STRUCTURE

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-frame MS-SSIM."""
        return ms_ssim(output[:, :1], target[:, :1], data_range=1.0, size_average=False)
