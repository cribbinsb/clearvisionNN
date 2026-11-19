"""Fidelity metrics: whether the pixels are right."""

from __future__ import annotations

import torch

from clearvisionnn.eval.metrics.base import METRICS, Family, Metric

# Weights of the Y, U and V planes in the combined PSNR (the usual 6:1:1).
_YUV_WEIGHTS = (6 / 8, 1 / 8, 1 / 8)
# Caps the PSNR of a perfect frame, so that means stay finite.
_MIN_MSE = 1e-10


def _plane_psnr(output: torch.Tensor, target: torch.Tensor, channel: int) -> torch.Tensor:
    mse = ((output[:, channel] - target[:, channel]) ** 2).mean(dim=(1, 2))
    return -10.0 * torch.log10(mse.clamp_min(_MIN_MSE))


@METRICS.register("psnr_y")
class LumaPsnr(Metric):
    """PSNR of luma, dB."""

    family = Family.FIDELITY

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-frame luma PSNR."""
        return _plane_psnr(output, target, 0)


@METRICS.register("psnr_yuv")
class YuvPsnr(Metric):
    """6:1:1 weighted mean of the three planes' PSNR, dB; catches colour errors."""

    family = Family.FIDELITY

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-frame weighted PSNR."""
        planes = [w * _plane_psnr(output, target, c) for c, w in enumerate(_YUV_WEIGHTS)]
        return torch.stack(planes).sum(dim=0)
