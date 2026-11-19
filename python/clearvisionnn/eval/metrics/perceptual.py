"""Perceptual metrics: softness and artefacts that PSNR forgives."""

from __future__ import annotations

import lpips
import torch

from clearvisionnn.eval.colour import yuv_to_rgb
from clearvisionnn.eval.metrics.base import METRICS, Family, Metric

# Frames per forward pass of the feature network, to bound memory at 1080p.
_CHUNK = 8


@METRICS.register("lpips")
class Lpips(Metric):
    """LPIPS (AlexNet features) on RGB; lower is better."""

    family = Family.PERCEPTUAL
    higher_is_better = False

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
        self._network = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-frame LPIPS distance."""
        scores = []
        for start in range(0, len(output), _CHUNK):
            a = yuv_to_rgb(output[start : start + _CHUNK]) * 2 - 1  # LPIPS expects [-1, 1]
            b = yuv_to_rgb(target[start : start + _CHUNK]) * 2 - 1
            scores.append(self._network(a, b).flatten())
        return torch.cat(scores)
