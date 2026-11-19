"""Conventional upscalers: what a network has to beat."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch

from clearvisionnn.contract import SR_SCALE
from clearvisionnn.models.resample import upscale
from clearvisionnn.registry import Registry

# (frames, 3, h, w) YUV in [0, 1] -> (frames, 3, 2h, 2w).
Baseline = Callable[[torch.Tensor], torch.Tensor]
BASELINES: Registry[Baseline] = Registry("baseline")

# Unsharp-mask settings of the "upscale then sharpen" baseline: a moderate,
# typical choice, not tuned to any metric.
_SHARPEN_AMOUNT = 0.5
_SHARPEN_RADIUS = 1.0


@BASELINES.register("bicubic")
def bicubic(frames: torch.Tensor) -> torch.Tensor:
    """Bicubic: the upscale the networks' detail is added to."""
    return upscale(frames)


def _lanczos_planes(frames: torch.Tensor) -> np.ndarray:
    planes = frames.cpu().numpy()
    count, channels, height, width = planes.shape
    size = (width * SR_SCALE, height * SR_SCALE)
    resized = [
        cv2.resize(p, size, interpolation=cv2.INTER_LANCZOS4)
        for p in planes.reshape(-1, height, width)
    ]
    return np.stack(resized).reshape(count, channels, *size[::-1])


@BASELINES.register("lanczos")
def lanczos(frames: torch.Tensor) -> torch.Tensor:
    """Lanczos (8x8 kernel)."""
    return torch.from_numpy(_lanczos_planes(frames)).clamp(0.0, 1.0).to(frames.device)


@BASELINES.register("lanczos_sharpen")
def lanczos_sharpen(frames: torch.Tensor) -> torch.Tensor:
    """Lanczos, then an unsharp mask on luma: the classic upscale-and-sharpen."""
    planes = _lanczos_planes(frames)
    for frame in planes:
        luma = frame[0]
        frame[0] = luma + _SHARPEN_AMOUNT * (luma - cv2.GaussianBlur(luma, (0, 0), _SHARPEN_RADIUS))
    return torch.from_numpy(planes).clamp(0.0, 1.0).to(frames.device)
