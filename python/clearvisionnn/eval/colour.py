"""Colour conversion for metrics defined on RGB."""

from __future__ import annotations

import torch

# 8-bit limited-range ("TV") levels, as fractions of full scale.
_LUMA_OFFSET, _LUMA_RANGE = 16 / 255, 219 / 255
_CHROMA_OFFSET, _CHROMA_RANGE = 128 / 255, 224 / 255
# BT.709 luma coefficients.
_KR, _KB = 0.2126, 0.0722
_KG = 1.0 - _KR - _KB


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    """BT.709 limited-range YUV (..., 3, H, W) in [0, 1] to full-range RGB in [0, 1]."""
    y = (yuv[..., 0, :, :] - _LUMA_OFFSET) / _LUMA_RANGE
    cb = (yuv[..., 1, :, :] - _CHROMA_OFFSET) / _CHROMA_RANGE
    cr = (yuv[..., 2, :, :] - _CHROMA_OFFSET) / _CHROMA_RANGE
    r = y + 2 * (1 - _KR) * cr
    b = y + 2 * (1 - _KB) * cb
    g = (y - _KR * r - _KB * b) / _KG
    return torch.stack([r, g, b], dim=-3).clamp(0.0, 1.0)
