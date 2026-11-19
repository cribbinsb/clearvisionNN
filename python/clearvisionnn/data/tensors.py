"""Clips as network tensors: 4:4:4 YUV in [0, 1], channels first."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from clearvisionnn.video.clip import CHROMA_SUBSAMPLING, MAX_SAMPLE, Clip
from clearvisionnn.video.coding import unknown_coding

LUMA = 0  # channel index of luma in every YUV tensor


def clip_to_tensor(clip: Clip) -> torch.Tensor:
    """(frames, 3, height, width) float YUV, chroma bilinearly upsampled to luma size."""
    luma = torch.from_numpy(np.ascontiguousarray(clip.y)).float()
    chroma = torch.from_numpy(np.stack([clip.u, clip.v], axis=1)).float()
    chroma = F.interpolate(
        chroma, scale_factor=CHROMA_SUBSAMPLING, mode="bilinear", align_corners=False
    )
    return torch.cat([luma[:, None], chroma], dim=1) / MAX_SAMPLE


def coding_to_tensor(clip: Clip) -> torch.Tensor:
    """Per-frame coding information, all zero for a clip no codec has touched."""
    coding = unknown_coding(clip.frames) if clip.coding is None else clip.coding
    return torch.from_numpy(np.ascontiguousarray(coding))
