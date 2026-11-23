"""The texture term: the output should carry as much fine texture as the original has.

A pixel loss and a perceptual term both let a network settle for a smooth
picture wherever the exact texture cannot be known: after a real-time encode
that is most of a background, and most of anything that moves. This term
compares how much there is, not where each grain of it sits: the local
amplitude of two frequency bands of the output against the original's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

# Samples each way over which a local mean is taken: the project's stillness
# neighbourhood, large enough to estimate an amplitude from.
NEIGHBOURHOOD = 15
# A neighbourhood counts as textured where the original's mean step between
# neighbours (across plus down) is above this (of 255): about the level of a
# furnished background, well above a flat wall's noise.
TEXTURED = 3 / 255
# Keeps the square root's gradient finite where a band is empty.
AMPLITUDE_EPSILON = 1e-8
HALF = 0.5


def _local(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, NEIGHBOURHOOD, stride=1, padding=NEIGHBOURHOOD // 2)


def bands(pictures: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, 1, H, W) pictures split in two bands.

    The top octave a half-size input can carry, and everything above it.
    """
    size = pictures.shape[-2:]
    half = F.interpolate(
        pictures, scale_factor=HALF, mode="bicubic", align_corners=False, antialias=True
    )
    quarter = F.interpolate(
        half, scale_factor=HALF, mode="bicubic", align_corners=False, antialias=True
    )
    from_half = F.interpolate(half, size=size, mode="bicubic", align_corners=False)
    from_quarter = F.interpolate(quarter, size=size, mode="bicubic", align_corners=False)
    return from_half - from_quarter, pictures - from_half


def textured(target: torch.Tensor) -> torch.Tensor:
    """1 where (n, 1, H, W) originals are textured, still or moving, else 0."""
    across = F.pad((target[..., 1:] - target[..., :-1]).abs(), (0, 1))
    down = F.pad((target[..., 1:, :] - target[..., :-1, :]).abs(), (0, 0, 0, 1))
    return (_local(across + down) > TEXTURED).float()


def _amplitude(band: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_local(band * band) + AMPLITUDE_EPSILON)


def texture_term(output: torch.Tensor, target: torch.Tensor, coded: torch.Tensor) -> torch.Tensor:
    """Mean difference in local band amplitude between output and original, both bands summed.

    `output` and `target` are (batch, frames, 1, H, W) luma; `coded` is 1 for
    the (batch, frames) that met a codec and 0 for the others, which are left
    alone: camera input has lost no texture to an encoder, and asking for
    more there costs accuracy for nothing. Samples outside textured places of
    coded frames count as zero in the mean.
    """
    flat_output, flat_target = output.float().flatten(0, 1), target.flatten(0, 1)
    where = textured(flat_target) * coded.flatten().float()[:, None, None, None]
    total = output.new_zeros((), dtype=torch.float32)
    for ours, theirs in zip(bands(flat_output), bands(flat_target), strict=True):
        total = total + ((_amplitude(ours) - _amplitude(theirs)).abs() * where).mean()
    return total
