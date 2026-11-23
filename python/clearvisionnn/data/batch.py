"""The sample and batch the models and losses consume."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from clearvisionnn.data.features import SequenceFeatures


@dataclass(frozen=True)
class Degraded:
    """One degraded view of a sequence with its pre-process features.

    Tensors have a leading time axis in a sample, and batch then time axes in
    a batch.
    """

    frames: torch.Tensor  # (..., 3, h, w) YUV in [0, 1]
    motion: torch.Tensor  # (..., 2, rows, cols)
    side: torch.Tensor  # (..., SIDE_CHANNELS, rows, cols)
    reset: torch.Tensor  # (...,) bool
    # (..., CODING_CHANNELS): how each frame was coded (`clearvisionnn.video.coding`);
    # for losses and measurements, never a network input.
    coding: torch.Tensor

    @classmethod
    def from_features(
        cls, frames: torch.Tensor, features: SequenceFeatures, coding: torch.Tensor
    ) -> Degraded:
        """Pair frames with the features computed from them and their coding information."""
        return cls(frames, features.motion, features.side, features.reset, coding)


@dataclass(frozen=True)
class SequenceBatch:
    """Clean targets with one or two degraded views of the same content."""

    target: torch.Tensor  # (..., 3, 2h, 2w) YUV in [0, 1]
    view: Degraded
    # A second view that differs only in what should not matter (the noise
    # realisation, the encoder's settings), for the consistency loss.
    alternate: Degraded | None = None

    def to(self, device: torch.device) -> SequenceBatch:
        """The batch on `device`."""
        return _map(self, lambda t: t.to(device, non_blocking=True))


def _map(value, function):  # type: ignore[no-untyped-def]
    """Apply `function` to every tensor of a (nested) batch dataclass."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return function(value)
    return type(value)(**{f.name: _map(getattr(value, f.name), function) for f in fields(value)})


def collate(samples: list[SequenceBatch]) -> SequenceBatch:
    """Stack samples along a new leading batch axis."""

    def stack(parts):  # type: ignore[no-untyped-def]
        first = parts[0]
        if first is None:
            return None
        if isinstance(first, torch.Tensor):
            return torch.stack(parts)
        return type(first)(
            **{f.name: stack([getattr(p, f.name) for p in parts]) for f in fields(first)}
        )

    return stack(samples)
