"""Prepared validation clips, kept on disk in 8 bits, and the held-out sets training watches.

Preparing a clip (decode, degrade, analyse) takes far longer than loading it,
and the sets never change, so each is built once and stored.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from clearvisionnn.config.load import load_config
from clearvisionnn.data.batch import Degraded, SequenceBatch
from clearvisionnn.data.loaders import validation_loader
from recipe.sources import Source

# Validation clips of each source that the plateau rule and the calibration of
# the 8-bit conversion see.
HELD_OUT_CLIPS = 16
MAX_SAMPLE = 255


def pack(batch: SequenceBatch) -> dict[str, torch.Tensor]:
    """`batch` as it is stored: pictures in 8 bits, the rest unchanged."""
    view = batch.view
    return {
        "target": to_bytes(batch.target),
        "frames": to_bytes(view.frames),
        "motion": view.motion,
        "side": view.side,
        "reset": view.reset,
        "coding": view.coding,
    }


def unpack(stored: dict[str, torch.Tensor]) -> SequenceBatch:
    """The batch `pack` stored."""
    view = Degraded(
        stored["frames"].float() / MAX_SAMPLE,
        stored["motion"],
        stored["side"],
        stored["reset"],
        stored["coding"],
    )
    return SequenceBatch(stored["target"].float() / MAX_SAMPLE, view)


def to_bytes(picture: torch.Tensor) -> torch.Tensor:
    """A picture in [0, 1] as 8-bit samples."""
    return (picture * MAX_SAMPLE).round().byte()


def cached(path: Path, build: Callable[[], Any]) -> Any:
    """What `build` returns, stored at `path` the first time and loaded from there afterwards."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(build(), path)
    return torch.load(path)


def held_out_set(root: Path, source: Source, workers: int) -> list[SequenceBatch]:
    """The first validation clips of `source`, one clip per batch, cached under `root`."""

    def build() -> list[dict[str, torch.Tensor]]:
        config = load_config(source.config)
        train = config.train.model_copy(
            update={
                "validation_clips": HELD_OUT_CLIPS,
                "data": config.train.data.model_copy(update={"workers": workers}),
            }
        )
        loader = validation_loader(config.model_copy(update={"train": train}))
        return [pack(batch) for batch in loader]

    return [unpack(stored) for stored in cached(root / f"{source.name}.pt", build)]
