"""The pre-process's output as network inputs: block motion and side planes.

This calls the Rust pre-process (`clearvisionnn_native`), the same code deployment
runs, on the degraded luma exactly as the network will see it.
"""

from __future__ import annotations

from dataclasses import dataclass

import clearvisionnn_native
import numpy as np
import torch

from clearvisionnn.contract import PLANE_NAMES


@dataclass(frozen=True)
class SequenceFeatures:
    """Per-frame features of one sequence, on the 8x8 block grid."""

    motion: torch.Tensor  # (frames, 2, rows, cols): (dx, dy) in input samples
    side: torch.Tensor  # (frames, SIDE_CHANNELS, rows, cols)
    reset: torch.Tensor  # (frames,) bool


def analyse(luma: np.ndarray, withheld: np.ndarray | None = None) -> SequenceFeatures:
    """Features of a (frames, height, width) uint8 luma sequence.

    `withheld` flags the planes (in `PLANE_NAMES` order) to blank out, as when
    a deployment cannot supply them.
    """
    flags = [False] * len(PLANE_NAMES) if withheld is None else [bool(w) for w in withheld]
    motion, side, reset = clearvisionnn_native.analyze_sequence(
        np.ascontiguousarray(luma), withheld=flags
    )
    return SequenceFeatures(
        motion=torch.from_numpy(motion), side=torch.from_numpy(side), reset=torch.from_numpy(reset)
    )
