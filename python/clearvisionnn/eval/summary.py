"""Reducing per-frame scores to per-clip numbers, and per-clip numbers to a verdict."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClipScore:
    """One metric on one clip for one method."""

    mean: float
    # Mean over the worst run of consecutive frames: a few bad frames or a
    # pulse are what viewers notice, and the clip mean hides them.
    worst_window: float


def summarise(per_frame: np.ndarray, window: int, higher_is_better: bool) -> ClipScore:
    """The clip mean and the mean of its worst `window` consecutive frames."""
    window = min(window, len(per_frame))
    sums = np.convolve(per_frame, np.ones(window) / window, mode="valid")
    worst = sums.min() if higher_is_better else sums.max()
    return ClipScore(float(per_frame.mean()), float(worst))


@dataclass(frozen=True)
class Interval:
    """A mean with a bootstrap confidence interval over clips."""

    mean: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the whole interval lies on one side of zero."""
        return self.low > 0 or self.high < 0


def bootstrap_mean(
    values: np.ndarray, groups: Sequence[str], samples: int, seed: int = 0, coverage: float = 0.95
) -> Interval:
    """The mean of per-clip `values` with a percentile bootstrap interval.

    Whole groups are resampled, a group being the clips cut from one source
    file: those share content, so treating them as independent would make
    the interval too narrow. A result has to hold across content.
    """
    names = sorted(set(groups))
    members = [np.flatnonzero(np.array(groups) == name) for name in names]
    sums = np.array([values[m].sum() for m in members])
    counts = np.array([len(m) for m in members])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(samples, len(names)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - coverage) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return Interval(float(values.mean()), float(low), float(high))
