"""The interface every degradation module implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from clearvisionnn.registry import Registry
from clearvisionnn.video.clip import Clip

Params = dict[str, Any]


class DegradeModule(ABC):
    """One step from clean video towards what the network will be given.

    A module separates what is fixed for a clip from what varies in time:
    `sample` draws the per-clip parameters (optics, tuning, encoder settings),
    and `apply` receives a generator for whatever is drawn fresh per frame
    (sensor noise). Only what really varies in time varies in training.
    """

    # Offline modules are too slow to run during training (an encoder round
    # trip); the pipeline stores their output as variants.
    offline: ClassVar[bool] = False
    # Output size is the input size divided by this.
    scale: ClassVar[int] = 1

    def __init__(self, probability: float = 1.0) -> None:
        # Chance that the module is applied to a clip at all. Leaving some
        # clips untouched teaches the network to do nothing when nothing is
        # needed.
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        self._probability = probability

    def sample(self, rng: np.random.Generator) -> Params | None:
        """Per-clip parameters, or None when the module is skipped for this clip."""
        if self.scale == 1 and rng.uniform() >= self._probability:
            return None
        return self.sample_params(rng)

    @abstractmethod
    def sample_params(self, rng: np.random.Generator) -> Params:
        """Draw the per-clip parameters."""

    @abstractmethod
    def apply(self, clip: Clip, params: Params, rng: np.random.Generator) -> Clip:
        """Degrade a float clip. `rng` is for per-frame draws only."""


MODULES: Registry[type[DegradeModule]] = Registry("degradation module")


def checked_range(
    name: str, interval: tuple[float, float], minimum: float = 0.0
) -> tuple[float, float]:
    """`interval` as (low, high), rejecting reversed bounds and values below `minimum`."""
    low, high = interval
    if not minimum <= low <= high:
        raise ValueError(f"{name} must satisfy {minimum} <= low <= high, got {interval}")
    return (float(low), float(high))


def skewed(rng: np.random.Generator, low: float, high: float, power: float) -> float:
    """A draw from [low, high] skewed towards `low` for `power` above 1."""
    return low + (high - low) * float(rng.uniform()) ** power
