"""Optical blur: the lens and anti-alias filter, as a Gaussian of random width."""

from __future__ import annotations

import cv2
import numpy as np

from clearvisionnn.pipeline.degrade.base import (
    MODULES,
    DegradeModule,
    Params,
    checked_range,
    skewed,
)
from clearvisionnn.video.clip import CHROMA_SUBSAMPLING, Clip

# Below this width a Gaussian changes nothing measurable.
_NEGLIGIBLE_SIGMA = 0.05


@MODULES.register("blur")
class Blur(DegradeModule):
    """Gaussian blur with one width per clip, skewed towards mild."""

    def __init__(
        self, sigma: tuple[float, float] = (0.0, 1.2), skew: float = 1.5, probability: float = 1.0
    ) -> None:
        super().__init__(probability)
        self._sigma, self._skew = checked_range("sigma", sigma), skew

    def sample_params(self, rng: np.random.Generator) -> Params:
        """Draw the blur width, in luma samples."""
        return {"sigma": skewed(rng, *self._sigma, self._skew)}

    def apply(self, clip: Clip, params: Params, rng: np.random.Generator) -> Clip:
        """Blur every frame; chroma by the same width in scene terms."""
        sigma = params["sigma"]
        if sigma < _NEGLIGIBLE_SIGMA:
            return clip
        return clip.map_planes(
            lambda f: cv2.GaussianBlur(f, (0, 0), sigma),
            lambda f: cv2.GaussianBlur(f, (0, 0), sigma / CHROMA_SUBSAMPLING),
        )
