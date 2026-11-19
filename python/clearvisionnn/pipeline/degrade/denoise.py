"""ISP spatial noise reduction: an edge-preserving filter that flattens fine texture."""

from __future__ import annotations

import cv2
import numpy as np

from clearvisionnn.pipeline.degrade.base import MODULES, DegradeModule, Params, checked_range
from clearvisionnn.video.clip import Clip

# Neighbourhood of the bilateral filter, in samples.
_DIAMETER = 5
_SPATIAL_SIGMA = 2.0


@MODULES.register("denoise")
class Denoise(DegradeModule):
    """Bilateral filter on luma with one strength per clip.

    A Gaussian would not reproduce what camera denoising does to texture: it
    blurs edges too, where a real ISP keeps edges and flattens what lies
    between them.
    """

    def __init__(
        self, strength: tuple[float, float] = (2.0, 12.0), probability: float = 0.7
    ) -> None:
        super().__init__(probability)
        self._strength = checked_range("strength", strength)

    def sample_params(self, rng: np.random.Generator) -> Params:
        """Draw the range sigma (0-255 units): differences below it are smoothed."""
        return {"strength": float(rng.uniform(*self._strength))}

    def apply(self, clip: Clip, params: Params, rng: np.random.Generator) -> Clip:
        """Filter luma; chroma is left to the encoder's subsampling."""
        strength = params["strength"]
        return clip.map_planes(
            lambda f: cv2.bilateralFilter(f, _DIAMETER, strength, _SPATIAL_SIGMA), lambda f: f
        )
