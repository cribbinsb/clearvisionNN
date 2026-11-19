"""Sensor noise: one level per clip, a fresh realisation per frame."""

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
from clearvisionnn.video.clip import Clip


@MODULES.register("noise")
class Noise(DegradeModule):
    """Additive noise, optionally spatially correlated as after demosaicing."""

    def __init__(
        self,
        sigma: tuple[float, float] = (0.0, 4.0),
        skew: float = 1.5,
        chroma_ratio: float = 0.6,
        correlation: tuple[float, float] = (0.0, 0.7),
        probability: float = 1.0,
    ) -> None:
        super().__init__(probability)
        self._sigma, self._skew = checked_range("sigma", sigma), skew
        self._chroma_ratio = chroma_ratio
        self._correlation = checked_range("correlation", correlation)

    def sample_params(self, rng: np.random.Generator) -> Params:
        """Draw the noise level (0-255 units) and its spatial correlation width."""
        return {
            "sigma": skewed(rng, *self._sigma, self._skew),
            "correlation": float(rng.uniform(*self._correlation)),
        }

    def apply(self, clip: Clip, params: Params, rng: np.random.Generator) -> Clip:
        """Add an independent realisation to every frame."""
        sigma, correlation = params["sigma"], params["correlation"]

        def add(frame: np.ndarray, level: float) -> np.ndarray:
            noise: np.ndarray = rng.standard_normal(frame.shape, dtype=np.float32)
            if correlation > 0:
                noise = cv2.GaussianBlur(noise, (0, 0), correlation)
                noise = noise / (noise.std() + 1e-6)  # blurring lowers the level; restore it
            return frame + level * noise

        return clip.map_planes(
            lambda f: add(f, sigma), lambda f: add(f, sigma * self._chroma_ratio)
        )
