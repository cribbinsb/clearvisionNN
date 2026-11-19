"""ISP sharpening: an unsharp mask, with the halos a camera's has."""

from __future__ import annotations

import cv2
import numpy as np

from clearvisionnn.pipeline.degrade.base import MODULES, DegradeModule, Params, checked_range
from clearvisionnn.video.clip import MAX_SAMPLE, Clip


@MODULES.register("sharpen")
class Sharpen(DegradeModule):
    """Unsharp mask on luma. Cameras sharpen, so the network must not sharpen twice."""

    def __init__(
        self,
        amount: tuple[float, float] = (0.2, 0.8),
        radius: tuple[float, float] = (0.6, 1.4),
        probability: float = 0.35,
    ) -> None:
        super().__init__(probability)
        self._amount = checked_range("amount", amount)
        self._radius = checked_range("radius", radius, minimum=0.1)

    def sample_params(self, rng: np.random.Generator) -> Params:
        """Draw the mask's gain and width."""
        return {
            "amount": float(rng.uniform(*self._amount)),
            "radius": float(rng.uniform(*self._radius)),
        }

    def apply(self, clip: Clip, params: Params, rng: np.random.Generator) -> Clip:
        """Sharpen luma and clip, as a camera's output stage does."""
        amount, radius = params["amount"], params["radius"]

        def sharpen(frame: np.ndarray) -> np.ndarray:
            detail = frame - cv2.GaussianBlur(frame, (0, 0), radius)
            return np.clip(frame + amount * detail, 0.0, MAX_SAMPLE)

        return clip.map_planes(sharpen, lambda f: f)
