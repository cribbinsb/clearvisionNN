"""Still-picture compression of every frame, as webcams that deliver Motion JPEG apply it."""

from __future__ import annotations

import cv2
import numpy as np

from clearvisionnn.pipeline.degrade.base import MODULES, DegradeModule, Params, checked_range
from clearvisionnn.video.clip import MAX_SAMPLE, Clip


def _round_trip(plane: np.ndarray, quality: int) -> np.ndarray:
    """One plane through a greyscale JPEG at `quality`, back on the 0-255 float scale."""
    samples = np.clip(np.rint(plane), 0, MAX_SAMPLE).astype(np.uint8)
    encoded = cv2.imencode(".jpg", samples, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise ValueError("JPEG round trip failed to decode")
    return decoded.astype(np.float32)


@MODULES.register("jpeg")
class Jpeg(DegradeModule):
    """JPEG at one quality per clip; each plane is coded on its own, frame by frame."""

    def __init__(
        self, quality: tuple[float, float] = (70.0, 95.0), probability: float = 0.5
    ) -> None:
        super().__init__(probability)
        self._quality = checked_range("quality", quality, minimum=1.0)

    def sample_params(self, rng: np.random.Generator) -> Params:
        """Draw the quality setting."""
        return {"quality": round(float(rng.uniform(*self._quality)))}

    def apply(self, clip: Clip, params: Params, rng: np.random.Generator) -> Clip:
        """Compress and decompress every frame."""
        quality = params["quality"]
        return clip.map_planes(lambda f: _round_trip(f, quality), lambda f: _round_trip(f, quality))
