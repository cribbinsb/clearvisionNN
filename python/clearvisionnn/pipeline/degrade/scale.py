"""Downscaling by the super-resolution factor, with a random real scaler kernel."""

from __future__ import annotations

import numpy as np
from PIL import Image

from clearvisionnn.pipeline.degrade.base import MODULES, DegradeModule, Params
from clearvisionnn.video.clip import Clip

# Pillow filters over the whole reduced footprint, as real scalers do. OpenCV's
# cubic and Lanczos modes do not, and a network recovers about 1 dB less from
# their aliased output than from a properly filtered downscale.
_KERNELS = {
    "area": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@MODULES.register("scale")
class Scale(DegradeModule):
    """2x downscale. Always applied: it is what makes the input low resolution."""

    scale = 2

    def __init__(self, kernels: tuple[str, ...] = ("area", "bicubic", "lanczos")) -> None:
        super().__init__()
        unknown = set(kernels) - set(_KERNELS)
        if unknown:
            raise ValueError(f"unknown scaler kernels {sorted(unknown)}")
        self._kernels = list(kernels)

    def sample_params(self, rng: np.random.Generator) -> Params:
        """Pick the kernel."""
        return {"kernel": str(rng.choice(self._kernels))}

    def apply(self, clip: Clip, params: Params, rng: np.random.Generator) -> Clip:
        """Halve every plane."""
        flag = _KERNELS[params["kernel"]]

        def halve(frame: np.ndarray) -> np.ndarray:
            height, width = frame.shape
            size = (width // self.scale, height // self.scale)
            return np.asarray(Image.fromarray(frame).resize(size, flag))

        return clip.map_planes(halve, halve)
