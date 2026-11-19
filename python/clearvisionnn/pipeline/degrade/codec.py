"""A real encoder round trip in a low-delay configuration (DESIGN.md section 9.3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from clearvisionnn.pipeline.degrade.base import MODULES, DegradeModule, Params, checked_range
from clearvisionnn.video.clip import Clip
from clearvisionnn.video.coding import parse_encoder_stats
from clearvisionnn.video.ffmpeg import decode, encode

# Frame rate the bitrate is computed for; the clip's own rate is not known
# here, and what matters is bits per frame.
_NOMINAL_FPS = 30.0
# A GOP longer than any clip: one key frame, then forward prediction only.
_INFINITE_GOP = 100_000

_LOW_DELAY = {
    "libx264": ["-tune", "zerolatency", "-bf", "0"],
    "libx265": ["-tune", "zerolatency", "-bf", "0", "-x265-params", "log-level=error"],
}


def encoder_args(params: Params, width: int, height: int) -> list[str]:
    """The ffmpeg arguments that realise `params` for a `width` x `height` clip."""
    bitrate = params["bits_per_pixel"] * width * height * _NOMINAL_FPS
    return [
        "-c:v",
        params["codec"],
        "-preset",
        params["preset"],
        "-b:v",
        f"{bitrate:.0f}",
        "-maxrate",
        f"{bitrate:.0f}",
        "-bufsize",
        f"{bitrate:.0f}",
        "-g",
        str(params["gop"]),
        *_LOW_DELAY[params["codec"]],
    ]


@MODULES.register("codec")
class Codec(DegradeModule):
    """Encode and decode with a randomly chosen encoder, bitrate, preset and GOP length."""

    offline = True

    def __init__(
        self,
        codecs: tuple[str, ...] = ("libx264", "libx265"),
        bits_per_pixel: tuple[float, float] = (0.04, 0.3),
        presets: tuple[str, ...] = ("ultrafast", "veryfast", "medium"),
        gops: tuple[int, ...] = (30, 60, _INFINITE_GOP),
        probability: float = 0.9,
    ) -> None:
        super().__init__(probability)
        unknown = set(codecs) - set(_LOW_DELAY)
        if unknown:
            raise ValueError(f"unsupported codecs {sorted(unknown)}")
        self._codecs, self._presets, self._gops = list(codecs), list(presets), list(gops)
        self._bits_per_pixel = checked_range("bits_per_pixel", bits_per_pixel, minimum=1e-4)

    def sample_params(self, rng: np.random.Generator) -> Params:
        """Draw the encoder settings; bitrate is log-uniform, as rate ladders are."""
        low, high = np.log(self._bits_per_pixel)
        return {
            "codec": str(rng.choice(self._codecs)),
            "preset": str(rng.choice(self._presets)),
            "gop": int(rng.choice(self._gops)),
            "bits_per_pixel": float(np.exp(rng.uniform(low, high))),
        }

    def apply(self, clip: Clip, params: Params, rng: np.random.Generator) -> Clip:
        """Round-trip the clip through the encoder, keeping how each frame was coded."""
        with tempfile.TemporaryDirectory(prefix="clearvisionnn-codec-") as directory:
            stream = Path(directory) / "stream.mkv"
            stats = Path(directory) / "frames.log"
            arguments = encoder_args(params, clip.width, clip.height)
            encode(clip, stream, _NOMINAL_FPS, [*arguments, "-vstats_file", str(stats)])
            decoded = decode(stream, clip.width, clip.height).as_float()
            # The quantiser and type of each frame, as the receiver would read them.
            return decoded.with_coding(parse_encoder_stats(stats.read_text()))
