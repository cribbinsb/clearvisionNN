"""Where and when a training sample is cut from a clip."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Crop corners fall on this grid of input samples, so that crops line up with
# the pre-process's block grid and with 4:2:0 chroma at every scale in use.
CROP_ALIGN = 8


@dataclass(frozen=True)
class Window:
    """A space-time window, in input-resolution samples."""

    start: int
    frames: int
    top: int
    left: int
    height: int
    width: int

    def scaled(self, factor: int) -> tuple[int, int, int, int]:
        """(top, left, height, width) of the same window at `factor` times the resolution."""
        return (self.top * factor, self.left * factor, self.height * factor, self.width * factor)


def full_frame(clip_frames: int, height: int, width: int) -> Window:
    """The window covering every frame of a clip, uncropped."""
    return Window(0, clip_frames, 0, 0, height, width)


def random_window(
    rng: np.random.Generator, clip_frames: int, height: int, width: int, frames: int, size: int
) -> Window:
    """A window of `frames` x `size` x `size` somewhere in a clip of the given input size."""
    if frames > clip_frames or size > min(height, width):
        raise ValueError(
            f"a {frames}x{size}x{size} window does not fit a {clip_frames}x{height}x{width} clip"
        )
    return Window(
        start=int(rng.integers(0, clip_frames - frames + 1)),
        frames=frames,
        top=int(rng.integers(0, (height - size) // CROP_ALIGN + 1)) * CROP_ALIGN,
        left=int(rng.integers(0, (width - size) // CROP_ALIGN + 1)) * CROP_ALIGN,
        height=size,
        width=size,
    )


def centre_window(clip_frames: int, height: int, width: int, frames: int, size: int) -> Window:
    """The window at the centre of the frame, from the clip's first frame."""
    size = min(size, height // CROP_ALIGN * CROP_ALIGN, width // CROP_ALIGN * CROP_ALIGN)
    return Window(
        start=0,
        frames=min(frames, clip_frames),
        top=(height - size) // 2 // CROP_ALIGN * CROP_ALIGN,
        left=(width - size) // 2 // CROP_ALIGN * CROP_ALIGN,
        height=size,
        width=size,
    )
