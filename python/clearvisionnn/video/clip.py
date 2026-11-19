"""The in-memory form of a short piece of video: 4:2:0 planes over time."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

CHROMA_SUBSAMPLING = 2
MAX_SAMPLE = 255.0


@dataclass(frozen=True)
class Clip:
    """4:2:0 video. `y` is (frames, height, width); `u` and `v` are half that in each dimension.

    Planes are uint8 on disk and at the ends of the degradation chain, and
    float32 on the same 0-255 scale inside it, so that chained modules do not
    round at every step.
    """

    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    # (frames, CODING_CHANNELS) once a codec has produced the clip (`clearvisionnn.video.coding`).
    coding: np.ndarray | None = None

    def __post_init__(self) -> None:
        frames, height, width = self.y.shape
        if height % CHROMA_SUBSAMPLING or width % CHROMA_SUBSAMPLING:
            raise ValueError(f"4:2:0 needs even dimensions, got {width}x{height}")
        chroma = (frames, height // CHROMA_SUBSAMPLING, width // CHROMA_SUBSAMPLING)
        if self.u.shape != chroma or self.v.shape != chroma:
            raise ValueError(f"chroma planes {self.u.shape}, {self.v.shape} do not match {chroma}")
        if self.coding is not None and len(self.coding) != frames:
            raise ValueError(f"coding of {len(self.coding)} frames for a clip of {frames}")

    @property
    def frames(self) -> int:
        """Number of frames."""
        return int(self.y.shape[0])

    @property
    def height(self) -> int:
        """Luma height."""
        return int(self.y.shape[1])

    @property
    def width(self) -> int:
        """Luma width."""
        return int(self.y.shape[2])

    @property
    def planes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The three planes, luma first."""
        return self.y, self.u, self.v

    def as_float(self) -> Clip:
        """The same clip with float32 planes."""
        y, u, v = (p.astype(np.float32) for p in self.planes)
        return Clip(y, u, v, self.coding)

    def as_uint8(self) -> Clip:
        """The same clip rounded and clipped to uint8 planes."""
        if self.y.dtype == np.uint8:
            return self
        y, u, v = (np.clip(np.rint(p), 0, MAX_SAMPLE).astype(np.uint8) for p in self.planes)
        return Clip(y, u, v, self.coding)

    def time_slice(self, start: int, count: int) -> Clip:
        """Frames `start` to `start + count`."""
        coding = None if self.coding is None else self.coding[start : start + count]
        y, u, v = (p[start : start + count] for p in self.planes)
        return Clip(y, u, v, coding)

    def crop(self, top: int, left: int, height: int, width: int) -> Clip:
        """The luma window at (`top`, `left`); all four values must be even."""
        s = CHROMA_SUBSAMPLING
        if any(value % s for value in (top, left, height, width)):
            raise ValueError("4:2:0 crops need even coordinates and sizes")
        return Clip(
            self.y[:, top : top + height, left : left + width],
            self.u[:, top // s : (top + height) // s, left // s : (left + width) // s],
            self.v[:, top // s : (top + height) // s, left // s : (left + width) // s],
            self.coding,
        )

    def with_coding(self, coding: np.ndarray | None) -> Clip:
        """The same pictures with per-frame coding information attached."""
        return replace(self, coding=coding)

    def map_planes(self, luma, chroma) -> Clip:  # type: ignore[no-untyped-def]
        """A clip with `luma` applied to each luma frame and `chroma` to each chroma frame."""
        return Clip(
            np.stack([luma(f) for f in self.y]),
            np.stack([chroma(f) for f in self.u]),
            np.stack([chroma(f) for f in self.v]),
            self.coding,
        )
