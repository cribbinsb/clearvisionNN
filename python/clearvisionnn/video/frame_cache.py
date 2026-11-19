"""Decoded copies of stored clips, mapped from disk, for fast data loading.

Opening and decoding a compressed clip costs far more than the few cropped
frames a training sample needs. The first read of a clip decodes all of it
into a raw array file; later reads map that file and touch only the samples
they use.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from clearvisionnn.video.clip import Clip
from clearvisionnn.video.reader import decode_all, split_planes


class FrameCache:
    """Raw copies of clips under `directory`, made on first use.

    A clip's cache file is named after its own file and parent directory;
    stored clips carry their fingerprint in their name, so a changed clip
    never finds a stale copy.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def read(self, path: Path, start: int, count: int) -> Clip:
        """Frames `start` to `start + count` of the stored clip at `path`."""
        packed = np.load(self._cached(path), mmap_mode="r")
        if start + count > packed.shape[0]:
            raise ValueError(
                f"{path}: wanted frames {start}..{start + count}, clip has {packed.shape[0]}"
            )
        return split_planes(packed[start : start + count])

    def _cached(self, path: Path) -> Path:
        cached = self._directory / path.parent.name / (path.stem + ".npy")
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            # Written under another name and renamed, so that workers racing
            # to fill the same entry never read a partial file.
            handle, temporary = tempfile.mkstemp(dir=cached.parent, suffix=".partial")
            with os.fdopen(handle, "wb") as file:
                np.save(file, decode_all(path))
            os.replace(temporary, cached)
        return cached
