"""In-process random access to stored clips, for data loading."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from clearvisionnn.video.clip import Clip


def read_frames(path: Path, start: int, count: int) -> Clip:
    """Frames `start` to `start + count` of the stored clip at `path`.

    Stored clips have a constant frame rate and no reordering, so a frame's
    index follows from its timestamp. Decoding starts at the key frame at or
    before `start`.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"  # the loader already runs one worker per core
        if stream.average_rate is None or stream.time_base is None:
            raise ValueError(f"{path}: no frame rate or time base")
        ticks_per_frame = 1 / (stream.average_rate * stream.time_base)
        if start > 0:
            container.seek(int(start * ticks_per_frame), stream=stream, backward=True)
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            index = round(frame.pts / ticks_per_frame)
            if index < start:
                continue
            frames.append(frame.to_ndarray(format="yuv420p"))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError(f"{path}: wanted frames {start}..{start + count}, got {len(frames)}")
    return split_planes(np.stack(frames))


def decode_all(path: Path) -> np.ndarray:
    """Every frame of the stored clip at `path`, packed as (frames, height * 3 / 2, width)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        frames = [frame.to_ndarray(format="yuv420p") for frame in container.decode(stream)]
    if not frames:
        raise ValueError(f"{path}: no frames")
    return np.stack(frames)


def has_regular_timing(path: Path, frames: int) -> bool:
    """Whether `path` holds exactly `frames` frames whose timestamps index them 0, 1, 2, ...

    `read_frames` relies on this; a clip that fails it would be read misaligned.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None or stream.time_base is None:
            return False
        ticks_per_frame = 1 / (stream.average_rate * stream.time_base)
        stamps = sorted(p.pts for p in container.demux(stream) if p.pts is not None)
    return [round(stamp / ticks_per_frame) for stamp in stamps] == list(range(frames))


def split_planes(packed: np.ndarray) -> Clip:
    """Split (frames, height * 3 / 2, width) arrays into the three planes."""
    count, rows, width = packed.shape
    height = rows * 2 // 3
    chroma = packed[:, height:].reshape(count, 2, height // 2, width // 2)
    return Clip(packed[:, :height], chroma[:, 0], chroma[:, 1])
