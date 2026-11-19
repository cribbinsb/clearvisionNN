"""Reading and writing clips through the ffmpeg command line.

Used by the pipeline, where each call handles a whole clip. Training reads
stored clips in-process instead (`clearvisionnn.video.reader`).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from clearvisionnn.video.clip import Clip

FFMPEG = os.environ.get("CLEARVISIONNN_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("CLEARVISIONNN_FFPROBE", "ffprobe")

_QUIET = ["-hide_banner", "-loglevel", "error", "-nostdin"]
# Every clip the pipeline stores is 8-bit 4:2:0, BT.709, limited range.
STORED_PIX_FMT = "yuv420p"
# Lossless H.264 for stored targets; short GOPs keep random access cheap.
LOSSLESS_ARGS = ["-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-g", "8", "-bf", "0"]


class FfmpegError(RuntimeError):
    """ffmpeg rejected an input or failed to produce an output."""


def _run(command: Sequence[str], stdin: bytes | None = None) -> bytes:
    result = subprocess.run(command, input=stdin, capture_output=True, check=False)
    if result.returncode != 0:
        raise FfmpegError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-2000:]}")
    return result.stdout


def frames_from_bytes(data: bytes, width: int, height: int) -> Clip:
    """Parse packed 8-bit 4:2:0 frames."""
    luma, chroma = width * height, (width // 2) * (height // 2)
    frame_bytes = luma + 2 * chroma
    count = len(data) // frame_bytes
    packed = np.frombuffer(data, np.uint8, count * frame_bytes).reshape(count, frame_bytes)
    return Clip(
        packed[:, :luma].reshape(count, height, width),
        packed[:, luma : luma + chroma].reshape(count, height // 2, width // 2),
        packed[:, luma + chroma :].reshape(count, height // 2, width // 2),
    )


def frames_to_bytes(clip: Clip) -> bytes:
    """Pack a clip as 8-bit 4:2:0 frames."""
    clip = clip.as_uint8()
    count = clip.frames
    return np.concatenate([p.reshape(count, -1) for p in clip.planes], axis=1).tobytes()


def decode(path: Path, width: int, height: int, input_args: Sequence[str] = ()) -> Clip:
    """Decode all of `path`, whose frames are `width` x `height`."""
    command = [
        FFMPEG,
        *_QUIET,
        # Planes are parsed at the stored geometry; a rotation tag must not transpose them.
        "-noautorotate",
        *input_args,
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        STORED_PIX_FMT,
        "-",
    ]
    return frames_from_bytes(_run(command), width, height)


def _partial(path: Path) -> Path:
    """The temporary name a file is written under before it is moved into place."""
    return path.with_name(path.stem + ".partial" + path.suffix)


def encode(clip: Clip, path: Path, fps: float, codec_args: Sequence[str]) -> None:
    """Encode `clip` to `path` with the given encoder arguments, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _partial(path)
    command = [
        FFMPEG, *_QUIET, "-y", "-f", "rawvideo", "-pix_fmt", STORED_PIX_FMT,
        "-s", f"{clip.width}x{clip.height}", "-r", f"{fps:.6f}", "-i", "-",
        *codec_args, "-pix_fmt", STORED_PIX_FMT, str(temporary),
    ]  # fmt: skip
    _run(command, stdin=frames_to_bytes(clip))
    temporary.replace(path)


def transcode(
    source: Path, target: Path, input_args: Sequence[str], output_args: Sequence[str]
) -> None:
    """Run one ffmpeg conversion from `source` to `target`, file to file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _partial(target)
    _run([FFMPEG, *_QUIET, "-y", *input_args, "-i", str(source), *output_args, str(temporary)])
    temporary.replace(target)
