"""What a video file contains, read with ffprobe."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel

from clearvisionnn.video.ffmpeg import FFPROBE


class ProbeError(ValueError):
    """A file that is not usable video."""


class VideoInfo(BaseModel):
    """Geometry, length and compression of a video stream."""

    width: int
    height: int
    fps: float
    frames: int
    codec: str
    pix_fmt: str
    # Bits per second of the video stream; 0 when uncompressed or unknown.
    bitrate: float
    # Colour description as tagged in the source ("" when untagged). Targets are
    # always converted to BT.709 limited range; this records what they came from.
    color_range: str = ""
    color_space: str = ""
    color_transfer: str = ""
    color_primaries: str = ""

    @property
    def bits_per_pixel(self) -> float:
        """Coded bits per luma sample per frame (infinite when uncompressed)."""
        if self.bitrate <= 0:
            return float("inf")
        return self.bitrate / (self.width * self.height * self.fps)


def probe(path: Path) -> VideoInfo:
    """Probe the first video stream of `path`."""
    command = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_packets,codec_name,pix_fmt,bit_rate,"
        "color_range,color_space,color_transfer,color_primaries"
        ":format=bit_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream, container = data["streams"][0], data.get("format", {})
        fps = float(Fraction(stream["avg_frame_rate"]))
    except (
        subprocess.CalledProcessError,
        KeyError,
        IndexError,
        ValueError,
        ZeroDivisionError,
    ) as e:
        raise ProbeError(f"{path}: not a readable video ({type(e).__name__})") from e
    if fps <= 0 or stream["width"] <= 0 or stream["height"] <= 0:
        raise ProbeError(f"{path}: no frame rate or picture size")
    bitrate = stream.get("bit_rate") or container.get("bit_rate") or 0
    return VideoInfo(
        width=stream["width"],
        height=stream["height"],
        fps=fps,
        frames=int(stream.get("nb_read_packets", 0)),
        codec=stream["codec_name"],
        pix_fmt=stream.get("pix_fmt", ""),
        bitrate=float(bitrate),
        color_range=stream.get("color_range", ""),
        color_space=stream.get("color_space", ""),
        color_transfer=stream.get("color_transfer", ""),
        color_primaries=stream.get("color_primaries", ""),
    )


def raw_info(path: Path, width: int, height: int, pix_fmt: str, fps: float) -> VideoInfo:
    """The description of headerless raw video with the given geometry."""
    bytes_per_frame = raw_frame_bytes(width, height, pix_fmt)
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frames=path.stat().st_size // bytes_per_frame,
        codec="rawvideo",
        pix_fmt=pix_fmt,
        bitrate=0.0,
    )


# Samples per pixel of the raw layouts the pipeline accepts.
_SAMPLES_PER_PIXEL = {"yuv420p": 1.5, "yuv422p": 2.0, "yuv444p": 3.0}


def raw_frame_bytes(width: int, height: int, pix_fmt: str) -> int:
    """Bytes one frame of raw `pix_fmt` video occupies."""
    layout = pix_fmt.rstrip("le").rstrip("be")
    bytes_per_sample = 1
    for depth in ("16", "12", "10"):
        if layout.endswith(depth):
            layout, bytes_per_sample = layout[: -len(depth)], 2
            break
    if layout not in _SAMPLES_PER_PIXEL:
        raise ValueError(f"unsupported raw pixel format '{pix_fmt}'")
    return int(width * height * _SAMPLES_PER_PIXEL[layout] * bytes_per_sample)
