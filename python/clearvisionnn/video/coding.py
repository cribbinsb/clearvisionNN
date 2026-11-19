"""What a decoder knows about how each frame was coded: quantiser and frame type.

A receiver reads both from the slice headers of the stream it decodes, so they
cost nothing to supply; they tell the network how much the encoder discarded.
"""

from __future__ import annotations

import re

import numpy as np

# Per frame: whether the values are known, the quantiser on a 0-1 scale, and
# whether the frame is a key frame. A frame that never met a codec is all zero.
CODING_CHANNELS = 3
# The largest quantiser of H.264 and H.265 at 8 bits; it sets the 0-1 scale.
MAX_QP = 51.0

# One line per frame of ffmpeg's -vstats_file output.
_VSTATS_LINE = re.compile(r"\bq=\s*([0-9.]+).*\btype=\s*(\w)")


def unknown_coding(frames: int) -> np.ndarray:
    """The coding of `frames` frames that no encoder has touched."""
    return np.zeros((frames, CODING_CHANNELS), np.float32)


def parse_encoder_stats(text: str) -> np.ndarray:
    """Per-frame coding from the text ffmpeg writes with `-vstats_file`."""
    rows = [_VSTATS_LINE.search(line) for line in text.splitlines() if line.strip()]
    if not rows or any(row is None for row in rows):
        raise ValueError("unrecognised encoder statistics")
    return np.array(
        [(1.0, float(row[1]) / MAX_QP, float(row[2] == "I")) for row in rows if row],
        np.float32,
    )
