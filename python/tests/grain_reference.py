"""The runtime's grain worked out a second time, for the tests: NumPy on whole arrays.

Independent of `cvnn_runtime::grain` in everything but the specification
(docs/INTEGER.md, section 9).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Ints = npt.NDArray[np.int64]

WORD = 0xFFFFFFFF
ROW, COLUMN = 0x9E3779B1, 0x85EBCA6B
KERNEL = (3, 10, 3)
AMPLITUDE_ONE, LEVEL_STEPS = 3848, 256
PICTURE_MAX = 8160


def scramble(word: Ints) -> Ints:
    """The 32-bit finaliser, on words held in 64 bits."""
    word = word ^ (word >> 16)
    word = (word * 0x7FEB352D) & WORD
    word = word ^ (word >> 15)
    word = (word * 0x846CA68B) & WORD
    return word ^ (word >> 16)


def white(frame: int, rows: Ints, columns: Ints) -> Ints:
    """The white samples at every (row, column), rows down and columns across."""
    frame_key = scramble(np.array(frame, np.int64))
    row_keys = scramble(frame_key ^ (((rows & WORD) * ROW) & WORD))
    words = scramble(row_keys[:, None] ^ (((columns[None, :] & WORD) * COLUMN) & WORD))
    return sum((words >> shift) & 255 for shift in (0, 8, 16, 24)) - 510


def field(frame: int, height: int, width: int) -> Ints:
    """The noise field of a `height` x `width` picture: the blurred white noise, in sixteenths."""
    samples = white(frame, np.arange(-1, height + 1), np.arange(-1, width + 1))
    across = sum(k * samples[:, i : i + width] for i, k in enumerate(KERNEL))
    blurred = sum(k * across[i : i + height] for i, k in enumerate(KERNEL))
    return (blurred + 8) >> 4


def grain_steps(frame: int, height: int, width: int, level: float) -> Ints:
    """The grain of frame `frame` at twice `height` x `width` for a profile of `level` grey levels.

    In picture steps.
    """
    steps = int(np.rint(np.float32(level) * np.float32(LEVEL_STEPS)))
    amplitude = (AMPLITUDE_ONE * steps + LEVEL_STEPS // 2) // LEVEL_STEPS
    return (amplitude * field(frame, 2 * height, 2 * width) + 65536) >> 17


def shown(clean: Ints, grain: Ints) -> Ints:
    """Clean luma in picture steps with its grain."""
    return np.clip(clean + grain, 0, PICTURE_MAX)
