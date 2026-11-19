"""An integer model in memory and as a file (the layout is docs/INTEGER.md's).

The file is 32-bit little-endian words: a header, the input slots, then each
layer's packed weights and requantisation constants, and the tanh table.
`cvnn_runtime::integer` reads it and is the reference for its arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from clearvisionnn.contract import (
    INTEGER_HEADER_WORDS,
    INTEGER_MAGIC,
    INTEGER_SCHEMA_VERSION,
    INTEGER_STATE_BITS,
)

WORD_CHANNELS = 4  # channels packed into one word, a byte each
PICTURE_OUTPUTS = 4  # the 2x2 output pixels of one input pixel
NO_CHANNEL = -1  # an input slot that holds padding

Ints = npt.NDArray[np.int64]


@dataclass(frozen=True)
class IntegerConv:
    """A 3x3 convolution with its requantisation, per output channel."""

    weights: npt.NDArray[np.int8]  # (outputs, inputs, 3, 3); inputs a multiple of WORD_CHANNELS
    bias: Ints  # in accumulator units
    multiplier: Ints
    shift: Ints
    identity: Ints | None = None  # the identity branch's multiplier, if the layer has one

    def words(self) -> list[npt.NDArray[np.int32]]:
        """The layer in the file: weights `[tap][input word][output]`, then the constants."""
        outputs, inputs = self.weights.shape[:2]
        taps = self.weights.transpose(2, 3, 1, 0).reshape(
            -1, inputs // WORD_CHANNELS, WORD_CHANNELS, outputs
        )
        packed = np.ascontiguousarray(taps.transpose(0, 1, 3, 2)).view("<i4").ravel()
        constants = [self.bias, self.multiplier, self.shift]
        return [packed, *(_words(c) for c in (*constants, self.identity) if c is not None)]


@dataclass(frozen=True)
class IntegerModel:
    """A flat network in integer form."""

    input_channels: int  # of the float tensor the input planes are taken from
    unsigned_words: int
    slots: list[int]  # per input byte, the float input channel it holds, or NO_CHANNEL
    stem: IntegerConv
    body: list[IntegerConv]
    skip: tuple[int, int, int]  # stem multiplier, body multiplier, shift
    head: IntegerConv
    tanh: npt.NDArray[np.uint8]  # magnitude of the state per magnitude of the tanh input
    halo: int
    tile: int

    @property
    def state_outputs(self) -> int:
        """State channels the head returns beside the picture outputs."""
        return self.head.weights.shape[0] - PICTURE_OUTPUTS

    def words(self) -> npt.NDArray[np.int32]:
        """The whole file."""
        header = np.zeros(INTEGER_HEADER_WORDS, np.int32)
        fields = [
            INTEGER_MAGIC, INTEGER_SCHEMA_VERSION, self.input_channels, self.unsigned_words,
            len(self.slots) // WORD_CHANNELS - self.unsigned_words, self.stem.weights.shape[0],
            len(self.body), PICTURE_OUTPUTS, self.state_outputs, self.halo, self.tile,
            len(self.tanh), INTEGER_STATE_BITS,
        ]  # fmt: skip
        header[: len(fields)] = fields
        layers = [*self.stem.words(), *(w for conv in self.body for w in conv.words())]
        table = np.ascontiguousarray(self.tanh).view("<i4")
        parts = [header, _words(self.slots), *layers, _words(self.skip), *self.head.words(), table]
        return np.concatenate(parts).astype("<i4")


def _words(values: Ints | list[int] | tuple[int, ...]) -> npt.NDArray[np.int32]:
    """`values` as file words; refuses anything that does not fit."""
    wide = np.asarray(values, np.int64)
    if np.abs(wide).max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("a constant of the integer model does not fit 32 bits")
    return wide.astype(np.int32)
