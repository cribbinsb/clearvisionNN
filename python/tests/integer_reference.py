"""The integer network worked out a second time, for the tests: NumPy on whole arrays.

Independent of `cvnn_runtime::integer` in everything but the specification
(docs/INTEGER.md): it reads the exporter's in-memory model, not the file.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from clearvisionnn.export.integer_model import (
    NO_CHANNEL,
    PICTURE_OUTPUTS,
    IntegerConv,
    IntegerModel,
)

Ints = npt.NDArray[np.int64]
Region = tuple[slice, slice]


def quantise_input(model: IntegerModel, features: npt.NDArray[np.float32]) -> Ints:
    """The input planes in slot order, on the stem's common scale of 1 / (255 * 127)."""
    unsigned = 4 * model.unsigned_words
    planes = np.zeros((len(model.slots), *features.shape[1:]), np.int64)
    for slot, channel in enumerate(model.slots):
        if channel == NO_CHANNEL:
            continue
        if slot < unsigned:
            planes[slot] = np.clip(np.rint(features[channel] * np.float32(255)), 0, 255) * 127
        else:
            planes[slot] = np.clip(np.rint(features[channel] * np.float32(127)), -127, 127) * 255
    return planes


def _accumulate(conv: IntegerConv, x: Ints) -> Ints:
    """Bias plus the zero-padded convolution; float64 sums of these integers are exact."""
    weights = torch.from_numpy(conv.weights.astype(np.float64))
    out = F.conv2d(torch.from_numpy(x.astype(np.float64))[None], weights, padding=1)[0]
    return np.asarray(np.rint(out.numpy()).astype(np.int64) + conv.bias[:, None, None])


def _requantise(conv: IntegerConv, accumulator: Ints, extra: Ints | int = 0) -> Ints:
    multiplier, shift = conv.multiplier[:, None, None], conv.shift[:, None, None]
    scaled = (accumulator >> 16) * multiplier + (((accumulator & 0xFFFF) * multiplier) >> 16)
    return np.asarray((scaled + extra + ((1 << shift) >> 1)) >> shift)


def _masked(values: Ints, region: Region | None) -> Ints:
    """`values` with zeros outside `region`: what the next layer of a tile may read."""
    if region is None:
        return values
    kept = np.zeros_like(values)
    kept[:, region[0], region[1]] = values[:, region[0], region[1]]
    return kept


def _forward(model: IntegerModel, planes: Ints, regions: list[Region | None]) -> Ints:
    """Head outputs (residual steps, then the stored state), valid inside `regions[-1]`."""
    stem = _masked(
        np.clip(_requantise(model.stem, _accumulate(model.stem, planes)), 0, 255), regions[0]
    )
    x = stem
    for conv, region in zip(model.body, regions[1:], strict=False):
        assert conv.identity is not None
        extra = conv.identity[:, None, None] * x
        x = _masked(np.clip(_requantise(conv, _accumulate(conv, x), extra), 0, 255), region)
    stem_by, body_by, shift = model.skip
    x = np.clip((x * body_by + stem * stem_by + ((1 << shift) >> 1)) >> shift, 0, 255)
    out = _requantise(model.head, _accumulate(model.head, _masked(x, regions[-2])))
    pictures = np.clip(out[:PICTURE_OUTPUTS], -(2**15), 2**15 - 1)
    magnitude = np.minimum(np.abs(out[PICTURE_OUTPUTS:]), max(len(model.tanh) - 1, 0))
    state = (
        np.sign(out[PICTURE_OUTPUTS:]) * model.tanh.astype(np.int64)[magnitude]
        if len(model.tanh)
        else out[:0]
    )
    return np.concatenate([pictures, state])


def infer(model: IntegerModel, planes: Ints, tile: int = 0) -> Ints:
    """The network's outputs (channels, height, width); `tile` 0 runs the whole frame at once."""
    layers = len(model.body) + 2
    height, width = planes.shape[1:]
    if tile == 0:
        return _forward(model, planes, [None] * layers)
    out = np.zeros((model.head.weights.shape[0], height, width), np.int64)
    for top in range(0, height, tile):
        for left in range(0, width, tile):

            def region(layer: int, top: int = top, left: int = left) -> Region:
                reach = min(model.halo, layers - 1 - layer)
                return (slice(max(0, top - reach), min(height, top + tile + reach)),
                        slice(max(0, left - reach), min(width, left + tile + reach)))  # fmt: skip

            regions: list[Region | None] = [region(layer) for layer in range(layers)]
            core = region(layers - 1)
            out[:, core[0], core[1]] = _forward(model, planes, regions)[:, core[0], core[1]]
    return out


def pack_output(outputs: Ints) -> npt.NDArray[np.uint32]:
    """Outputs as the words `IntegerNetwork.infer` returns: two 16-bit pairs, then bytes."""
    pictures = (outputs[:PICTURE_OUTPUTS] & 0xFFFF).astype(np.uint32)
    words = [pictures[0] | pictures[1] << 16, pictures[2] | pictures[3] << 16]
    state = (outputs[PICTURE_OUTPUTS:] & 0xFF).astype(np.uint32)
    words += [sum(state[4 * w + b] << (8 * b) for b in range(4)) for w in range(len(state) // 4)]
    return np.stack(words, axis=-1).ravel()
