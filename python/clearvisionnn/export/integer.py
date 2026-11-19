"""Turning a trained flat network into an integer model (docs/INTEGER.md).

The scheme is the one quantisation-aware training simulates: weights rounded
to signed 8 bits per output channel with the identity branch kept apart,
unsigned 8-bit activations whose top is learnt per layer, and every input
plane 8-bit. This module fixes what the simulation leaves open: biases in
accumulator units and each real-valued scale as an integer multiplier and shift.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from clearvisionnn.config.model import ModelConfig
from clearvisionnn.contract import (
    INTEGER_MULTIPLIER_BITS,
    INTEGER_RESIDUAL_STEPS,
    INTEGER_TANH_STEPS,
)
from clearvisionnn.export.integer_model import (
    NO_CHANNEL,
    PICTURE_OUTPUTS,
    WORD_CHANNELS,
    IntegerConv,
    IntegerModel,
    Ints,
)  # fmt: skip
from clearvisionnn.models.blocks import RepConv
from clearvisionnn.models.factory import trained_core
from clearvisionnn.models.layouts import FlatLayout
from clearvisionnn.models.luma import state_input_channels
from clearvisionnn.models.sequence import SequenceModel
from clearvisionnn.models.state import BoundedState

UNSIGNED_LEVELS, SIGNED_LEVELS = 255, 127  # steps of an unsigned and of a signed 8-bit plane
WEIGHT_LEVELS = 127  # a channel's largest weight, in steps
SMALLEST_WEIGHT = 1e-8  # floor of a channel's weight range, as in the training simulation
# Bits of the accumulator a requantisation multiplies separately (cvnn_runtime::integer::arith).
SPLIT_BITS = 16
MAX_SHIFT = 30  # a requantisation's final shift stays below the word size
# Identity and skip multipliers stay below 2^22 so that times an 8-bit value they fit a word.
BRANCH_BITS = 22
# The tanh table ends where 127 * tanh has long settled at 127.
TANH_RANGE = 4
Floats = npt.NDArray[np.float64]
# Deployment defaults, measured on the hand-written WebGPU prototype (docs/INTEGER.md):
# six pixels of halo keep a tile's seams under a quarter of an 8-bit level, and at
# 128 pixels all tiles of a layer in one dispatch cost no more than the whole frame.
DEFAULT_HALO = 6
DEFAULT_TILE = 128


def _step(top: float) -> float:
    """An activation's step from its learnt top, in the simulation's own precision."""
    return float(np.abs(np.float32(top)) / np.float32(UNSIGNED_LEVELS))


def _rounded_weights(conv: RepConv) -> tuple[npt.NDArray[np.int8], Floats]:
    """Conv plus 1x1 branch as one signed 8-bit kernel, and each output channel's step."""
    kernel = conv.conv.weight.detach() + F.pad(conv.point.weight.detach(), [1] * 4)
    step = kernel.abs().amax((1, 2, 3), keepdim=True).clamp_min(SMALLEST_WEIGHT) / WEIGHT_LEVELS
    return torch.round(kernel / step).numpy().astype(np.int8), step.flatten().double().numpy()


def _multiplier(scale: Floats, most_shift: int = MAX_SHIFT) -> tuple[Ints, Ints]:
    """`scale` per channel as `multiplier / 2^(SPLIT_BITS + shift)`, as wide as allowed."""
    ceiling = 2**INTEGER_MULTIPLIER_BITS - 1
    shift = np.minimum(np.floor(np.log2(ceiling / scale)).astype(np.int64) - SPLIT_BITS, most_shift)
    if (shift < 0).any():
        raise ValueError("a layer's output scale is too coarse for the fixed-point multiplier")
    return np.rint(scale * 2.0 ** (SPLIT_BITS + shift)).astype(np.int64), shift


def _check_range(weights: npt.NDArray[np.int8], bias: Ints, largest_input: Ints) -> None:
    """Refuse a layer whose accumulator could leave 32 bits (`largest_input` per channel)."""
    reach = np.abs(weights.astype(np.int64)).sum((2, 3)) @ largest_input + np.abs(bias)
    if reach.max() > np.iinfo(np.int32).max:
        raise ValueError("a layer's accumulator can overflow 32 bits")


def _conv(conv: RepConv, weights: npt.NDArray[np.int8], unit: Floats, out_step: Floats | float,
          largest_input: Ints, identity_step: float | None = None) -> IntegerConv:  # fmt: skip
    """The integer layer whose accumulator counts `unit` and whose output counts `out_step`."""
    bias = np.rint(conv.conv.bias.detach().double().numpy() / unit).astype(np.int64)  # type: ignore[union-attr]
    _check_range(weights, bias, largest_input)
    if identity_step is None:
        return IntegerConv(weights, bias, *_multiplier(unit / out_step))
    ratio = identity_step / float(out_step)
    most = BRANCH_BITS - 1 - max(0, int(np.ceil(np.log2(ratio))))
    multiplier, shift = _multiplier(unit / out_step, most)
    return IntegerConv(
        weights, bias, multiplier, shift, np.rint(ratio * 2.0**shift).astype(np.int64)
    )


def _stem(conv: RepConv, signed: range, out_step: float) -> tuple[IntegerConv, list[int], int]:
    """The stem, its inputs regrouped: unsigned planes, then signed, each in whole words."""
    weights, weight_step = _rounded_weights(conv)
    channels = weights.shape[1]
    groups = [[c for c in range(channels) if c not in signed], list(signed)]
    slots = [s for group in groups for s in group + [NO_CHANNEL] * (-len(group) % WORD_CHANNELS)]
    padded = np.concatenate([weights, np.zeros_like(weights[:, :1])], axis=1)[
        :, slots
    ]  # NO_CHANNEL picks the zeros
    unsigned_words = (len(groups[0]) + WORD_CHANNELS - 1) // WORD_CHANNELS
    # Unsigned planes count 1/255 and signed ones 1/127: both are brought onto 1/(255 * 127).
    largest = np.array([UNSIGNED_LEVELS * SIGNED_LEVELS] * len(slots), np.int64)
    unit = weight_step / (UNSIGNED_LEVELS * SIGNED_LEVELS)
    return _conv(conv, padded, unit, out_step, largest), slots, unsigned_words


def _skip(stem_step: float, body_step: float, out_step: float) -> tuple[int, int, int]:
    """Multipliers and shift bringing the stem's and body's outputs onto the head's input step."""
    ratios = np.array([stem_step, body_step]) / out_step
    shift = (
        BRANCH_BITS - 1 - UNSIGNED_LEVELS.bit_length() - max(0, int(np.ceil(np.log2(ratios.max()))))
    )
    stem, body = np.rint(ratios * 2.0**shift).astype(int).tolist()
    return stem, body, shift


def tanh_table() -> npt.NDArray[np.uint8]:
    """Magnitude of the stored state (in 1/127) per magnitude of the tanh input (in 1/1024)."""
    inputs = np.arange(TANH_RANGE * INTEGER_TANH_STEPS, dtype=np.float64) / INTEGER_TANH_STEPS
    return np.rint(SIGNED_LEVELS * np.tanh(inputs)).astype(np.uint8)


def quantise(
    layout: FlatLayout, tops: Sequence[float], signed: range, halo: int, tile: int
) -> IntegerModel:
    """`layout` (not fused) as an integer model.

    `tops` are the learnt activation tops: the stem's, each body block's, then
    the long skip's. `signed` are the input channels holding the recurrent
    state; every other input plane is unsigned.
    """
    stem_conv, *blocks = [
        m for part in (layout.stem, layout.body) for m in part.modules() if isinstance(m, RepConv)
    ]
    if len(tops) != len(blocks) + 2:
        raise ValueError(f"{len(blocks) + 2} activation tops needed, {len(tops)} given")
    steps = [_step(top) for top in tops]
    stem, slots, unsigned_words = _stem(stem_conv, signed, steps[0])
    width = stem.weights.shape[0]
    full = np.full(width, UNSIGNED_LEVELS, np.int64)
    body = []
    for index, block in enumerate(blocks):
        weights, weight_step = _rounded_weights(block)
        unit = steps[index] * weight_step
        body.append(_conv(block, weights, unit, steps[index + 1], full, identity_step=steps[index]))
    weights, weight_step = _rounded_weights(layout.head)
    # Picture outputs count 1/32 of an 8-bit level; state outputs count the tanh table's input step.
    out_steps = np.full(weights.shape[0], 1 / INTEGER_TANH_STEPS)
    out_steps[:PICTURE_OUTPUTS] = 1 / (UNSIGNED_LEVELS * INTEGER_RESIDUAL_STEPS)
    head = _conv(layout.head, weights, steps[-1] * weight_step, out_steps, full)
    state = weights.shape[0] > PICTURE_OUTPUTS
    return IntegerModel(
        input_channels=layout.in_channels, unsigned_words=unsigned_words, slots=slots, stem=stem,
        body=body, skip=_skip(steps[0], steps[-2], steps[-1]), head=head,
        tanh=tanh_table() if state else np.zeros(0, np.uint8), halo=halo, tile=tile,
    )  # fmt: skip


def quantise_model(
    config: ModelConfig, model: SequenceModel, tops: Sequence[float]
) -> IntegerModel:
    """The integer form of a luma-only flat `model` built from `config`, deployed as by default."""
    core = trained_core(model)
    layout = core.layout if isinstance(core, BoundedState) else core
    if not (config.luma_only and isinstance(layout, FlatLayout)):
        raise ValueError("only luma-only models of the flat layout have an integer form")
    return quantise(layout, tops, state_input_channels(config), DEFAULT_HALO, DEFAULT_TILE)
