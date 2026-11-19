"""The network's input contract, as the Rust pre-process and runtime define it.

The block size, the side-plane layout, the coding planes and the confidence
plane's scale are constants of the trained weights. They are defined once, in
Rust (`cvnn-preproc`, `cvnn-runtime`), and read here so that both the data layer
and the model definitions use them without one importing the other.
"""

from __future__ import annotations

import clearvisionnn_native

BLOCK_SIZE: int = clearvisionnn_native.BLOCK_SIZE  # side of an analysis block, in input samples
# The side planes, in channel order; each is followed by a flag channel saying
# whether it was supplied.
PLANE_NAMES = ("match_error", "match_error_in_noise_units", "noise_sigma", "still_run")
SIDE_CHANNELS: int = clearvisionnn_native.SIDE_CHANNELS
SR_SCALE = 2  # output size over input size
# Per-frame planes saying how a frame was coded: known, quantiser / MAX_QP, key frame.
CODING_CHANNELS: int = clearvisionnn_native.CODING_CHANNELS
MAX_QP: float = clearvisionnn_native.MAX_QP
# A block's mean motion-prediction error (samples in [0, 1]) at which the
# confidence plane is 1, and at which it reaches 0.
TRUSTED_ERROR: float = clearvisionnn_native.TRUSTED_ERROR
UNTRUSTED_ERROR: float = clearvisionnn_native.UNTRUSTED_ERROR

# The integer network's file format and fixed-point scales (docs/INTEGER.md).
INTEGER_MAGIC: int = clearvisionnn_native.INTEGER_MAGIC
INTEGER_SCHEMA_VERSION: int = clearvisionnn_native.INTEGER_SCHEMA_VERSION
INTEGER_HEADER_WORDS: int = clearvisionnn_native.INTEGER_HEADER_WORDS
INTEGER_MULTIPLIER_BITS: int = clearvisionnn_native.INTEGER_MULTIPLIER_BITS
INTEGER_RESIDUAL_STEPS: int = clearvisionnn_native.INTEGER_RESIDUAL_STEPS
INTEGER_TANH_STEPS: int = clearvisionnn_native.INTEGER_TANH_STEPS
INTEGER_STATE_BITS: int = clearvisionnn_native.INTEGER_STATE_BITS

if 2 * len(PLANE_NAMES) != SIDE_CHANNELS or len(PLANE_NAMES) != clearvisionnn_native.SIDE_PLANES:
    raise ImportError("clearvisionnn_native is out of date: rebuild crates/cvnn-py")
