"""The model contract a runtime needs beside the weights (`cvnn_runtime::ModelSpec`)."""

from __future__ import annotations

from pydantic import BaseModel

from clearvisionnn.config.model import ModelConfig
from clearvisionnn.contract import INTEGER_SCHEMA_VERSION, INTEGER_STATE_BITS
from clearvisionnn.models.layouts import Layout
from clearvisionnn.models.per_frame import reconstructed_channels
from clearvisionnn.train.provenance import Provenance

CONTRACT_FILE = "model.json"
ONNX_FILE = "model.onnx"
INTEGER_FILE = "model.i8"


# Raise when the meaning of any field, or of the network's input channels, changes.
SCHEMA_VERSION = 2
# The colour the runtime converts to and from: 8-bit BT.709, limited range.
COLOUR = "bt709-limited"
# Input width and height must be multiples of this (the network halves twice).
SIZE_MULTIPLE = 4


class Origin(BaseModel):
    """Where exported weights came from."""

    step: int
    checkpoint_hash: str
    provenance: Provenance


class IntegerContract(BaseModel):
    """What a model in 8-bit integer form adds to its contract (docs/INTEGER.md)."""

    schema_version: int = INTEGER_SCHEMA_VERSION
    file: str = INTEGER_FILE
    sha256: str
    # Bits of each recurrent-state value: a field, so a later model can carry a finer state.
    state_bits: int = INTEGER_STATE_BITS
    # Each tile is given this much real neighbourhood, and tiles are this wide:
    # both are part of the model's arithmetic, so every device computes the same picture.
    halo: int
    tile: int


class Contract(BaseModel):
    """What the exported network expects and produces, and where it came from."""

    schema_version: int = SCHEMA_VERSION
    colour: str = COLOUR
    size_multiple: int = SIZE_MULTIPLE
    name: str
    layout: str
    size: str
    step: int
    planes: int
    temporal: str  # "none", "blend" or "recurrent"
    align: bool
    state_channels: int
    side_planes: bool
    # The network input is luma alone, optionally with the planes below.
    luma_only: bool
    coding_planes: bool
    confidence_plane: bool
    input_channels: int
    fine_channels: int
    parameters: int
    # The kind of network file beside this contract: "onnx" (`model.onnx`, float
    # arithmetic) or "integer" (`model.i8`, 8-bit). It decides what can run the model.
    network: str = "onnx"
    # Arithmetic of the network: "fp32" or "fp16" inside a graph whose inputs
    # and outputs are 32-bit either way, or "int8".
    precision: str = "fp32"
    # Identity of the files and of the run behind them.
    onnx_sha256: str | None = None
    checkpoint_hash: str
    provenance: Provenance
    # The integer form's own contract; present exactly when `network` is "integer".
    integer: IntegerContract | None = None


def contract_for(
    name: str,
    config: ModelConfig,
    layout: Layout,
    origin: Origin,
    *,
    onnx_sha256: str | None = None,
    half: bool = False,
    integer: IntegerContract | None = None,
) -> Contract:
    """The contract of a model built from `config` whose core `layout` was exported.

    Give `onnx_sha256` (and `half`) for a graph, or `integer` for the integer form.
    """
    if (onnx_sha256 is None) == (integer is None):
        raise ValueError("a contract describes one network file: a graph or the integer form")
    return Contract(
        name=name,
        layout=config.layout,
        size=config.size,
        step=origin.step,
        planes=reconstructed_channels(config.planes),
        temporal=config.temporal.value,
        align=config.align,
        state_channels=config.state_channels,
        side_planes=config.side_planes,
        luma_only=config.luma_only,
        coding_planes=config.coding_planes,
        confidence_plane=config.confidence_plane,
        input_channels=layout.in_channels,
        fine_channels=layout.fine_channels,
        parameters=sum(p.numel() for p in layout.parameters()),
        network="integer" if integer else "onnx",
        precision="int8" if integer else "fp16" if half else "fp32",
        onnx_sha256=onnx_sha256,
        checkpoint_hash=origin.checkpoint_hash,
        provenance=origin.provenance,
        integer=integer,
    )
