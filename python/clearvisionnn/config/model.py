"""Network and temporal-design settings (DESIGN.md section 8)."""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from clearvisionnn.config.base import Config, Positive


class Temporal(str, Enum):
    """How earlier frames are used (the variants of section 8.4)."""

    NONE = "none"  # per-frame
    BLEND = "blend"  # recursive convex blend with history
    # A hidden state and the previous output as inputs; luma-only.
    RECURRENT = "recurrent"


class Planes(str, Enum):
    """Which planes the network reconstructs; the rest are upscaled conventionally."""

    LUMA = "y"
    ALL = "yuv"


class ModelConfig(Config):
    """One network."""

    layout: str = "unet"  # registered layout: "unet", "flat" or "packed"
    size: str = "xl"  # rung of the size ladder
    planes: Planes = Planes.LUMA
    temporal: Temporal = Temporal.NONE
    # Temporal designs: warp history by the motion field before it is used.
    align: bool = True
    # Temporal designs: channels of hidden state carried between frames.
    state_channels: Positive = 8
    side_planes: bool = True
    # The network is shown luma alone (no chroma, no side-plane channels) and
    # reconstructs luma. Always so for RECURRENT, never for BLEND.
    luma_only: bool = False
    # Luma-only models: also show the network how each frame was coded.
    coding_planes: bool = False
    # RECURRENT only: also show the network how well motion predicts the frame.
    confidence_plane: bool = False

    @model_validator(mode="after")
    def _is_a_model_that_exists(self) -> ModelConfig:
        recurrent = self.temporal is Temporal.RECURRENT
        if self.luma_only and (self.planes is not Planes.LUMA or self.side_planes):
            raise ValueError("a luma-only model needs planes: y and side_planes: false")
        if self.luma_only and self.temporal is Temporal.BLEND:
            raise ValueError("the blend design is not luma-only")
        if recurrent and not self.luma_only:
            raise ValueError("the recurrent design needs luma_only: true")
        if self.coding_planes and not self.luma_only:
            raise ValueError("only a luma-only model takes coding planes")
        if self.confidence_plane and not recurrent:
            raise ValueError("only the recurrent design takes a confidence plane")
        return self
