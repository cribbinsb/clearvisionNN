"""Settings of the data pipeline's stages."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, model_validator

from clearvisionnn.config.base import Config, NonNegativeFloat, Positive, PositiveFloat, Share


class SegmentConfig(Config):
    """How clips are cut from source files."""

    frames: Positive = 64
    # At most this many clips per source file, spread evenly along it.
    max_per_file: Positive = 4
    # Sources with a higher frame rate are decimated to at most this.
    max_fps: PositiveFloat = 30.0


class GradeConfig(Config):
    """How far a source must be downscaled before it counts as ground truth.

    Compression damage is judged by bits per pixel per frame; each entry of
    `downscale_by_bpp` is (minimum bits per pixel, downscale factor), best first.
    """

    downscale_by_bpp: list[tuple[NonNegativeFloat, Positive]] = [(0.30, 2), (0.08, 3), (0.03, 4)]
    # Targets smaller than this (shorter side, after downscaling) are not made.
    min_target_side: Positive = 352
    # Targets are cropped to a multiple of this, so 2x pairs align on every grid in use.
    align: Positive = 32


class CleanConfig(Config):
    """Which extracted clips are rejected."""

    # Mean absolute luma change between consecutive frames (0-255, at low
    # resolution) above which a clip contains a cut.
    cut_threshold: PositiveFloat = 30.0
    # Clips whose mean frame-to-frame change is below this are frozen.
    frozen_threshold: NonNegativeFloat = 0.02
    # Clips whose mean gradient magnitude is below this hold no detail to learn.
    min_sharpness: NonNegativeFloat = 2.0


class SplitConfig(Config):
    """Shares of source files held out. Assigned once, by file, at ingest."""

    validation: Share = 0.05
    test: Share = 0.10
    seed: int = 0

    @model_validator(mode="after")
    def _leaves_training_data(self) -> SplitConfig:
        if self.validation + self.test >= 1:
            raise ValueError("validation and test shares must leave some files for training")
        return self


class ModuleSpec(Config):
    """One degradation module: its registered name and its parameters."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class DegradeConfig(Config):
    """The degradation chain, applied in order from clean target to network input."""

    chain: Annotated[list[ModuleSpec], Field(min_length=1)]
    # Stored variants per clip of the chain's offline part (encoder round trips).
    variants_per_clip: Positive = 2
    seed: int = 0


class VerifyConfig(Config):
    """The pairing check that closes the pipeline."""

    clips_per_source: Positive = 60
    # Largest mean absolute luma difference (0-255) between a target and its
    # conventionally upscaled input. The heaviest configured degradation
    # measures under 20; a wrong frame or transposed picture measures 30 to 100.
    max_error: PositiveFloat = 25.0


class PipelineConfig(Config):
    """Everything between raw video and training samples."""

    segment: SegmentConfig = SegmentConfig()
    grade: GradeConfig = GradeConfig()
    clean: CleanConfig = CleanConfig()
    split: SplitConfig = SplitConfig()
    degrade: DegradeConfig
    verify: VerifyConfig = VerifyConfig()
    workers: Positive = 8
    # The pipeline refuses to start a stage when its work directory already holds this much.
    disk_budget_gb: PositiveFloat = 80.0
