"""Training settings."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from clearvisionnn.config.base import Config, NonNegativeFloat, Positive, PositiveFloat, Share

# Crops must tile the pre-process's 8x8 blocks, which also satisfies the
# network's two halvings and 4:2:0 chroma.
CROP_MULTIPLE = 8


class DataConfig(Config):
    """How training samples are drawn."""

    crop: Positive = 128  # input-resolution crop side; the target crop is twice this
    frames: Positive = 8  # consecutive frames per sample
    batch: Positive = 8
    workers: Annotated[int, Field(ge=0)] = 12
    # Share of samples that restart history part-way through (section 8.5).
    reset_share: Share = 0.15
    # Chance of each side plane being withheld, so the network works without it.
    side_plane_dropout: Share = 0.2
    # Draw two noise realisations per sample, for the consistency loss.
    noise_pair: bool = False

    @field_validator("crop")
    @classmethod
    def _crop_on_the_block_grid(cls, crop: int) -> int:
        if crop % CROP_MULTIPLE:
            raise ValueError(f"crop must be a multiple of {CROP_MULTIPLE}")
        return crop


class TrainConfig(Config):
    """One training run."""

    steps: Positive = 100_000
    lr: PositiveFloat = 4e-4
    lr_min: PositiveFloat = 1e-6
    warmup_steps: Positive = 1000
    weight_decay: NonNegativeFloat = 0.0
    grad_clip: PositiveFloat = 1.0
    # Mixed precision (bfloat16, which needs no gradient scaling).
    amp: bool = True
    seed: int = 0
    # Start from the best checkpoint of this run (of the same work directory)
    # instead of from scratch. A perceptual loss is used this way: its
    # gradient is hundreds of times a pixel loss's and stalls an untrained
    # network, so the model is trained on pixel loss first.
    init_from_run: str | None = None
    # Loss name -> weight; names are registered in clearvisionnn.losses.
    losses: dict[str, float] = Field(default_factory=lambda: {"charbonnier": 1.0})
    validate_every: Positive = 2000
    checkpoint_every: Positive = 2000
    log_every: Positive = 50
    validation_clips: Positive = 24
    # A checkpoint may become `best.pt` only while its flicker in still regions
    # is within this share above the original's own; quality is never bought with flicker
    # unnoticed (every validated checkpoint is kept as a candidate regardless).
    temporal_tolerance: NonNegativeFloat = 0.10
    # The validation score `best.pt` is chosen by: "psnr_y" for a model trained
    # to match the original, "lpips" for one trained to look like it.
    best_by: Literal["psnr_y", "lpips", "psnr_y_downscaled"] = "psnr_y"
    data: DataConfig = DataConfig()

    @model_validator(mode="after")
    def _consistent(self) -> TrainConfig:
        if "consistency" in self.losses and not self.data.noise_pair:
            raise ValueError("the consistency loss needs train.data.noise_pair: true")
        if self.lr_min > self.lr:
            raise ValueError("lr_min must not exceed lr")
        return self
