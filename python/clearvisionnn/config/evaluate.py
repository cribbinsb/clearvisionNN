"""Evaluation settings (DESIGN.md section 10)."""

from __future__ import annotations

from enum import Enum

from clearvisionnn.config.base import Config, Positive


class Precision(str, Enum):
    """The arithmetic the model is evaluated in."""

    FP32 = "fp32"  # what the exported network and the runtimes use
    BF16 = "bf16"  # what training uses


class EvalConfig(Config):
    """What is scored and against what."""

    metrics: list[str] = ["psnr_y", "psnr_yuv", "ms_ssim", "lpips", "frame_diff_error"]
    baselines: list[str] = ["bicubic", "lanczos", "lanczos_sharpen"]
    # The baseline that gains are quoted against.
    reference_baseline: str = "bicubic"
    max_clips: Positive | None = None
    # Frames in the worst-window measure.
    window: Positive = 8
    bootstrap_samples: Positive = 2000
    precision: Precision = Precision.FP32
