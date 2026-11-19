"""The measures a model is scored by: fidelity, structure, detail and flicker."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

# Private to the package: the flicker ratios here are built from the same two
# pieces as its stability measures, and must stay in step with them.
from clearvisionnn.train.stability import (
    MATCHED_THRESHOLD,
    MAX_SAMPLE,
    STILL_THRESHOLD,
    _along_motion,
    _local_mean,
)

# Leading frames left out of every score: the recurrence is still filling its history.
WARM_UP_FRAMES = 2
# Floor of the mean squared error, so identical pictures score a finite PSNR.
MIN_ERROR = 1e-10
# The usual SSIM window and stabilising constants (Wang et al., 2004) for samples in [0, 1].
SSIM_WINDOW, SSIM_SIGMA = 11, 1.5
SSIM_C1, SSIM_C2 = 0.01**2, 0.03**2
# Guards the ratios against an empty mask or a featureless original.
TINY = 1e-9


def psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB of pictures in [0, 1]."""
    return float(-10 * torch.log10(((output.float() - target) ** 2).mean().clamp_min(MIN_ERROR)))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean structural similarity of (frames, h, w) luma in [0, 1]."""
    offsets = torch.arange(SSIM_WINDOW, device=a.device) - (SSIM_WINDOW - 1) / 2
    window = torch.exp(-(offsets**2) / (2 * SSIM_SIGMA**2))
    window = (window / window.sum())[None, None]

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.conv2d(x[:, None], window[..., None]), window[:, :, None])[:, 0]

    mean_a, mean_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mean_a**2, blur(b * b) - mean_b**2
    covariance = blur(a * b) - mean_a * mean_b
    score = ((2 * mean_a * mean_b + SSIM_C1) * (2 * covariance + SSIM_C2)) / (
        (mean_a**2 + mean_b**2 + SSIM_C1) * (var_a + var_b + SSIM_C2)
    )
    return float(score.mean())


def detail(luma: torch.Tensor) -> float:
    """Mean gradient magnitude of (frames, h, w) luma: how much fine structure it carries."""
    across = (luma[:, :, 1:] - luma[:, :, :-1]).abs().mean()
    down = (luma[:, 1:] - luma[:, :-1]).abs().mean()
    return float(across + down)


def detail_ratio(output: torch.Tensor, target: torch.Tensor) -> float:
    """The output's detail over the original's: below 1 is softer, above 1 is invented texture."""
    return detail(output) / max(detail(target), TINY)


def psnr_at_input_size(output: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR once both are averaged back to input size: fidelity to what the input did carry."""

    def down(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x[:, None], 2)[:, 0]

    return psnr(down(output), down(target))


def flicker(
    output: torch.Tensor, target: torch.Tensor, flow: torch.Tensor, continuous: torch.Tensor
) -> dict[str, float]:
    """Frame-to-frame change of (frames, h, w) output luma beside the original's own.

    `flow` is the original's motion and `continuous` is 1 for each frame after
    the first that does not follow a scene cut. The means are the stability
    measures of `clearvisionnn.train.stability`; the sums let the caller form
    output-over-original ratios across clips, separately where the motion
    predicts the original and where it does not.
    """

    def change(x: torch.Tensor) -> torch.Tensor:
        return (x[1:] - x[:-1]).abs() * MAX_SAMPLE

    def mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float((values * mask).sum() / mask.sum().clamp_min(1))

    def total(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float((values * mask).sum())

    still = (_local_mean(change(target)) < STILL_THRESHOLD).float()
    along, along_original = _along_motion(output, flow), _along_motion(target, flow)
    matched = (_local_mean(along_original) < MATCHED_THRESHOLD).float()
    predicted, unpredicted = matched * continuous, (1 - matched) * continuous
    return {
        "still": mean(change(output), still),
        "still_orig": mean(change(target), still),
        "motion": mean(along, predicted),
        "motion_orig": mean(along_original, predicted),
        "predicted_sum": total(along, predicted),
        "predicted_sum_orig": total(along_original, predicted),
        "unpredicted_sum": total(along, unpredicted),
        "unpredicted_sum_orig": total(along_original, unpredicted),
        "unpredicted_samples": float(unpredicted.sum()),
        "samples": float(continuous.sum() * along[0].numel()),
    }
