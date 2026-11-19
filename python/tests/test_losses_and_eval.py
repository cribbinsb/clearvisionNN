"""Losses, metrics and the statistics evaluation relies on."""

import numpy as np
import pytest
import torch

from clearvisionnn.eval.baselines import BASELINES
from clearvisionnn.eval.colour import yuv_to_rgb
from clearvisionnn.eval.metrics import METRICS
from clearvisionnn.eval.summary import bootstrap_mean, summarise
from clearvisionnn.losses import CompositeLoss, Rollout


def _rollout(alternate: bool = False) -> Rollout:
    torch.manual_seed(0)
    target = torch.rand(2, 4, 1, 16, 16)
    output = target + 0.1 * torch.randn_like(target)
    return Rollout(output, target, output + 0.05 if alternate else None)


def test_losses_vanish_for_a_perfect_output() -> None:
    target = torch.rand(2, 4, 1, 16, 16)
    loss = CompositeLoss({"charbonnier": 1.0, "temporal_difference": 1.0, "consistency": 1.0})
    total, parts = loss(Rollout(target, target, target))
    assert float(total) < 0.01
    assert set(parts) == {"charbonnier", "temporal_difference", "consistency"}


def test_temporal_loss_ignores_a_constant_offset_but_not_flicker() -> None:
    target = torch.rand(1, 4, 1, 16, 16)
    loss = CompositeLoss({"temporal_difference": 1.0})
    offset, _ = loss(Rollout(target + 0.2, target))
    flicker = target.clone()
    flicker[:, ::2] += 0.2
    flickering, _ = loss(Rollout(flicker, target))
    assert float(offset) < 0.01 < float(flickering)


def test_consistency_needs_the_alternate_view() -> None:
    loss = CompositeLoss({"consistency": 1.0})
    assert loss.needs_alternate
    with pytest.raises(ValueError, match="noise_pair"):
        loss(_rollout())
    assert float(loss(_rollout(alternate=True))[0]) > 0


def test_metrics_rank_a_better_output_higher() -> None:
    torch.manual_seed(0)
    target = torch.rand(3, 3, 192, 192)
    good, bad = target + 0.01 * torch.randn_like(target), target + 0.1 * torch.randn_like(target)
    device = torch.device("cpu")
    for name in ("psnr_y", "psnr_yuv", "ms_ssim", "frame_diff_error"):
        metric = METRICS.get(name)(device)
        better = metric(good.clamp(0, 1), target).mean() > metric(bad.clamp(0, 1), target).mean()
        assert bool(better) == metric.higher_is_better, name


def test_baselines_double_the_size() -> None:
    frames = torch.rand(2, 3, 16, 24)
    for name in BASELINES:
        assert BASELINES.get(name)(frames).shape == (2, 3, 32, 48), name


def test_grey_stays_grey_in_rgb() -> None:
    grey = torch.tensor([126 / 255, 128 / 255, 128 / 255]).view(1, 3, 1, 1)
    rgb = yuv_to_rgb(grey)
    assert torch.allclose(rgb[:, 0], rgb[:, 1]) and torch.allclose(rgb[:, 1], rgb[:, 2])


def test_worst_window_follows_the_metric_direction() -> None:
    values = np.array([10.0, 10.0, 2.0, 4.0, 10.0])
    assert summarise(values, 2, higher_is_better=True).worst_window == 3.0
    assert summarise(values, 2, higher_is_better=False).worst_window == 10.0


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = np.array([0.9, 1.0, 1.1, 1.2, 0.8])
    interval = bootstrap_mean(values, ["a", "b", "c", "d", "e"], samples=500)
    assert interval.low < interval.mean < interval.high
    assert interval.excludes_zero


def test_clips_of_one_source_file_widen_the_interval() -> None:
    values = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    independent = bootstrap_mean(values, list("abcdef"), samples=2000)
    grouped = bootstrap_mean(values, list("aaabbb"), samples=2000)
    assert grouped.high - grouped.low > independent.high - independent.low
