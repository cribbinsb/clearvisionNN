"""Quick validation during training: fidelity and temporal error on fixed windows."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from clearvisionnn.data.batch import SequenceBatch
from clearvisionnn.data.tensors import LUMA
from clearvisionnn.losses.fidelity import downscaled
from clearvisionnn.losses.perceptual import LumaLpips
from clearvisionnn.models.sequence import SequenceModel
from clearvisionnn.train.rollout import roll
from clearvisionnn.train.stability import stability

MAX_SAMPLE = 255.0
# The names of the scores `validate` returns.
FIDELITY, TEMPORAL = "psnr_y", "frame_diff_error"
# Perceptual distance (lower is better), and PSNR after shrinking both
# pictures back to input size: how well the low frequencies are kept.
PERCEPTUAL, DOWNSCALED = "lpips", "psnr_y_downscaled"
# Whether a larger value of each selectable score is better.
HIGHER_IS_BETTER = {FIDELITY: True, DOWNSCALED: True, PERCEPTUAL: False}


def luma_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR of the luma channel, in dB, over everything given."""
    error = (output[..., LUMA, :, :] - target[..., LUMA, :, :]).float()
    return float(-10.0 * torch.log10((error**2).mean().clamp_min(1e-10)))


def frame_difference_error(output: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute mismatch of luma frame-to-frame change, in 0-255 units."""
    change = lambda x: x[:, 1:, LUMA] - x[:, :-1, LUMA]  # noqa: E731 - a local shorthand
    return float((change(output.float()) - change(target.float())).abs().mean() * MAX_SAMPLE)


@torch.no_grad()
def validate(
    model: SequenceModel,
    loader: DataLoader[SequenceBatch],
    device: torch.device,
    perceptual: LumaLpips,
) -> dict[str, float]:
    """Mean per-clip scores over the validation windows."""
    was_training = model.training
    model.eval()
    scores: dict[str, list[float]] = {}

    def record(name: str, value: float) -> None:
        scores.setdefault(name, []).append(value)

    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        output = roll(model, batch.view).clamp(0.0, 1.0)
        record(FIDELITY, luma_psnr(output, batch.target))
        record(TEMPORAL, frame_difference_error(output, batch.target))
        record(DOWNSCALED, luma_psnr(downscaled(output), downscaled(batch.target)))
        luma = [x[0, :, LUMA : LUMA + 1].float() for x in (output, batch.target)]
        record(PERCEPTUAL, float(perceptual(*luma)))
        for name, value in stability(luma[0][:, 0], luma[1][:, 0]).items():
            record(name, value)
    model.train(was_training)
    return {name: sum(values) / len(values) for name, values in scores.items()}
