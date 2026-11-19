"""Score models clip by clip on a held-out dataset, beside the conventional upscale.

    python -m recipe.score --config configs/xiph_ncodec_blend.yaml CHECKPOINT [CHECKPOINT ...]

Appends one JSON line per model and source file to `--out`: PSNR, LPIPS,
SSIM, detail and flicker against the original, and VMAF, MS-SSIM and XPSNR
when a `vmaf` program is given (`--vmaf-binary` and `--vmaf-model`, or the
VMAF_BINARY and VMAF_MODEL environment variables). Only meaningful on a
dataset the models never trained on. Clips come from the dataset's train and
validation splits, never its sealed test split.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from clearvisionnn.config.load import load_config
from clearvisionnn.config.run import RunConfig
from clearvisionnn.data.batch import Degraded, SequenceBatch, collate
from clearvisionnn.data.dataset import ValidationDataset
from clearvisionnn.data.features import analyse

# `_views` is private to the package: its public loaders cover one split and
# a capped number of clips, and scoring spreads its clips over two splits.
from clearvisionnn.data.loaders import VALIDATION_CROP, VALIDATION_FRAMES, _views
from clearvisionnn.losses.perceptual import LumaLpips
from clearvisionnn.models.resample import block_field_to_flow, upscale
from clearvisionnn.pipeline.records import Split
from recipe import metrics
from recipe.clips import cached, pack, to_bytes, unpack
from recipe.metrics import WARM_UP_FRAMES
from recipe.model import Model, rollout
from recipe.options import load_checkpoint
from recipe.sources import OUT
from recipe.vmaf import NAMES as VMAF_NAMES
from recipe.vmaf import VmafTool

# Clips scored, evenly spread over the dataset; 0 scores every clip.
CLIPS = 100
# The pipeline's record of source files, in a dataset's work directory.
FILES = "files.jsonl"
BASELINE = "bicubic"
# Per-clip measures reported as their mean over a source file's clips.
MEANS = (
    "psnr",
    "still",
    "motion",
    "motion_orig",
    "still_orig",
    "lpips",
    "ssim",
    "psnr_down",
    "detail",
)


class Bicubic(nn.Module):
    """The conventional upscale: what every model must beat."""

    def forward(self, view: Degraded) -> torch.Tensor:
        """The upscaled luma of every frame."""
        frames = range(view.frames.shape[1])
        return torch.stack([upscale(view.frames[:, t])[:, :1] for t in frames], 1)


def prepare(config: RunConfig, limit: int) -> list[tuple[str, dict[str, torch.Tensor]]]:
    """The clips to score, each with its source file's name: read, degraded and analysed."""
    names = {}
    with (config.data_dir / FILES).open() as lines:
        for line in lines:
            record = json.loads(line)
            names[record["file_id"]] = Path(record["path"]).stem
    chosen: list[tuple[str, ValidationDataset, int]] = []
    for split in (Split.TRAIN, Split.VALIDATION):
        index, views = _views(config, split)
        data = ValidationDataset(index.clips, views, VALIDATION_FRAMES, VALIDATION_CROP)
        chosen += [(names[clip.record.file_id], data, i) for i, clip in enumerate(index.clips)]
    if limit and len(chosen) > limit:
        last = len(chosen) - 1
        chosen = [chosen[round(j * last / (limit - 1))] for j in range(limit)]
    prepared = []
    for name, data, i in chosen:
        batch = collate([data[i]])
        # The original's own motion, for the flicker measures: the same for every model.
        original = analyse(to_bytes(batch.target[0, WARM_UP_FRAMES:, 0]).numpy())
        stored = {**pack(batch), "target_motion": original.motion, "target_reset": original.reset}
        prepared.append((name, stored))
    return prepared


class Scorer:
    """Every measure of one model's output on one clip."""

    def __init__(self, device: torch.device, vmaf: VmafTool | None) -> None:
        self._lpips = LumaLpips("alex").to(device)
        self._vmaf = vmaf

    def __call__(
        self, output: torch.Tensor, batch: SequenceBatch, stored: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Scores of (1, frames, 1, H, W) `output` for the clip in `batch`."""
        device = output.device
        target = batch.target[:, :, :1]
        flow = block_field_to_flow(stored["target_motion"].to(device), *target.shape[-2:])
        continuous = (~stored["target_reset"].to(device)[1:, None, None]).float()
        luma, original = output[0, WARM_UP_FRAMES:, 0], target[0, WARM_UP_FRAMES:, 0]
        scores = {
            "psnr": metrics.psnr(output[:, WARM_UP_FRAMES:], target[:, WARM_UP_FRAMES:]),
            **metrics.flicker(luma, original, flow, continuous),
            "lpips": float(self._lpips(luma[:, None], original[:, None])),
            "ssim": metrics.ssim(luma, original),
            "psnr_down": metrics.psnr_at_input_size(luma, original),
            "detail": metrics.detail_ratio(luma, original),
        }
        if self._vmaf is not None:
            # Chroma is the conventional upscale of the input's, as a player would show it.
            chroma = upscale(batch.view.frames[0, WARM_UP_FRAMES:])[:, 1:]
            shown = torch.cat([output[0, WARM_UP_FRAMES:], chroma], 1)
            scores.update(self._vmaf.score(shown, batch.target[0, WARM_UP_FRAMES:]))
        return scores


def summarise(clips: list[dict[str, float]]) -> dict[str, float]:
    """One source file's scores from its clips': means, and flicker ratios pooled over the clips."""

    def total(name: str) -> float:
        return sum(clip[name] for clip in clips)

    def ratio(above: str, below: str) -> float:
        return total(above) / max(total(below), metrics.TINY)

    means = [*MEANS, *(name for name in VMAF_NAMES if name in clips[0])]
    return {
        "clips": len(clips),
        **{name: total(name) / len(clips) for name in means},
        # Output flicker over the original's, where its motion predicts it and where it does not.
        "pred_ratio": ratio("predicted_sum", "predicted_sum_orig"),
        "unpred_ratio": ratio("unpredicted_sum", "unpredicted_sum_orig"),
        "unpred_share": ratio("unpredicted_samples", "samples"),
    }


def main() -> None:
    """Score the checkpoints named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, required=True, help="the dataset under a chain")
    parser.add_argument("--clips", type=int, default=CLIPS, help="clips to score; 0 for all")
    parser.add_argument("--out", type=Path, default=OUT / "scores.jsonl")
    parser.add_argument("--cache", type=Path, default=OUT / "eval", help="prepared clips")
    parser.add_argument("--vmaf-binary", type=Path, default=os.environ.get("VMAF_BINARY"))
    parser.add_argument("--vmaf-model", type=Path, default=os.environ.get("VMAF_MODEL"))
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    arguments = parser.parse_args()
    device: torch.device = arguments.device
    vmaf = None
    if arguments.vmaf_binary and arguments.vmaf_model:
        vmaf = VmafTool(arguments.vmaf_binary, arguments.vmaf_model)
    else:
        print("no vmaf program given: scoring without VMAF, MS-SSIM and XPSNR", flush=True)

    models: dict[str, nn.Module] = {BASELINE: Bicubic()}
    for path in arguments.checkpoints:
        saved = load_checkpoint(path)
        models[path.stem] = Model(saved.options)
        models[path.stem].load_state_dict(saved.weights)
    dataset = arguments.config.stem
    kept = arguments.cache / f"{dataset}-{arguments.clips or 'all'}.pt"
    prepared = cached(kept, lambda: prepare(load_config(arguments.config), arguments.clips))
    score = Scorer(device, vmaf)
    table: dict[str, dict[str, list[dict[str, float]]]] = {
        name: collections.defaultdict(list) for name in models
    }
    with torch.no_grad():
        for name, model in models.items():
            model.to(device).eval()
            for file, stored in prepared:
                batch = unpack(stored).to(device)
                table[name][file].append(score(rollout(model, batch), batch, stored))
            print(f"scored {name} on {dataset}", flush=True)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("a") as out:
        for name, files in table.items():
            for file, clips in files.items():
                row: dict[str, Any] = {"dataset": dataset, "model": name, "file": file}
                out.write(json.dumps({**row, **summarise(clips)}) + "\n")


if __name__ == "__main__":
    main()
