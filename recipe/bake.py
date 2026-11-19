"""Bake pools of training sequences, so the short training steps need not prepare data on the fly.

    python -m recipe.bake [--samples 4000] [--pools recipe/out/pools]

For each dataset under each chain the live loader is run for `--samples`
sequences, and the luma the network reads is stored as memory-mapped arrays
(about 0.5 MB a sequence) under `<pools>/<dataset>-<chain>/`. The sharpening
fine-tune and the 8-bit conversion train from these. The price is that their
degradations are not drawn afresh on every pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from clearvisionnn.data.batch import SequenceBatch
from recipe.clips import to_bytes
from recipe.data import POOL_DONE, live_loader
from recipe.options import BASE_BATCH, FRAMES
from recipe.sources import OUT, Source, add_source_arguments, sources_from

SAMPLES = 4000


def _arrays(batch: SequenceBatch) -> dict[str, np.ndarray]:
    """The parts of a batch a pool keeps: luma in 8 bits, motion, cut flags, coding information."""
    view = batch.view
    parts: dict[str, torch.Tensor] = {
        "target": to_bytes(batch.target[:, :, :1]),
        "frames": to_bytes(view.frames[:, :, :1]),
        "motion": view.motion.float(),
        "reset": view.reset,
        "coding": view.coding.float(),
    }
    return {name: tensor.numpy() for name, tensor in parts.items()}


def bake(source: Source, folder: Path, samples: int, workers: int) -> None:
    """Fill `folder` with `samples` sequences of `source`."""
    folder.mkdir(parents=True, exist_ok=True)
    loader = live_loader(source, FRAMES, BASE_BATCH, workers)
    pool: dict[str, np.memmap] = {}
    count = 0
    while count < samples:
        for batch in loader:
            parts = _arrays(batch)
            if not pool:
                pool = {
                    name: np.lib.format.open_memmap(
                        folder / f"{name}.npy", "w+", part.dtype, (samples, *part.shape[1:])
                    )
                    for name, part in parts.items()
                }
            take = min(len(batch.target), samples - count)
            for name, part in parts.items():
                pool[name][count : count + take] = part[:take]
            count += take
            if count >= samples:
                break
    for array in pool.values():
        array.flush()
    (folder / POOL_DONE).write_text(str(samples))
    print(f"baked {folder} ({samples} sequences)", flush=True)


def main() -> None:
    """Bake every pool that is not there yet."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arguments(parser)
    parser.add_argument("--samples", type=int, default=SAMPLES, help="sequences per pool")
    parser.add_argument("--pools", type=Path, default=OUT / "pools")
    parser.add_argument("--workers", type=int, default=8, help="loader processes")
    arguments = parser.parse_args()
    for source in sources_from(arguments):
        folder = arguments.pools / source.name
        if (folder / POOL_DONE).exists():
            print(f"have {folder}", flush=True)
        else:
            bake(source, folder, arguments.samples, arguments.workers)


if __name__ == "__main__":
    main()
