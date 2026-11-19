"""Training batches: live loaders over the pipeline's clips, pre-baked pools, and augmentation."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvisionnn.config.load import load_config
from clearvisionnn.data.batch import Degraded, SequenceBatch, collate
from clearvisionnn.data.dataset import TrainingDataset

# Private to the package: its public loaders take every clip of a split, and
# this recipe needs the clip index itself to train on a bounded subset.
from clearvisionnn.data.loaders import _views
from clearvisionnn.pipeline.records import Split
from recipe.sources import Source

# Training clips taken from any one dataset. Every clip read is decoded whole
# into the pipeline's frame cache, so this bounds the disk a run takes.
MAX_CLIPS = 250
# The subset is the same in every run and every step of the recipe.
SUBSET_SEED = 0
# A pool has no natural end; a pass over the pools is this many batches.
POOL_PASS = 1000
POOL_ARRAYS = ("target", "frames", "motion", "reset", "coding")
# Written last into a pool's folder, so a half-baked pool is never read.
POOL_DONE = "done"
MAX_SAMPLE = 255
FLIP_CHANCE = 0.5
Pool = np.ndarray[Any, Any]


def live_loader(source: Source, frames: int, batch: int, workers: int) -> DataLoader[SequenceBatch]:
    """Shuffled, freshly cropped and degraded training batches of one dataset under one chain."""
    config = load_config(source.config)
    # No spliced scene cuts: every training sequence is one continuous shot.
    changes = {"reset_share": 0.0, "frames": frames, "batch": batch, "workers": workers}
    data = config.train.data.model_copy(update=changes)
    index, views = _views(config, Split.TRAIN)
    clips = index.clips
    if len(clips) > MAX_CLIPS:
        clips = random.Random(SUBSET_SEED).sample(clips, MAX_CLIPS)
    return DataLoader(
        TrainingDataset(clips, views, data),
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=workers > 0,
        pin_memory=True,
    )


class Alternating:
    """One batch from each loader in turn, until the shortest runs out."""

    def __init__(self, loaders: list[Iterable[SequenceBatch]]) -> None:
        self._loaders = loaders

    def __iter__(self) -> Iterator[SequenceBatch]:
        for group in zip(*self._loaders, strict=False):
            yield from group


def live_batches(
    sources: list[Source], frames: int, batch: int, workers: int
) -> Iterable[SequenceBatch]:
    """Live batches alternating over `sources`, which share `workers` loader processes."""
    each = max(1, workers // len(sources))
    return Alternating([live_loader(source, frames, batch, each) for source in sources])


class Pools:
    """Batches drawn at random from the baked pools (`recipe.bake`), taking each source in turn.

    The pools hold luma only, which is all this network reads.
    """

    def __init__(
        self, root: Path, sources: list[Source], frames: int, batch: int, device: torch.device
    ) -> None:
        self._batch, self._device = batch, device
        self._pools = [self._open(root / source.name, frames) for source in sources]

    @staticmethod
    def _open(folder: Path, frames: int) -> dict[str, Pool]:
        if not (folder / POOL_DONE).exists():
            raise FileNotFoundError(f"{folder} is not baked (run `python -m recipe.bake`)")
        pool = {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in POOL_ARRAYS}
        if pool["target"].shape[1] != frames:
            raise ValueError(f"{folder} holds sequences of {pool['target'].shape[1]} frames")
        return pool

    def __iter__(self) -> Iterator[SequenceBatch]:
        for i in range(POOL_PASS):
            pool = self._pools[i % len(self._pools)]
            # Sorted, so the memory-mapped reads run forwards through the file.
            rows = np.sort(np.random.choice(len(pool["target"]), self._batch, replace=False))

            def get(name: str, pool: dict[str, Pool] = pool, rows: Pool = rows) -> torch.Tensor:
                return torch.from_numpy(pool[name][rows]).to(self._device, non_blocking=True)

            motion = get("motion")
            # The side planes are not baked: the network does not take them.
            side = motion.new_zeros(*motion.shape[:2], 1, *motion.shape[3:])
            view = Degraded(
                get("frames").float().div(MAX_SAMPLE), motion, side, get("reset"), get("coding")
            )
            yield SequenceBatch(get("target").float().div(MAX_SAMPLE), view)


def augment(batch: SequenceBatch) -> SequenceBatch:
    """A random mirror and transpose of a whole batch.

    Pictures, block grids and the motion vectors on them are turned together,
    so the motion still brings each previous frame onto the current one.
    """
    view = batch.view
    target, frames, motion, side = batch.target, view.frames, view.motion, view.side

    def negate(dx: float, dy: float) -> torch.Tensor:
        return motion.new_tensor([dx, dy]).view(1, 1, 2, 1, 1)

    if random.random() < FLIP_CHANCE:
        target, frames, side = target.flip(-1), frames.flip(-1), side.flip(-1)
        motion = motion.flip(-1) * negate(-1.0, 1.0)
    if random.random() < FLIP_CHANCE:
        target, frames, side = target.flip(-2), frames.flip(-2), side.flip(-2)
        motion = motion.flip(-2) * negate(1.0, -1.0)
    if random.random() < FLIP_CHANCE and target.shape[-1] == target.shape[-2]:
        target, frames, side = (x.transpose(-1, -2) for x in (target, frames, side))
        motion = motion.transpose(-1, -2)[:, :, [1, 0]]
    turned = Degraded(
        frames.contiguous(), motion.contiguous(), side.contiguous(), view.reset, view.coding
    )
    return SequenceBatch(target.contiguous(), turned)
