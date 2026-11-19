"""Building data loaders from a run configuration."""

from __future__ import annotations

from torch.utils.data import DataLoader

from clearvisionnn.config.run import RunConfig
from clearvisionnn.data.batch import SequenceBatch, collate
from clearvisionnn.data.dataset import TrainingDataset, ValidationDataset
from clearvisionnn.data.index import ClipIndex
from clearvisionnn.data.views import ViewMaker
from clearvisionnn.pipeline.degrade.chain import DegradeChain
from clearvisionnn.pipeline.records import Split
from clearvisionnn.pipeline.workdir import Workdir
from clearvisionnn.video.frame_cache import FrameCache

# Validation windows are larger than training crops: closer to real frames.
VALIDATION_CROP = 256
VALIDATION_FRAMES = 16
# Decoded copies of the clips the loaders read, inside the work directory.
FRAME_CACHE = "cache"


def _views(config: RunConfig, split: Split) -> tuple[ClipIndex, ViewMaker]:
    workdir = Workdir(config.data_dir, config.pipeline.disk_budget_gb)
    index = ClipIndex(workdir, split, config.pipeline.degrade)
    chain = DegradeChain.from_specs(config.pipeline.degrade.chain)
    return index, ViewMaker(index, chain, FrameCache(workdir.path(FRAME_CACHE)).read)


def training_loader(config: RunConfig) -> DataLoader[SequenceBatch]:
    """Shuffled, freshly degraded training batches."""
    index, views = _views(config, Split.TRAIN)
    data = config.train.data
    dataset = TrainingDataset(index.clips, views, data)
    if len(dataset) < data.batch:
        raise ValueError(
            f"{len(index.clips)} training clips give {len(dataset)} samples per pass, "
            f"fewer than one batch of {data.batch}"
        )
    return DataLoader(
        dataset,
        batch_size=data.batch,
        shuffle=True,
        num_workers=data.workers,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=data.workers > 0,
        pin_memory=True,
    )


def validation_loader(config: RunConfig) -> DataLoader[SequenceBatch]:
    """Fixed validation windows, one clip per batch (clips differ in size)."""
    index, views = _views(config, Split.VALIDATION)
    clips = index.clips[: config.train.validation_clips]
    return DataLoader(
        ValidationDataset(clips, views, VALIDATION_FRAMES, VALIDATION_CROP),
        batch_size=1,
        num_workers=min(4, config.train.data.workers),
        collate_fn=collate,
    )
