"""The training and validation datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from clearvisionnn.config.train import DataConfig
from clearvisionnn.contract import PLANE_NAMES
from clearvisionnn.data.batch import Degraded, SequenceBatch
from clearvisionnn.data.crops import Window, centre_window, random_window
from clearvisionnn.data.features import analyse
from clearvisionnn.data.index import IndexedClip
from clearvisionnn.data.tensors import clip_to_tensor, coding_to_tensor
from clearvisionnn.data.views import ViewMaker
from clearvisionnn.video.clip import Clip

# Draws per pass over the training set: each clip is long enough to give many
# distinct windows.
WINDOWS_PER_CLIP = 16


def _join(first: Clip, second: Clip, cut: int) -> Clip:
    """`first` up to frame `cut`, then `second`: a scene cut."""
    coding = None
    if first.coding is not None and second.coding is not None:
        coding = np.concatenate([first.coding[:cut], second.coding[cut:]])
    y, u, v = (
        np.concatenate([a[:cut], b[cut:]]) for a, b in zip(first.planes, second.planes, strict=True)
    )
    return Clip(y, u, v, coding)


class TrainingDataset(Dataset[SequenceBatch]):
    """Random windows of random clips, freshly degraded on every draw."""

    def __init__(self, clips: list[IndexedClip], views: ViewMaker, config: DataConfig) -> None:
        self._clips, self._views, self._config = clips, views, config

    def __len__(self) -> int:
        return len(self._clips) * WINDOWS_PER_CLIP

    def __getitem__(self, index: int) -> SequenceBatch:
        # Seeded from torch's generator, which the loader seeds differently
        # per worker and per epoch.
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        clip = self._clips[index % len(self._clips)]
        windows, cut = self._windows(clip, rng)
        target = self._spliced(lambda w: self._views.target(clip, w), windows, cut)
        live = self._views.sample_live(rng)
        variants = rng.permutation(self._views.variant_count(clip))
        withheld = rng.uniform(size=len(PLANE_NAMES)) < self._config.side_plane_dropout

        def view(variant: int) -> Degraded:
            degraded = self._spliced(
                lambda w: self._views.degraded(clip, w, variant, live, rng), windows, cut
            )
            return Degraded.from_features(
                clip_to_tensor(degraded), analyse(degraded.y, withheld), coding_to_tensor(degraded)
            )

        alternate = view(int(variants[-1])) if self._config.noise_pair else None
        return SequenceBatch(clip_to_tensor(target), view(int(variants[0])), alternate)

    def _windows(self, clip: IndexedClip, rng: np.random.Generator) -> tuple[list[Window], int]:
        """One window, or two with the frame at which the sample cuts between them."""
        config, record = self._config, clip.record
        scale = self._views.input_scale
        size = (record.frames, record.height // scale, record.width // scale)
        first = random_window(rng, *size, config.frames, config.crop)
        if config.frames < 2 or rng.uniform() >= config.reset_share:
            return [first], 0
        second = random_window(rng, *size, config.frames, config.crop)
        return [first, second], int(rng.integers(1, config.frames))

    @staticmethod
    def _spliced(read, windows: list[Window], cut: int) -> Clip:  # type: ignore[no-untyped-def]
        if len(windows) == 1:
            return read(windows[0])
        return _join(read(windows[0]), read(windows[1]), cut)


class ValidationDataset(Dataset[SequenceBatch]):
    """The same centre window of each clip every time, so scores are comparable."""

    def __init__(self, clips: list[IndexedClip], views: ViewMaker, frames: int, crop: int) -> None:
        self._clips, self._views, self._frames, self._crop = clips, views, frames, crop

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, index: int) -> SequenceBatch:
        clip = self._clips[index]
        record, scale = clip.record, self._views.input_scale
        window = centre_window(
            record.frames, record.height // scale, record.width // scale, self._frames, self._crop
        )
        rng = np.random.default_rng(index)
        degraded = self._views.degraded(clip, window, 0, self._views.sample_live(rng), rng)
        view = Degraded.from_features(
            clip_to_tensor(degraded), analyse(degraded.y), coding_to_tensor(degraded)
        )
        return SequenceBatch(clip_to_tensor(self._views.target(clip, window)), view)
