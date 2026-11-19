"""Whole clips at full frame size, for evaluation."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from clearvisionnn.data.batch import Degraded, SequenceBatch
from clearvisionnn.data.crops import full_frame
from clearvisionnn.data.features import analyse
from clearvisionnn.data.index import IndexedClip
from clearvisionnn.data.tensors import clip_to_tensor, coding_to_tensor
from clearvisionnn.data.views import ViewMaker


class EvaluationDataset(Dataset[tuple[str, SequenceBatch]]):
    """Every frame of each clip, uncropped, with its first stored variant."""

    def __init__(self, clips: list[IndexedClip], views: ViewMaker) -> None:
        self._clips, self._views = clips, views

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, index: int) -> tuple[str, SequenceBatch]:
        """The clip's identifier and its frames as an unbatched sample."""
        clip = self._clips[index]
        record, scale = clip.record, self._views.input_scale
        height, width = record.height // scale, record.width // scale
        window = full_frame(record.frames, height, width)
        rng = np.random.default_rng(index)
        degraded = self._views.degraded(clip, window, 0, self._views.sample_live(rng), rng)
        view = Degraded.from_features(
            clip_to_tensor(degraded), analyse(degraded.y), coding_to_tensor(degraded)
        )
        return record.clip_id, SequenceBatch(clip_to_tensor(self._views.target(clip, window)), view)
