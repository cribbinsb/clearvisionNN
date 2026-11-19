"""Producing a clip's degraded view for a window: stored variant plus the live chain."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from clearvisionnn.contract import SR_SCALE
from clearvisionnn.data.crops import Window
from clearvisionnn.data.index import ClipIndex, IndexedClip
from clearvisionnn.pipeline.degrade.chain import ChainParams, DegradeChain
from clearvisionnn.video.clip import Clip
from clearvisionnn.video.reader import read_frames


class ViewMaker:
    """Reads targets and degraded inputs for windows of indexed clips."""

    def __init__(
        self,
        index: ClipIndex,
        chain: DegradeChain,
        read: Callable[[Path, int, int], Clip] = read_frames,
    ) -> None:
        """`read(path, start, count)` fetches frames of a stored clip."""
        self._index, self._read = index, read
        self._stored, self._live = chain.split()
        if chain.scale != SR_SCALE:
            raise ValueError(f"the chain scales by {chain.scale}, the network by {SR_SCALE}")
        missing = (
            [c.record.clip_id for c in index.clips if not c.variants] if len(self._stored) else []
        )
        if missing:
            # Reading the target in a variant's place would feed the network
            # input at the wrong scale with nothing to show for it.
            raise ValueError(
                f"{len(missing)} clips have no stored variants (e.g. {missing[0]}); "
                "run the pipeline for this degradation chain"
            )

    @property
    def input_scale(self) -> int:
        """Target size over input size."""
        return SR_SCALE

    def variant_count(self, clip: IndexedClip) -> int:
        """Stored variants to choose from (1 when the whole chain runs live)."""
        return len(clip.variants) if len(self._stored) else 1

    def target(self, clip: IndexedClip, window: Window) -> Clip:
        """The clean target for `window`."""
        path = self._index.resolve(clip.record.target)
        return self._read(path, window.start, window.frames).crop(*window.scaled(SR_SCALE))

    def sample_live(self, rng: np.random.Generator) -> ChainParams:
        """Per-clip parameters of the live part of the chain."""
        return self._live.sample(rng)

    def degraded(
        self,
        clip: IndexedClip,
        window: Window,
        variant: int,
        live_params: ChainParams,
        rng: np.random.Generator,
    ) -> Clip:
        """The network input for `window`: variant `variant`, then the live chain."""
        record = clip.variants[variant] if len(self._stored) else None
        path = self._index.resolve(record.path if record else clip.record.target)
        stored = self._read(path, window.start, window.frames)
        if record and record.coding:
            coding = np.load(self._index.resolve(record.coding))
            stored = stored.with_coding(coding[window.start : window.start + window.frames])
        # The stored part has already scaled by its own factor; the live part
        # scales by the rest, so read that much more than the window.
        return self._live.apply(stored.crop(*window.scaled(self._live.scale)), live_params, rng)
