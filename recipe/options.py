"""The settings a checkpoint records, and reading and writing checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

# The state the network carries from frame to frame, in channels at input resolution.
STATE_CHANNELS = 16
# Weight of the steadiness term beside the pixel loss in base training.
STEADINESS = 0.05
BASE_RATE = 4e-4
BASE_BATCH = 8
# Frames per training sequence: the recurrence is unrolled over all of them.
FRAMES = 6

# Checkpoint keys, by field. The published checkpoints use these names (and
# carry other keys, which are ignored), so one reader serves old and new files.
_SAVED_KEYS = {
    "channels": "ch",
    "blocks": "blocks",
    "state_channels": "hid",
    "steadiness": "steady",
    "perceptual": "lpips",
    "texture": "texture",
    "rate": "lr",
    "batch": "batch",
    "frames": "frames",
    "round_state": "state_round",
}


@dataclass(frozen=True)
class Options:
    """The network's shape and the settings of the training step that last touched it."""

    channels: int
    blocks: int
    state_channels: int = STATE_CHANNELS
    steadiness: float = STEADINESS
    perceptual: float = 0.0  # weight of the perceptual term; zero in base training
    texture: float = 0.0  # weight of the texture term (recipe.texture); zero in base training
    rate: float = BASE_RATE
    batch: int = BASE_BATCH
    frames: int = FRAMES
    # The state is rounded to 8 bits where it is produced, as the integer runtime stores it.
    round_state: bool = False

    @classmethod
    def from_saved(cls, saved: dict[str, Any]) -> Options:
        """The options stored in a checkpoint; keys this recipe does not use are ignored."""
        present = {
            f.name: saved[_SAVED_KEYS[f.name]] for f in fields(cls) if _SAVED_KEYS[f.name] in saved
        }
        return cls(**present)

    def to_saved(self) -> dict[str, Any]:
        """The options as a checkpoint stores them."""
        return {_SAVED_KEYS[f.name]: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class Checkpoint:
    """A trained model on disk."""

    options: Options
    weights: dict[str, torch.Tensor]
    # Training steps behind the weights; zero where the file does not record it.
    step: int = 0
    # The learnt tops of the 8-bit activations, present after the 8-bit conversion only.
    tops: list[float] | None = None


def load_checkpoint(path: Path) -> Checkpoint:
    """The checkpoint at `path`, on the CPU."""
    saved = torch.load(path, map_location="cpu")
    return Checkpoint(
        Options.from_saved(saved["opts"]), saved["model"], saved.get("step", 0), saved.get("tops")
    )


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """Write `checkpoint` to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {
        "opts": checkpoint.options.to_saved(),
        "model": checkpoint.weights,
        "step": checkpoint.step,
    }
    if checkpoint.tops is not None:
        saved["tops"] = checkpoint.tops
    torch.save(saved, path)
    print(f"saved {path} (step {checkpoint.step})", flush=True)
