"""The Rust runtime session must reproduce the training model, frame for frame."""

import clearvisionnn_native
import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from clearvisionnn.config.model import ModelConfig
from clearvisionnn.contract import CODING_CHANNELS, MAX_QP
from clearvisionnn.data.features import analyse
from clearvisionnn.data.tensors import clip_to_tensor
from clearvisionnn.models.factory import build_model, trained_core
from clearvisionnn.models.per_frame import reconstructed_channels
from clearvisionnn.models.sequence import FrameInput
from clearvisionnn.video.clip import Clip

HEIGHT, WIDTH, FRAMES = 96, 128, 5


def _moving_clip() -> Clip:
    """Smooth texture drifting sideways, with a scene cut before the last frame."""
    rng = np.random.default_rng(0)
    noise = rng.uniform(0, 255, (HEIGHT + 16, WIDTH + 16)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 2.0)
    texture = (texture - texture.min()) / (texture.max() - texture.min()) * 255
    frames = [texture[8 : 8 + HEIGHT, 2 * t : 2 * t + WIDTH] for t in range(FRAMES - 1)]
    frames.append(255 - texture[16 : 16 + HEIGHT, 16 : 16 + WIDTH][::-1])
    y = np.stack(frames).astype(np.uint8)
    chroma = rng.integers(96, 160, (FRAMES, HEIGHT // 2, WIDTH // 2), dtype=np.uint8)
    return Clip(y, chroma, np.ascontiguousarray(chroma[:, ::-1]))


# How each frame of the clip was coded: unknown, then varying quantisers and one key frame.
CODING = [None, (30.0, True), (38.5, False), None, (24.0, False)]
LUMA_ONLY = {"planes": "y", "side_planes": False, "luma_only": True}
CONFIGS = [
    *({"temporal": temporal, "planes": planes, "align": align}
      for temporal in ("none", "blend") for planes in ("y", "yuv") for align in (True, False)),
    {**LUMA_ONLY, "coding_planes": True},
    {**LUMA_ONLY, "temporal": "recurrent", "state_channels": 16, "align": False},
    {**LUMA_ONLY, "temporal": "recurrent", "state_channels": 16, "coding_planes": True,
     "confidence_plane": True},
    {**LUMA_ONLY, "temporal": "recurrent", "confidence_plane": True, "align": False},
]  # fmt: skip


def _coding_planes(coding: tuple[float, bool] | None) -> torch.Tensor:
    """The (1, CODING_CHANNELS) tensor training would hold for `coding`."""
    if coding is None:
        return torch.zeros(1, CODING_CHANNELS)
    return torch.tensor([[1.0, coding[0] / MAX_QP, float(coding[1])]])


@pytest.mark.parametrize("settings", CONFIGS, ids=lambda s: "-".join(map(str, s.values())))
def test_runtime_matches_the_training_model(settings: dict[str, object]) -> None:
    torch.manual_seed(0)
    config = ModelConfig(size="s", **settings)
    model = build_model(config).eval()
    for parameter in trained_core(model).parameters():
        if not parameter.abs().sum():  # the head starts at zero: give the network an output
            torch.nn.init.normal_(parameter, std=0.05)

    clip = _moving_clip()
    frames, features = clip_to_tensor(clip), analyse(clip.y)
    assert features.reset.tolist() == [True, False, False, False, True]
    session = clearvisionnn_native.RuntimeSession(
        WIDTH, HEIGHT, reconstructed_channels(config.planes), config.temporal.value, config.align,
        config.state_channels, config.side_planes, config.luma_only, config.coding_planes,
        config.confidence_plane,
    )  # fmt: skip
    history = None
    for t in range(FRAMES):
        inputs = FrameInput(frames[t : t + 1], features.motion[t : t + 1], features.side[t : t + 1],
                            features.reset[t : t + 1], _coding_planes(CODING[t]))  # fmt: skip
        session.set_coding(*(CODING[t] or (None,)))
        with torch.no_grad():
            expected, history = model.step(inputs, history)
            network_input = torch.from_numpy(session.prepare(clip.y[t], clip.u[t], clip.v[t]))
            fine, coarse = model.core(network_input[None])
        actual = session.finish(fine[0].numpy(), coarse[0].contiguous().numpy())
        assert np.abs(actual - expected[0].numpy()).max() < 2e-4, f"frame {t}"


def test_sender_reduction_is_the_training_downscale() -> None:
    """The demo's 2x reduction must be Pillow's antialiased bicubic, as `degrade/scale.py` uses."""
    rng = np.random.default_rng(1)
    smooth = cv2.GaussianBlur(rng.uniform(0, 255, (70, 106, 3)).astype(np.float32), (0, 0), 1.5)
    rgb = np.clip(smooth * 2 - 128, 0, 255).astype(np.uint8)  # reaches both ends of the range
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    reduced = clearvisionnn_native.reduce_rgba(np.ascontiguousarray(rgba))
    expected = Image.fromarray(rgb).resize((53, 35), Image.Resampling.BICUBIC)
    assert np.array_equal(reduced[..., :3], np.asarray(expected))
    assert (reduced[..., 3] == 255).all()
