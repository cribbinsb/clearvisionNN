"""Windows, features and batching."""

import numpy as np
import torch

from clearvisionnn.contract import BLOCK_SIZE, PLANE_NAMES, SIDE_CHANNELS
from clearvisionnn.data.batch import Degraded, SequenceBatch, collate
from clearvisionnn.data.crops import CROP_ALIGN, centre_window, full_frame, random_window
from clearvisionnn.data.features import analyse


def test_random_windows_stay_inside_and_on_the_grid() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        window = random_window(rng, clip_frames=32, height=136, width=240, frames=8, size=128)
        assert 0 <= window.start <= 24
        assert window.top % CROP_ALIGN == 0 and window.top + 128 <= 136
        assert window.left % CROP_ALIGN == 0 and window.left + 128 <= 240


def test_window_scales_to_target_resolution() -> None:
    assert centre_window(16, 256, 480, 8, 128).scaled(2) == (128, 352, 256, 256)
    assert full_frame(16, 256, 480).scaled(2) == (0, 0, 512, 960)


def _moving_luma(frames: int = 6, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (size // 4 + 8, size // 4 + 8)).astype(np.float32)
    big = np.kron(base, np.ones((4, 4), np.float32))
    return np.stack([big[8 : 8 + size, 2 * t : 2 * t + size] for t in range(frames)]).astype(
        np.uint8
    )


def test_features_have_block_grid_shapes_and_find_the_motion() -> None:
    luma = _moving_luma()
    features = analyse(luma)
    grid = 96 // BLOCK_SIZE
    assert features.motion.shape == (6, 2, grid, grid)
    assert features.side.shape == (6, SIDE_CHANNELS, grid, grid)
    assert features.reset.tolist() == [True, False, False, False, False, False]
    # Content moves left by 2 samples a frame, so its match in the previous frame is 2 to the right.
    assert abs(float(features.motion[2:, 0, 2:-2, 2:-2].median()) - 2.0) < 0.5


def test_withheld_planes_are_zero_with_a_cleared_flag() -> None:
    withheld = np.array([True, False, False, False])
    side = analyse(_moving_luma(), withheld).side
    assert float(side[:, 0].abs().max()) == 0.0  # the plane
    assert float(side[:, 1].max()) == 0.0  # its flag
    assert float(side[:, 3].min()) == 1.0  # the next plane's flag
    assert len(PLANE_NAMES) * 2 == SIDE_CHANNELS


def test_collate_stacks_nested_views() -> None:
    def sample() -> SequenceBatch:
        view = Degraded(
            torch.zeros(4, 3, 16, 16),
            torch.zeros(4, 2, 2, 2),
            torch.zeros(4, SIDE_CHANNELS, 2, 2),
            torch.zeros(4, dtype=torch.bool),
            torch.zeros(4, 3),
        )
        return SequenceBatch(torch.zeros(4, 3, 32, 32), view)

    batch = collate([sample(), sample()])
    assert batch.target.shape == (2, 4, 3, 32, 32)
    assert batch.view.reset.shape == (2, 4)
    assert batch.alternate is None
