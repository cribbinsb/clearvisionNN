"""Grain: the Rust reference against an independent NumPy calculation, and its place in a frame."""

import clearvisionnn_native
import numpy as np
import pytest
from grain_reference import grain_steps
from test_integer_network import HEIGHT, WIDTH, _model, _real_frames

STEPS = 32  # picture steps per grey level
# The profiles and their grey levels of grain, as the runtime has them.
PROFILES = dict(clearvisionnn_native.GRAIN_PROFILES)
# One grey level of luma is this much of a level of green.
GREEN_PER_LUMA = 255 / 219


def test_the_profiles_are_the_two_fixed_levels_and_none() -> None:
    assert pytest.approx({"none": 0.0, "produced": 0.35, "webcam": 0.75}) == PROFILES
    with pytest.raises(ValueError, match="grain profile"):
        clearvisionnn_native.grain_steps(0, HEIGHT, WIDTH, "1.5")


@pytest.mark.parametrize("frame", [0, 1, 77, 2**32 - 1])
@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_rust_grain_matches_numpy_bit_for_bit(frame: int, profile: str) -> None:
    got = clearvisionnn_native.grain_steps(frame, HEIGHT, WIDTH, profile)
    assert got.shape == (2 * HEIGHT, 2 * WIDTH)
    assert np.array_equal(got, grain_steps(frame, HEIGHT, WIDTH, PROFILES[profile]))


@pytest.mark.parametrize("profile", ["produced", "webcam"])
def test_grain_is_as_strong_as_its_profile_says(profile: str) -> None:
    grain = clearvisionnn_native.grain_steps(3, HEIGHT, WIDTH, profile) / STEPS
    assert abs(grain.mean()) < 0.03
    assert grain.std() == pytest.approx(PROFILES[profile], rel=0.03)


def test_integer_runtime_shows_grain_and_keeps_none_of_it() -> None:
    """With grain only the picture changes: the luma kept and the next frames' input do not.

    The network is silent, so the picture is the conventional upscale of real frames: a random
    one drives most samples to black or white, where grain is clipped away.
    """
    config, model = _model("recurrent", silent=True)
    args = (model.words(), WIDTH, HEIGHT, config.temporal.value, config.align,
            config.state_channels, config.confidence_plane)  # fmt: skip
    still = np.zeros((2, -(-HEIGHT // 8), -(-WIDTH // 8)), np.float32)
    clean, untold, webcam = (clearvisionnn_native.IntegerRuntimeSession(*args) for _ in range(3))
    clean.set_grain("none")
    webcam.set_grain("webcam")
    for y, u, v in _real_frames():
        kept = clean.step(y, u, v, still, False)
        assert np.array_equal(untold.step(y, u, v, still, False), kept)
        assert np.array_equal(webcam.step(y, u, v, still, False), kept)
        assert np.array_equal(untold.picture(), webcam.picture()), "webcam is the default"
        assert np.array_equal(webcam.packed_input(), clean.packed_input())
        added = webcam.picture()[..., 1].astype(int) - clean.picture()[..., 1]
        # The picture is whole levels, so the grain shows with its rounding on top.
        assert added.std() == pytest.approx(PROFILES["webcam"] * GREEN_PER_LUMA, rel=0.2)
