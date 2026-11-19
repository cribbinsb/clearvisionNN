"""Registry, configuration loading and the clip container."""

import numpy as np
import pytest

from clearvisionnn import contract
from clearvisionnn.config.load import load_config
from clearvisionnn.registry import Registry
from clearvisionnn.video import coding
from clearvisionnn.video.clip import Clip
from clearvisionnn.video.coding import MAX_QP, parse_encoder_stats
from clearvisionnn.video.ffmpeg import LOSSLESS_ARGS, encode
from clearvisionnn.video.reader import has_regular_timing, read_frames


def test_registry_returns_what_was_registered() -> None:
    registry: Registry[int] = Registry("thing")
    registry.register("one")(1)
    assert registry.get("one") == 1
    assert registry.names() == ["one"]


def test_registry_rejects_duplicates_and_unknown_names() -> None:
    registry: Registry[int] = Registry("thing")
    registry.register("one")(1)
    with pytest.raises(ValueError, match="twice"):
        registry.register("one")(2)
    with pytest.raises(KeyError, match="unknown thing"):
        registry.get("two")


def test_config_overrides_are_laid_over_the_base(tmp_path) -> None:
    (tmp_path / "base.yaml").write_text(
        "name: base\nwork_dir: w\nsources: []\n"
        "pipeline: {degrade: {chain: [{type: scale}]}, workers: 3}\n"
    )
    (tmp_path / "child.yaml").write_text("base: base.yaml\nname: child\npipeline: {workers: 5}\n")
    config = load_config(tmp_path / "child.yaml")
    assert config.name == "child"
    assert config.pipeline.workers == 5
    assert config.pipeline.degrade.chain[0].type == "scale"


def test_config_rejects_unknown_keys(tmp_path) -> None:
    (tmp_path / "bad.yaml").write_text(
        "name: x\nwork_dir: w\nsources: []\ntypo: 1\npipeline: {degrade: {chain: []}}\n"
    )
    with pytest.raises(ValueError, match="typo"):
        load_config(tmp_path / "bad.yaml")


def _clip(frames: int = 3, height: int = 8, width: int = 12) -> Clip:
    rng = np.random.default_rng(0)
    plane = lambda h, w: rng.integers(0, 256, (frames, h, w), dtype=np.uint8)  # noqa: E731
    return Clip(
        plane(height, width), plane(height // 2, width // 2), plane(height // 2, width // 2)
    )


def test_clip_crop_keeps_chroma_aligned() -> None:
    cropped = _clip().crop(2, 4, 4, 8)
    assert cropped.y.shape == (3, 4, 8)
    assert cropped.u.shape == (3, 2, 4)


def test_clip_rejects_odd_sizes_and_crops() -> None:
    with pytest.raises(ValueError, match="even"):
        Clip(
            np.zeros((1, 3, 4), np.uint8),
            np.zeros((1, 1, 2), np.uint8),
            np.zeros((1, 1, 2), np.uint8),
        )
    with pytest.raises(ValueError, match="even"):
        _clip().crop(1, 0, 4, 4)


def test_clip_float_round_trip_is_exact() -> None:
    clip = _clip()
    assert np.array_equal(clip.as_float().as_uint8().y, clip.y)


def test_reader_returns_the_frames_asked_for(tmp_path) -> None:
    clip = _clip(frames=20, height=32, width=48)
    path = tmp_path / "clip.mkv"
    encode(clip, path, 30.0, LOSSLESS_ARGS)
    assert has_regular_timing(path, 20)
    assert not has_regular_timing(path, 19)
    part = read_frames(path, 11, 5)
    assert np.array_equal(part.y, clip.y[11:16])
    assert np.array_equal(part.v, clip.v[11:16])


def test_encoder_statistics_give_quantiser_and_frame_type() -> None:
    text = (
        "out=  0 st=  0 frame=     0 q= 35.0 f_size=   2962 time= 0.010 type= I\n"
        "out=  0 st=  0 frame=     1 q= 42.5 f_size=    461 time= 0.033 type= P\n"
    )
    coding = parse_encoder_stats(text)
    expected = [[1.0, 35.0 / MAX_QP, 1.0], [1.0, 42.5 / MAX_QP, 0.0]]
    assert np.allclose(coding, expected)


def test_coding_planes_mean_the_same_to_the_data_and_to_the_runtime() -> None:
    assert (coding.CODING_CHANNELS, coding.MAX_QP) == (contract.CODING_CHANNELS, contract.MAX_QP)
