"""Planning, splitting and degradation, without touching real video."""

import numpy as np
import pytest

from clearvisionnn.config.pipeline import GradeConfig, ModuleSpec, SegmentConfig, SplitConfig
from clearvisionnn.pipeline.degrade import DegradeChain
from clearvisionnn.pipeline.degrade.jpeg import Jpeg
from clearvisionnn.pipeline.ingest import assign_split
from clearvisionnn.pipeline.records import SourceFile, Split
from clearvisionnn.pipeline.segment import choose_downscale, plan_file, target_size
from clearvisionnn.pipeline.workdir import fingerprint
from clearvisionnn.video.clip import Clip
from clearvisionnn.video.probe import VideoInfo, raw_frame_bytes


def test_downscale_grows_as_bitrate_falls() -> None:
    grade = GradeConfig()
    assert choose_downscale(float("inf"), grade) == 2
    assert choose_downscale(0.1, grade) == 3
    assert choose_downscale(0.04, grade) == 4
    assert choose_downscale(0.001, grade) is None


def test_target_size_is_aligned() -> None:
    assert target_size(1920, 1080, 2, 32) == (960, 512)


def _file(bitrate: float, frames: int = 3000, fps: float = 60.0) -> SourceFile:
    info = VideoInfo(
        width=1920,
        height=1080,
        fps=fps,
        frames=frames,
        codec="h264",
        pix_fmt="yuv420p",
        bitrate=bitrate,
    )
    return SourceFile(
        file_id="s-a-0123abcd",
        revision="1-1",
        source="s",
        path="/x/a.mp4",
        info=info,
        split=Split.TRAIN,
    )


def test_plan_decimates_high_frame_rates_and_spreads_clips() -> None:
    segments = plan_file(
        _file(bitrate=0.0), SegmentConfig(frames=64, max_per_file=3), GradeConfig()
    )
    assert [s.clip_id for s in segments] == [f"s-a-0123abcd-0{i}" for i in range(3)]
    assert len({s.fingerprint for s in segments}) == 3
    assert segments[0].step == 2 and segments[0].fps == 30.0
    assert segments[1].start_seconds > segments[0].start_seconds


def test_decimation_never_leaves_the_rate_above_the_ceiling() -> None:
    (segment,) = plan_file(
        _file(bitrate=0.0, fps=40.0), SegmentConfig(max_per_file=1), GradeConfig()
    )
    assert segment.step == 2 and segment.fps == 20.0


def test_plan_skips_files_too_compressed_to_use() -> None:
    assert plan_file(_file(bitrate=1000.0), SegmentConfig(), GradeConfig()) == []


def test_split_is_stable_and_roughly_proportional() -> None:
    config = SplitConfig(validation=0.1, test=0.2)
    splits = [assign_split("s", f"file{i}.mp4", config) for i in range(2000)]
    assert splits == [assign_split("s", f"file{i}.mp4", config) for i in range(2000)]
    assert 0.15 < splits.count(Split.TEST) / 2000 < 0.25
    assert 0.06 < splits.count(Split.VALIDATION) / 2000 < 0.14


def test_fingerprint_changes_with_content() -> None:
    assert fingerprint(GradeConfig()) == fingerprint(GradeConfig())
    assert fingerprint(GradeConfig()) != fingerprint(GradeConfig(align=16))


def test_raw_frame_bytes() -> None:
    assert raw_frame_bytes(4, 2, "yuv420p") == 12
    assert raw_frame_bytes(4, 2, "yuv444p16le") == 48
    with pytest.raises(ValueError, match="unsupported"):
        raw_frame_bytes(4, 2, "rgb24")


def _grey_clip(frames: int = 4, size: int = 32) -> Clip:
    return Clip(
        np.full((frames, size, size), 128, np.uint8),
        np.full((frames, size // 2, size // 2), 128, np.uint8),
        np.full((frames, size // 2, size // 2), 128, np.uint8),
    )


def _chain(*specs: tuple[str, dict]) -> DegradeChain:
    return DegradeChain.from_specs([ModuleSpec(type=t, params=p) for t, p in specs])


def test_chain_scales_and_splits_at_the_last_offline_module() -> None:
    chain = _chain(("blur", {}), ("scale", {}), ("codec", {}), ("noise", {}))
    stored, live = chain.split()
    assert chain.scale == 2
    assert (len(stored), len(live)) == (3, 1)
    assert (stored.scale, live.scale) == (2, 1)


def test_chain_with_no_offline_module_is_all_live() -> None:
    stored, live = _chain(("scale", {}), ("noise", {})).split()
    assert (len(stored), len(live)) == (0, 2)


def test_noise_is_fixed_in_level_per_clip_and_fresh_per_frame() -> None:
    chain = _chain(
        ("scale", {"kernels": ["area"]}),
        ("noise", {"sigma": [3.0, 3.0], "correlation": [0.0, 0.0]}),
    )
    rng = np.random.default_rng(1)
    out = chain.apply(_grey_clip(), chain.sample(rng), rng).y.astype(np.float32)
    assert out.shape == (4, 16, 16)
    assert not np.array_equal(out[0], out[1])
    assert 2.0 < out.std() < 4.0


def test_skipped_module_leaves_the_clip_untouched() -> None:
    chain = _chain(("scale", {"kernels": ["area"]}), ("noise", {"probability": 0.0}))
    rng = np.random.default_rng(1)
    params = chain.sample(rng)
    assert params[1] is None
    assert np.all(chain.apply(_grey_clip(), params, rng).y == 128)


def test_jpeg_changes_the_picture_a_little_and_keeps_its_shape() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 255, (2, 32, 48)).astype(np.float32)
    clip = Clip(y, np.full((2, 16, 24), 128, np.float32), np.full((2, 16, 24), 128, np.float32))
    out = Jpeg(quality=(80, 80), probability=1.0).apply(clip, {"quality": 80}, rng)
    assert out.y.shape == clip.y.shape and out.u.shape == clip.u.shape
    assert 0.0 < float(np.abs(out.y - clip.y).mean()) < 30.0
