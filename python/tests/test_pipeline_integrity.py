"""The pipeline on real (synthetic) video: identity, caching and failure handling.

Each test guards a way the dataset could silently go wrong.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from clearvisionnn.config.load import load_config
from clearvisionnn.config.run import RunConfig
from clearvisionnn.data.index import ClipIndex
from clearvisionnn.data.views import ViewMaker
from clearvisionnn.pipeline import clean, ingest, segment, verify
from clearvisionnn.pipeline.degrade import DegradeChain
from clearvisionnn.pipeline.ingest import file_id
from clearvisionnn.pipeline.records import ClipRecord, Segment, SourceFile, Split, VariantRecord
from clearvisionnn.pipeline.runner import run_pipeline
from clearvisionnn.pipeline.variants import records_name
from clearvisionnn.video.ffmpeg import FFMPEG, decode

CONFIG = """
name: test
work_dir: {work}
sources: [{{name: s, root: {root}}}]
pipeline:
  workers: 2
  segment: {{frames: 8, max_per_file: 1}}
  grade: {{min_target_side: 64, align: 16}}
  clean: {{cut_threshold: 255.0, frozen_threshold: 0.0, min_sharpness: 0.0}}
  degrade:
    variants_per_clip: 1
    chain:
      - {{type: scale, params: {{kernels: [area]}}}}
      - {{type: codec, params: {{codecs: [libx264], bits_per_pixel: [0.5, 0.5], probability: 1.0}}}}
"""


def _video(path: Path, extra: tuple[str, ...] = ()) -> None:
    """A short, nearly lossless clip of a moving test pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=192x128:rate=30", "-frames:v", "12", "-c:v", "libx264", "-qp", "4",
               "-pix_fmt", "yuv420p", *extra, str(path)]  # fmt: skip
    subprocess.run(command, check=True)


@pytest.fixture
def config(tmp_path: Path) -> RunConfig:
    root = tmp_path / "source"
    _video(root / "scene0" / "video.mp4")
    _video(root / "scene8" / "video.mp4")
    (tmp_path / "run.yaml").write_text(CONFIG.format(work=tmp_path / "work", root=root))
    return load_config(tmp_path / "run.yaml")


def _clips(config: RunConfig) -> list[ClipRecord]:
    workdir = run_pipeline(config)
    return workdir.read_records(clean.RECORDS, ClipRecord)


def test_same_named_files_in_different_directories_stay_distinct(config: RunConfig) -> None:
    assert file_id("s", "scene0/video.mp4") != file_id("s", "scene8/video.mp4")
    clips = _clips(config)
    assert len({c.clip_id for c in clips}) == len({c.file_id for c in clips}) == 2
    assert len({c.target for c in clips}) == 2


def test_a_file_added_later_is_picked_up(config: RunConfig) -> None:
    assert len(_clips(config)) == 2
    _video(config.sources[0].root / "scene9" / "other.mp4")
    assert len(_clips(config)) == 3


def test_a_changed_plan_or_source_gets_new_media(config: RunConfig) -> None:
    before = {c.clip_id: c.target for c in _clips(config)}
    shorter = config.model_copy(
        update={"pipeline": config.pipeline.model_copy(
            update={"segment": config.pipeline.segment.model_copy(update={"frames": 6})})}
    )  # fmt: skip
    after = {c.clip_id: c for c in _clips(shorter)}
    assert set(after) == set(before)
    assert all(after[c].target != before[c] and after[c].frames == 6 for c in before)
    _video(config.sources[0].root / "scene0" / "video.mp4", extra=("-vf", "negate"))
    replaced = {c.clip_id: c.target for c in _clips(shorter)}
    changed = [c for c in replaced if replaced[c] != after[c].target]
    assert len(changed) == 1


def test_variants_match_their_targets_even_for_rotated_sources(config: RunConfig) -> None:
    _video(config.sources[0].root / "phone" / "rotated.mp4", extra=("-metadata:s:v", "rotate=90"))
    workdir = run_pipeline(config)
    clips = {c.clip_id: c for c in workdir.read_records(clean.RECORDS, ClipRecord)}
    variants = workdir.read_records(records_name(config.pipeline.degrade), VariantRecord)
    assert len(variants) == len(clips) == 3
    for variant in variants:
        clip = clips[variant.clip_id]
        target = decode(workdir.path(clip.target), clip.width, clip.height).y[3]
        small = decode(workdir.path(variant.path), variant.width, variant.height).y[3]
        restored = cv2.resize(small, (clip.width, clip.height), interpolation=cv2.INTER_CUBIC)
        error = np.abs(restored.astype(np.float32) - target).mean()
        assert error < 12, f"{variant.clip_id}: input does not show its target ({error:.1f})"


def test_unreadable_files_are_recorded_not_fatal(config: RunConfig) -> None:
    (config.sources[0].root / "broken.mp4").write_bytes(b"not a video")
    workdir = run_pipeline(config)
    assert len(workdir.read_records(ingest.RECORDS, SourceFile)) == 2
    failures = workdir.path("ingest.failures.jsonl").read_text()
    assert "broken.mp4" in failures


def test_a_clip_without_its_stored_variants_is_refused(config: RunConfig) -> None:
    workdir = run_pipeline(config)
    degrade = config.pipeline.degrade
    workdir.write_records(records_name(degrade), [])
    for split in Split:
        try:
            index = ClipIndex(workdir, split, degrade)
        except ValueError:
            continue  # no clips fell in this split
        with pytest.raises(ValueError, match="no stored variants"):
            ViewMaker(index, DegradeChain.from_specs(degrade.chain))


def test_planned_identifiers_are_unique(config: RunConfig) -> None:
    workdir = run_pipeline(config)
    planned = workdir.read_records(segment.RECORDS, Segment)
    assert len({s.clip_id for s in planned}) == len(planned) == 2


def test_a_variant_showing_the_wrong_picture_fails_the_pipeline(config: RunConfig) -> None:
    _video(config.sources[0].root / "scene0" / "video.mp4", extra=("-vf", "negate"))
    workdir = run_pipeline(config)
    name = records_name(config.pipeline.degrade)
    first, second = workdir.read_records(name, VariantRecord)
    swapped = [first.model_copy(update={"path": second.path}),
               second.model_copy(update={"path": first.path})]  # fmt: skip
    workdir.write_records(name, swapped)
    with pytest.raises(RuntimeError, match="do not show their target"):
        verify.run(config, workdir)
