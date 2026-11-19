"""Checkpoints, configuration limits and the checks that guard a run."""

import pytest
import torch
from pydantic import ValidationError

from clearvisionnn.config.model import ModelConfig
from clearvisionnn.config.pipeline import DegradeConfig, ModuleSpec, PipelineConfig, SplitConfig
from clearvisionnn.config.run import RunConfig
from clearvisionnn.config.train import DataConfig, TrainConfig
from clearvisionnn.models.factory import build_model
from clearvisionnn.pipeline.degrade import DegradeChain
from clearvisionnn.train import checkpoint
from clearvisionnn.train.checkpoint import TrainState
from clearvisionnn.train.optim import build_optimizer, build_scheduler


def _config(tmp_path, **model) -> RunConfig:
    degrade = DegradeConfig(chain=[ModuleSpec(type="scale")])
    (tmp_path / "data").mkdir(exist_ok=True)
    for name in ("clips.jsonl", "variants-x.jsonl"):
        (tmp_path / "data" / name).write_text("")
    return RunConfig(name="t", work_dir=tmp_path, sources=[], model=ModelConfig(size="s", **model),
                     pipeline=PipelineConfig(degrade=degrade))  # fmt: skip


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr("clearvisionnn.train.provenance.data_fingerprint", lambda config: "test")
    config = _config(tmp_path)
    model = build_model(config.model)
    optimizer = build_optimizer(model.parameters(), config.train)
    state = TrainState(model, optimizer, build_scheduler(optimizer, config.train), step=7)
    path = tmp_path / "best.pt"
    checkpoint.save(state, config, path, {"psnr_y": 30.0})
    return config, state, path


def test_checkpoint_describes_its_run(saved) -> None:
    config, _, path = saved
    described = checkpoint.describe(path)
    assert described.step == 7 and described.scores == {"psnr_y": 30.0}
    assert described.config.model == config.model
    assert described.provenance.data_fingerprint == "test"


def test_a_behavioural_model_change_is_refused_even_when_shapes_match(saved, tmp_path) -> None:
    _, state, path = saved
    changed = _config(tmp_path, side_planes=False)
    build_model(changed.model).load_state_dict(state.model.state_dict())  # shapes do match
    with pytest.raises(ValueError, match="side_planes"):
        checkpoint.require_same_model(changed.model, checkpoint.describe(path).config.model, path)
    with pytest.raises(ValueError, match="different model"):
        checkpoint.restore(state, changed, path)


def test_resume_restores_step_and_random_state(saved) -> None:
    config, state, path = saved
    expected = torch.rand(3)
    state.step = 0
    checkpoint.restore(state, config, path)
    assert state.step == 7
    assert torch.equal(torch.rand(3), expected)


def test_out_of_range_settings_are_rejected_at_load() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(validate_every=0)
    with pytest.raises(ValidationError):
        DataConfig(crop=100)
    with pytest.raises(ValidationError):
        SplitConfig(validation=0.6, test=0.5)
    with pytest.raises(ValidationError, match="noise_pair"):
        TrainConfig(losses={"consistency": 1.0})


def test_bad_degradation_parameters_are_rejected_when_the_chain_is_built() -> None:
    with pytest.raises(ValueError, match="sigma"):
        DegradeChain.from_specs([ModuleSpec(type="noise", params={"sigma": [4.0, 1.0]})])
    with pytest.raises(ValueError, match="probability"):
        DegradeChain.from_specs([ModuleSpec(type="sharpen", params={"probability": 1.5})])
    with pytest.raises(TypeError):
        DegradeChain.from_specs([ModuleSpec(type="blur", params={"sgima": [0, 1]})])
