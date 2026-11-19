"""Networks: shapes, export equivalence and the temporal design's guarantees."""

import pytest
import torch

from clearvisionnn.config.model import ModelConfig
from clearvisionnn.contract import CODING_CHANNELS, SIDE_CHANNELS
from clearvisionnn.models.blocks import RepConv, fuse
from clearvisionnn.models.confidence import confidence
from clearvisionnn.models.factory import build_model
from clearvisionnn.models.resample import block_field_to_flow, upscale, warp
from clearvisionnn.models.sequence import FrameInput

HEIGHT, WIDTH = 40, 56


def _inputs(reset: bool = False, batch: int = 2) -> FrameInput:
    torch.manual_seed(0)
    rows, cols = HEIGHT // 8, WIDTH // 8
    return FrameInput(
        torch.rand(batch, 3, HEIGHT, WIDTH),
        torch.randn(batch, 2, rows, cols),
        torch.rand(batch, SIDE_CHANNELS, rows, cols),
        torch.full((batch,), reset),
        torch.rand(batch, CODING_CHANNELS),
    )


def _randomise_head(model: torch.nn.Module) -> None:
    """Heads start at zero; give them weights so that tests exercise the whole network."""
    for parameter in model.core.head.parameters():
        torch.nn.init.normal_(parameter, std=0.05)


@pytest.mark.parametrize("layout", ["unet", "flat", "packed"])
@pytest.mark.parametrize("temporal", ["none", "blend"])
def test_output_is_twice_the_input_size(layout: str, temporal: str) -> None:
    model = build_model(ModelConfig(layout=layout, size="s", temporal=temporal)).eval()
    with torch.no_grad():
        output, history = model.step(_inputs(), None)
        again, _ = model.step(_inputs(), history)
    assert output.shape == again.shape == (2, 3, 2 * HEIGHT, 2 * WIDTH)


def test_untrained_network_is_the_conventional_upscale() -> None:
    model = build_model(ModelConfig(size="s")).eval()
    inputs = _inputs()
    with torch.no_grad():
        output, _ = model.step(inputs, None)
    assert torch.allclose(output, upscale(inputs.frame))


def test_fused_convolution_matches_its_branches() -> None:
    torch.manual_seed(0)
    block = RepConv(6, 6)
    x = torch.randn(1, 6, 9, 9)
    assert torch.allclose(block(x), block.fused()(x), atol=1e-5)


@pytest.mark.parametrize("layout", ["unet", "flat", "packed"])
def test_fused_model_matches_the_training_model(layout: str) -> None:
    model = build_model(ModelConfig(layout=layout, size="s", temporal="blend")).eval()
    _randomise_head(model)
    with torch.no_grad():
        _, history = model.step(_inputs(), None)
        before, _ = model.step(_inputs(), history)
        after, _ = fuse(model).step(_inputs(), history)
    assert torch.allclose(before, after, atol=1e-4)


def test_luma_only_model_leaves_chroma_conventional() -> None:
    model = build_model(ModelConfig(size="s", planes="y")).eval()
    _randomise_head(model)
    inputs = _inputs()
    with torch.no_grad():
        output, _ = model.step(inputs, None)
    base = upscale(inputs.frame)
    assert torch.equal(output[:, 1:], base[:, 1:])
    assert not torch.equal(output[:, :1], base[:, :1])


def test_reset_makes_history_irrelevant() -> None:
    model = build_model(ModelConfig(size="s", temporal="blend")).eval()
    _randomise_head(model)
    with torch.no_grad():
        _, history = model.step(_inputs(), None)
        with_history, _ = model.step(_inputs(reset=True), history)
        without, _ = model.step(_inputs(reset=True), None)
    assert torch.allclose(with_history, without, atol=1e-6)


def test_warp_moves_content_by_the_flow() -> None:
    image = torch.zeros(1, 1, 16, 16)
    image[0, 0, 5, 7] = 1.0
    flow = torch.zeros(1, 2, 16, 16)
    flow[:, 0], flow[:, 1] = 2.0, 1.0  # the match is 2 right and 1 down
    warped = warp(image, flow)
    assert warped[0, 0, 4, 5] == pytest.approx(1.0)


def test_block_field_covers_frames_that_blocks_overhang() -> None:
    assert block_field_to_flow(torch.ones(1, 2, 34, 60), 270, 480).shape == (1, 2, 270, 480)


RECURRENT = {"side_planes": False, "luma_only": True, "temporal": "recurrent", "state_channels": 6}


def test_recurrent_model_reconstructs_luma_and_carries_bounded_state() -> None:
    config = ModelConfig(
        size="s", layout="flat", coding_planes=True, confidence_plane=True, **RECURRENT
    )
    model = build_model(config).eval()
    for parameter in model.core.layout.head.parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    inputs = _inputs()
    with torch.no_grad():
        output, history = model.step(inputs, None)
        again, _ = model.step(inputs, history)
    base = upscale(inputs.frame)
    assert output.shape == again.shape == base.shape
    assert torch.equal(output[:, 1:], base[:, 1:]) and not torch.equal(output[:, :1], base[:, :1])
    assert history.hidden.shape == (2, 6, HEIGHT, WIDTH) and float(history.hidden.abs().max()) <= 1
    assert not torch.equal(output, again)  # the second frame saw history


def test_recurrent_reset_makes_history_irrelevant() -> None:
    model = build_model(ModelConfig(size="s", confidence_plane=True, **RECURRENT)).eval()
    for parameter in model.core.layout.head.parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    with torch.no_grad():
        _, history = model.step(_inputs(), None)
        with_history, _ = model.step(_inputs(reset=True), history)
        without, _ = model.step(_inputs(reset=True), None)
    assert torch.allclose(with_history, without, atol=1e-6)


def test_confidence_is_full_where_history_matches_and_gone_where_it_cannot() -> None:
    luma = torch.rand(1, 1, HEIGHT, WIDTH)
    assert torch.equal(confidence(luma, luma), torch.ones_like(luma))
    assert float(confidence(luma, luma + 0.5).max()) == 0.0


@pytest.mark.parametrize(
    "settings",
    [
        {"temporal": "recurrent"},  # the recurrent design is luma-only
        {"luma_only": True},  # needs side_planes: false
        {"luma_only": True, "side_planes": False, "temporal": "blend"},
        {"coding_planes": True},
        {"luma_only": True, "side_planes": False, "confidence_plane": True},
    ],
)
def test_configurations_of_models_that_do_not_exist_are_rejected(
    settings: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"luma|recurrent|blend"):
        ModelConfig(**settings)
