"""The integer network: exporter and Rust reference against an independent NumPy calculation."""

from pathlib import Path

import clearvisionnn_native
import numpy as np
import pytest
import torch
from integer_reference import infer, pack_output, quantise_input

from clearvisionnn.config.model import ModelConfig
from clearvisionnn.export.integer import quantise
from clearvisionnn.export.integer_model import IntegerModel
from clearvisionnn.models.factory import build_model, trained_core
from clearvisionnn.models.layouts import FlatLayout
from clearvisionnn.models.luma import state_input_channels
from clearvisionnn.models.state import BoundedState
from clearvisionnn.video.reader import read_frames

CLIP = Path(__file__).parents[2] / "web" / "clips" / "bbb-tree-720p.mp4"
HEIGHT, WIDTH, FRAMES = 72, 104, 4  # not multiples of the tile, so edge tiles are partial
TILE, HALO = 32, 6
TOP, LEFT = 240, 400  # corner of the crop


def _real_frames() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Consecutive 4:2:0 crops of a bundled clip, where it has detail and motion."""
    clip = read_frames(CLIP, 0, FRAMES)
    rows, columns = slice(TOP, TOP + HEIGHT), slice(LEFT, LEFT + WIDTH)
    half_rows = slice(TOP // 2, (TOP + HEIGHT) // 2)
    half_columns = slice(LEFT // 2, (LEFT + WIDTH) // 2)
    return [
        tuple(
            np.ascontiguousarray(plane)
            for plane in (y[rows, columns], u[half_rows, half_columns], v[half_rows, half_columns])
        )
        for y, u, v in zip(clip.y, clip.u, clip.v, strict=True)
    ]


def _model(
    temporal: str, halo: int = HALO, silent: bool = False
) -> tuple[ModelConfig, IntegerModel]:
    """A randomly weighted luma model of the given design, in integer form.

    A `silent` one keeps its untrained head, whose outputs are all zero.
    """
    torch.manual_seed(0)
    recurrent = temporal == "recurrent"
    config = ModelConfig(
        layout="flat", size="s", temporal=temporal, luma_only=True, side_planes=False,
        coding_planes=True, confidence_plane=recurrent,
    )  # fmt: skip
    core = trained_core(build_model(config))
    layout = core.layout if isinstance(core, BoundedState) else core
    assert isinstance(layout, FlatLayout)
    for parameter in layout.parameters():
        if not parameter.abs().sum() and not silent:  # the head starts at zero
            torch.nn.init.normal_(parameter, std=0.05)
    tops = [1.5] * (len(layout.body) + 2)
    return config, quantise(layout, tops, state_input_channels(config), halo, TILE)


def _session(config: ModelConfig) -> clearvisionnn_native.RuntimeSession:
    return clearvisionnn_native.RuntimeSession(
        WIDTH, HEIGHT, 1, config.temporal.value, config.align, config.state_channels,
        config.side_planes, config.luma_only, config.coding_planes, config.confidence_plane,
    )  # fmt: skip


@pytest.mark.parametrize("temporal", ["none", "recurrent"])
@pytest.mark.parametrize("tile", [0, TILE])
def test_rust_reference_matches_numpy_bit_for_bit_over_a_run(temporal: str, tile: int) -> None:
    """Real frames through the session, the network in Rust, its state fed back each frame."""
    config, model = _model(temporal)
    network = clearvisionnn_native.IntegerNetwork(model.words())
    session = _session(config)
    session.set_coding(30.0, False)
    for index, (y, u, v) in enumerate(_real_frames()):
        features = session.prepare(y, u, v)
        packed = network.pack_input(features)
        outputs = network.infer(packed, WIDTH, HEIGHT, tile)
        expected = pack_output(infer(model, quantise_input(model, features), tile))
        assert np.array_equal(outputs, expected), f"frame {index}"
        assert len(np.unique(outputs)) > 100, "the network must be doing something"
        session.finish(*network.unpack_output(outputs, WIDTH, HEIGHT))


def test_a_halo_as_deep_as_the_network_makes_tiling_exact() -> None:
    config, model = _model("recurrent", halo=len(_model("recurrent")[1].body) + 1)
    network = clearvisionnn_native.IntegerNetwork(model.words())
    features = _session(config).prepare(*_real_frames()[0])
    packed = network.pack_input(features)
    whole = network.infer(packed, WIDTH, HEIGHT, 0)
    assert np.array_equal(network.infer(packed, WIDTH, HEIGHT, TILE), whole)
    assert network.tile == TILE and network.halo == len(model.body) + 1


def test_a_short_halo_changes_little() -> None:
    """The deployed halo is an approximation; it must stay a small one."""
    config, model = _model("none")
    network = clearvisionnn_native.IntegerNetwork(model.words())
    packed = network.pack_input(_session(config).prepare(*_real_frames()[0]))
    fine = [network.unpack_output(network.infer(packed, WIDTH, HEIGHT, tile), WIDTH, HEIGHT)[0]
            for tile in (0, TILE)]  # fmt: skip
    difference = np.abs(fine[0] - fine[1]) * 255
    assert 0 < difference.max() < 8 and difference.mean() < 0.05


def test_unpacked_outputs_are_the_residual_and_the_bounded_state() -> None:
    config, model = _model("recurrent")
    network = clearvisionnn_native.IntegerNetwork(model.words())
    features = _session(config).prepare(*_real_frames()[0])
    raw = infer(model, quantise_input(model, features))
    fine, coarse = network.unpack_output(pack_output(raw), WIDTH, HEIGHT)
    residual = torch.pixel_shuffle(torch.from_numpy(raw[:4] / (255 * 32)), 2).numpy()
    assert np.allclose(fine, residual, atol=1e-7) and np.allclose(coarse, raw[4:] / 127, atol=1e-7)
    assert np.abs(coarse).max() <= 1


@pytest.mark.parametrize("damage", ["magic", "schema", "truncated", "extended", "multiplier"])
def test_a_damaged_file_is_refused(damage: str) -> None:
    model = _model("recurrent")[1]
    words = model.words().copy()
    if damage == "magic":
        words[0] += 1
    elif damage == "schema":
        words[1] += 1
    elif damage == "truncated":
        words = words[:-1]
    elif damage == "extended":
        words = np.append(words, 0).astype(np.int32)
    else:  # the file ends with the head's multipliers, its shifts and the tanh table
        outputs = len(model.head.shift)
        words[-(len(model.tanh) // 4 + outputs + outputs // 2)] = 1 << 20
    with pytest.raises(ValueError, match="integer model"):
        clearvisionnn_native.IntegerNetwork(words)


@pytest.mark.parametrize(
    ("coding", "expected"), [((30.0, True), [255, 150, 255]), ((None,), [0, 0, 0])]
)
def test_coding_planes_reach_the_packed_input(coding: tuple, expected: list[int]) -> None:
    """Whether the model is told how a frame was coded shows in its 8-bit input planes."""
    config, model = _model("recurrent")
    network = clearvisionnn_native.IntegerNetwork(model.words())
    session = _session(config)
    session.set_coding(*coding)
    packed = network.pack_input(session.prepare(*_real_frames()[0]))
    stored = packed.view(np.int8).reshape(HEIGHT * WIDTH, -1)  # a byte per input slot
    first = model.input_channels - len(expected)  # the coding planes are the last channels
    for channel, value in enumerate(expected, start=first):
        plane = stored[:, model.slots.index(channel)].astype(np.int64) + 128  # unsigned planes
        assert (plane == value).all(), f"channel {channel}"
