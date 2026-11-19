"""The integer runtime against the float runtime it stands in for, on real frames.

The two are different arithmetic on purpose (docs/INTEGER.md, "The runtime"),
so this is a tolerance, not an identity: the network is the same integer
network in both, and what differs is the upscale, the warp and the confidence
plane around it.
"""

import clearvisionnn_native
import numpy as np
from test_integer_network import HEIGHT, WIDTH, _model, _real_frames, _session

from clearvisionnn.data.features import analyse

LEVELS = 255
CODING = (30.0, False)
# Mean distances allowed between the two runtimes: of the output in 8-bit
# levels, and of the network's input planes in their own steps.
OUTPUT_LEVELS, INPUT_STEPS = 0.02, 0.1  # and no input sample more than one step away


def test_integer_runtime_stays_close_to_the_float_runtime() -> None:
    """Upscale, warp and confidence in fixed point against their float definitions.

    The network is silent (its untrained head outputs zero), so both runtimes
    carry the same history and every difference is the runtime's own.
    """
    config, model = _model("recurrent", silent=True)
    words = model.words()
    network = clearvisionnn_native.IntegerNetwork(words)
    floating = _session(config)
    floating.set_coding(*CODING)
    integer = clearvisionnn_native.IntegerRuntimeSession(
        words, WIDTH, HEIGHT, config.temporal.value, config.align, config.state_channels,
        config.coding_planes, config.confidence_plane,
    )  # fmt: skip
    frames = _real_frames()
    features = analyse(np.stack([y for y, _, _ in frames]))
    assert float(features.motion.abs().max()) > 0, "the clip must move, or no warp is tested"
    for index, (y, u, v) in enumerate(frames):
        packed = network.pack_input(floating.prepare(y, u, v))
        result = network.infer(packed, WIDTH, HEIGHT, 0)
        expected = floating.finish(*network.unpack_output(result, WIDTH, HEIGHT))[0]
        motion = np.ascontiguousarray(features.motion[index].numpy(), dtype=np.float32)
        luma = integer.step(y, u, v, motion, bool(features.reset[index]), *CODING)
        output_gap = np.abs(np.clip(expected, 0, 1) - luma) * LEVELS
        stored = [words.view(np.int8).astype(int) for words in (packed, integer.packed_input())]
        input_gap = np.abs(stored[0] - stored[1])
        assert output_gap.mean() < OUTPUT_LEVELS, f"frame {index}: {output_gap.mean():.4f} levels"
        assert input_gap.mean() < INPUT_STEPS, f"frame {index}: {input_gap.mean():.4f} steps"
        assert input_gap.max() <= 1, f"frame {index}: an input sample {input_gap.max()} steps away"


def test_integer_runtime_restarts_cleanly() -> None:
    """After a restart a frame is upscaled exactly as the first frame of a stream is."""
    config, model = _model("recurrent")
    args = (model.words(), WIDTH, HEIGHT, config.temporal.value, config.align,
            config.state_channels, config.coding_planes, config.confidence_plane)  # fmt: skip
    frames = _real_frames()
    still = np.zeros((2, -(-HEIGHT // 8), -(-WIDTH // 8)), np.float32)
    runtime = clearvisionnn_native.IntegerRuntimeSession(*args)
    for y, u, v in frames[:2]:
        runtime.step(y, u, v, still, False)
    runtime.restart()
    fresh = clearvisionnn_native.IntegerRuntimeSession(*args)
    assert np.array_equal(
        runtime.step(*frames[2], still, False), fresh.step(*frames[2], still, False)
    )
    assert np.array_equal(runtime.picture(), fresh.picture())
