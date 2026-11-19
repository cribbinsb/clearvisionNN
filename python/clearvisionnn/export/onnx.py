"""ONNX export of a network's core: plain convolutions, ReLUs and depth-to-space.

Only the trained core is exported. The conventional upscale, the warp and the
blend stay outside it, in `cvnn-runtime`, so the graph uses nothing beyond what
every runtime (ONNX Runtime, WebGPU, WebNN) supports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torch import nn

from clearvisionnn.models.blocks import fuse
from clearvisionnn.models.layouts import Layout

OPSET = 17
INPUT, FINE, COARSE = "features", "fine", "coarse"
# Size of the tensor used to trace and to check the export.
_TRACE_HEIGHT, _TRACE_WIDTH = 64, 96
# Largest difference allowed between ONNX Runtime and PyTorch on the check input.
_TOLERANCE = 1e-4
# The same for a half-precision graph against the full-precision network: a
# quarter of one 8-bit level, which rounding the picture to 8 bits hides.
_HALF_TOLERANCE = 1e-3


class _Core(nn.Module):
    """The layout with tuple outputs; a per-frame core has no coarse output to export.

    With `half`, the arithmetic inside is 16-bit float while inputs and outputs
    stay 32-bit, so the runtime around the graph does not change.
    """

    def __init__(self, layout: Layout, half: bool = False) -> None:
        super().__init__()
        self.layout = layout.half() if half else layout
        self.half_precision = half

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.half_precision:
            features = features.half()
        fine, coarse = self.layout(features)
        outputs = (fine, coarse) if self.layout.coarse_channels else (fine,)
        return tuple(output.float() for output in outputs)


def export_core(layout: Layout, path: Path, half: bool = False) -> None:
    """Write `layout`, fused, to `path`, and check ONNX Runtime reproduces it.

    `half` writes 16-bit weights and arithmetic, which GPUs run faster.
    """
    fuse(layout)
    example = torch.rand(1, layout.in_channels, _TRACE_HEIGHT, _TRACE_WIDTH)
    with torch.no_grad():
        expected = _Core(layout).eval()(example)
    core = _Core(layout, half).eval()
    outputs = [FINE, COARSE] if layout.coarse_channels else [FINE]
    spatial = {2: "height", 3: "width"}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        core, (example,), str(path), opset_version=OPSET, input_names=[INPUT],
        output_names=outputs, dynamic_axes={INPUT: spatial}, dynamo=False,
    )  # fmt: skip
    _check(path, example, expected, _HALF_TOLERANCE if half else _TOLERANCE)


def _check(
    path: Path, example: torch.Tensor, expected: tuple[torch.Tensor, ...], tolerance: float
) -> None:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1  # a one-off check; no need for a thread pool
    session = onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    exported = session.run(None, {INPUT: example.numpy()})
    for got, want in zip(exported, expected, strict=True):
        difference = float(np.abs(got - want.numpy()).max())
        if difference > tolerance:
            raise RuntimeError(f"{path}: ONNX Runtime differs from PyTorch by {difference:.2e}")
