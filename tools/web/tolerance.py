"""What the browser checks accept of each form of the 8-bit network's shaders.

An exact form (integer, 32-bit float) must give the reference's words. The
approximate one (16-bit float) is held to the bounds below, which
docs/INTEGER.md, section 8, derives from measurements: each is the worst
measured figure with room for another GPU's rounding, and far inside what
would be seen.
"""

from __future__ import annotations

from typing import Any

EXACT_FORMS = ("integer", "float")
# A layer given the exact layer's input: never more than one step out, on a small share of
# samples (2.1% at worst, in a stem). The picture outputs count thirty-seconds of a step, so most
# of them differ by one of those; only their size is bounded.
LAYER_STEPS = 1.0
LAYER_SHARE = {"values": 0.05, "state": 0.05, "picture": 1.0}
# Through the whole network and, in the whole-frame check, on through the history, frame after
# frame: the worst frame's PSNR against the exact result, and the largest difference in 8-bit
# steps of the plane. By what is compared: `picture` and `state` are the network's two outputs,
# `values` its input (which holds the last frame's state and output), `colour` a finished RGBA
# picture. The worst measured (the extra-large model, 40 frames of history on the most detailed
# clip) were 53.7, 52.9, 54.5 and 50.9 dB and 11.2, 7, 11 and 13 steps; the bounds leave about
# 3 dB and as many steps again.
WORST_FRAME_PSNR = {"picture": 50.0, "state": 49.0, "values": 51.0, "colour": 48.0}
LARGEST_STEPS = {"picture": 24.0, "state": 16.0, "values": 24.0, "colour": 26.0}

Report = dict[str, Any]


def _within(report: Report, steps: float, psnr: float | None, share: float = 1.0) -> bool:
    """Whether one plane's differences (differences.js's report of it) are inside the bounds."""
    worst = report["worstFramePsnr"]
    return (
        report["largestSteps"] <= steps
        and report["differingShare"] <= share
        and (psnr is None or worst is None or worst >= psnr)
    )


def output_failures(form: str, differences: Report) -> list[str]:
    """The planes of a result that a shader of `form` got wrong, by name; empty if none."""
    if form in EXACT_FORMS:
        return ["words"] if differences["differingWords"] else []
    planes = {name: plane for name, plane in differences.items() if isinstance(plane, dict)}
    return [
        name
        for name, plane in planes.items()
        if not _within(plane, LARGEST_STEPS[name], WORST_FRAME_PSNR[name])
    ]


def layer_failures(differences: Report) -> list[str]:
    """The planes of one layer's output, given exact input, outside the approximate form's bound."""
    planes = {name: plane for name, plane in differences.items() if isinstance(plane, dict)}
    return [
        name
        for name, plane in planes.items()
        if not _within(plane, LAYER_STEPS, None, LAYER_SHARE[name])
    ]
