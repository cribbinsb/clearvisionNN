"""VMAF, MS-SSIM and XPSNR from an external `vmaf` program, where one is available.

The command line asks for the CUDA backend and the XPSNR feature, so the
program must be a build of libvmaf that has both.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

MAX_SAMPLE = 255
# The model is named so its pooled score can be found in the report.
MODEL_NAME = "vmaf_v1"
# XPSNR weights by temporal activity and needs a frame rate; clips are scored as 30 fps.
XPSNR_FPS = 30
NAMES = ("vmaf", "ms_ssim", "xpsnr_y")


@dataclass(frozen=True)
class VmafTool:
    """The `vmaf` program and the model file it scores with."""

    binary: Path
    model: Path

    def score(self, output: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Scores of (frames, 3, H, W) YUV `output` against `target`, both in [0, 1].

        VMAF is taken at the clip's own size, not a standard viewing size, so
        it compares models with each other and not with published figures.
        """
        height, width = target.shape[-2:]
        model = (
            f"path={self.model}:name={MODEL_NAME}:cambi.enc_width={width}"
            f":cambi.enc_height={height}:cambi.enc_bitdepth=8"
        )
        with tempfile.TemporaryDirectory() as folder:
            reference, distorted, report = (
                Path(folder, n) for n in ("ref.yuv", "out.yuv", "out.json")
            )
            reference.write_bytes(_yuv420(target))
            distorted.write_bytes(_yuv420(output))
            command = [str(self.binary), "-r", str(reference), "-d", str(distorted)]
            command += ["-w", str(width), "-h", str(height), "-p", "420", "-b", "8"]
            command += ["--model", model, "--feature", "float_ms_ssim"]
            command += ["--feature", f"xpsnr=fps={XPSNR_FPS}", "--backend", "cuda"]
            command += ["--precision", "max", "--json", "-o", str(report), "-q"]
            subprocess.run(command, check=True)
            pooled = json.loads(report.read_text())["pooled_metrics"]
        return {
            "vmaf": pooled[MODEL_NAME]["mean"],
            "ms_ssim": pooled["float_ms_ssim"]["mean"],
            "xpsnr_y": pooled["xpsnr_y"]["mean"],
        }


def _yuv420(frames: torch.Tensor) -> bytes:
    """8-bit planar 4:2:0 bytes of (frames, 3, H, W) YUV with full-size chroma, in [0, 1]."""

    def plane(x: torch.Tensor) -> bytes:
        return (x.clamp(0, 1) * MAX_SAMPLE).round().byte().cpu().numpy().tobytes()

    chroma = F.avg_pool2d(frames[:, 1:].float(), 2)
    return b"".join(
        plane(frames[i, 0]) + plane(chroma[i, 0]) + plane(chroma[i, 1]) for i in range(len(frames))
    )
