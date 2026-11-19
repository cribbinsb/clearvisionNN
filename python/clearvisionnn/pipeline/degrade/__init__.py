"""Degradation modules and the chain that applies them (DESIGN.md section 9).

Importing this package registers every built-in module.
"""

from clearvisionnn.pipeline.degrade import blur, codec, denoise, jpeg, noise, scale, sharpen
from clearvisionnn.pipeline.degrade.base import MODULES, DegradeModule
from clearvisionnn.pipeline.degrade.chain import DegradeChain

__all__ = [
    "MODULES",
    "DegradeChain",
    "DegradeModule",
    "blur",
    "codec",
    "denoise",
    "jpeg",
    "noise",
    "scale",
    "sharpen",
]
