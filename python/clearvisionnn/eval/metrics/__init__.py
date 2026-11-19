"""Quality metrics, grouped in families (DESIGN.md section 10).

Importing this package registers the built-in metrics.
"""

from clearvisionnn.eval.metrics import fidelity, perceptual, structure, temporal
from clearvisionnn.eval.metrics.base import METRICS, Family, Metric

__all__ = ["METRICS", "Family", "Metric", "fidelity", "perceptual", "structure", "temporal"]
