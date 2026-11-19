"""Writing an evaluation's results as JSON and as a readable table."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from clearvisionnn.config.evaluate import EvalConfig
from clearvisionnn.eval.metrics import METRICS
from clearvisionnn.eval.scoring import MODEL, Results
from clearvisionnn.eval.summary import Interval, bootstrap_mean

# With fewer independent source files than this, resampling them says nothing
# about other content, and a verdict would be false confidence.
MIN_SOURCES_FOR_INTERVAL = 8


def gain_over_reference(
    results: Results, metric: str, config: EvalConfig, groups: dict[str, str]
) -> Interval:
    """The model's per-clip gain over the reference baseline, signed so positive is better."""
    model, reference = results[MODEL][metric], results[config.reference_baseline][metric]
    sign = 1.0 if METRICS.get(metric).higher_is_better else -1.0
    clips = sorted(model)
    gains = np.array([sign * (model[c].mean - reference[c].mean) for c in clips])
    return bootstrap_mean(gains, [groups[c] for c in clips], config.bootstrap_samples)


def _table(results: Results, config: EvalConfig, groups: dict[str, str]) -> list[str]:
    metrics = list(next(iter(results.values())))
    lines = [
        "| Method | " + " | ".join(f"{m} (worst window)" for m in metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for method, by_metric in results.items():
        cells = []
        for metric in metrics:
            scores = list(by_metric[metric].values())
            mean = np.mean([s.mean for s in scores])
            worst = np.mean([s.worst_window for s in scores])
            cells.append(f"{mean:.4g} ({worst:.4g})")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Model gain over {config.reference_baseline} (positive is better), "
        "mean and 95% interval over source files:",
        "",
    ]
    sources = len(set(groups.values()))
    for metric in metrics:
        gain = gain_over_reference(results, metric, config, groups)
        if sources < MIN_SOURCES_FOR_INTERVAL:
            verdict = f"only {sources} source files: no interval can be trusted"
        else:
            verdict = "clear" if gain.excludes_zero else "not separated from zero"
        lines.append(
            f"- {metric}: {gain.mean:+.4g} [{gain.low:+.4g}, {gain.high:+.4g}] ({verdict})"
        )
    return lines


def write_report(
    results: Results,
    config: EvalConfig,
    directory: Path,
    title: str,
    groups: dict[str, str],
    inputs: dict[str, object],
) -> Path:
    """Write `results.json` and `results.md` under `directory`; returns the Markdown path.

    `groups` maps each clip to its source file; `inputs` records what was
    evaluated (checkpoint, code and data identity, precision).
    """
    directory.mkdir(parents=True, exist_ok=True)
    scores = {
        method: {m: {c: asdict(s) for c, s in clips.items()} for m, clips in by_metric.items()}
        for method, by_metric in results.items()
    }
    (directory / "results.json").write_text(
        json.dumps({"inputs": inputs, "scores": scores}, indent=1)
    )
    header = [
        f"# {title}",
        "",
        f"{len(groups)} held-out clips from {len(set(groups.values()))} source files.",
        "",
    ]
    header += [f"- {key}: {value}" for key, value in inputs.items()] + [""]
    markdown = directory / "results.md"
    markdown.write_text("\n".join([*header, *_table(results, config, groups), ""]))
    return markdown
