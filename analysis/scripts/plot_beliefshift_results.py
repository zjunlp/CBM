#!/usr/bin/env python3
"""Plot BELIEFSHIFT analysis figures.

FSR/FUR/FIR are read from `category_percentages["belief_failure"]` in
`category_breakdown_summary.json`. They are failure rates, so lower raw
values are better. The main analysis figure plots failure reduction vs. Vanilla:
Vanilla - RL, where higher values are better.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
_MPLCONFIGDIR = ROOT / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "hatch.linewidth": 0.55,
        "axes.unicode_minus": False,
    }
)


DEFAULT_OUTPUTS_ROOT = ROOT / "outputs"
DEFAULT_FIGURES_DIR = ROOT / "figures"
FIG_DPI = 600

SCENARIOS = (
    ("task_a", "Scenario A"),
    ("task_b", "Scenario B"),
)


@dataclass(frozen=True)
class ModelSpec:
    folder: str
    label: str
    color: str
    hatch: str
    marker: str


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    pipeline: str
    x_labels: Sequence[str]
    bucket_keys: Sequence[str]
    x_axis_label: str
    ordered: bool


@dataclass(frozen=True)
class TrainingScenarioSpec:
    key: str
    label: str
    color: str
    marker: str
    linestyle: str


@dataclass(frozen=True)
class TrainingMetricSpec:
    key: str
    title: str
    challenge_type: str
    pipeline: str


@dataclass(frozen=True)
class TrainingLineSpec:
    key: str
    label: str
    challenge_type: str
    color: str
    marker: str
    linestyle: str


@dataclass(frozen=True)
class TrainingRunSpec:
    key: str
    label: str
    directory_train_key: str
    ckpts: Sequence[int]


MODELS: Sequence[ModelSpec] = (
    ModelSpec("base", "Vanilla", "#6d8fc6", "", "s"),
    ModelSpec("a_ckpt_520", "RL-RD", "#68b7ae", "//", "o"),
    ModelSpec("b_ckpt_338", "RL-CD", "#d9a06b", "..", "D"),
)

ADAPTERS: Sequence[ModelSpec] = MODELS[1:]

METRICS: Sequence[MetricSpec] = (
    MetricSpec(
        key="FSR",
        title="FSR",
        pipeline="failed_stay_depth",
        x_labels=("n=1", "n=5", "n=9"),
        bucket_keys=(
            "failed_stay_depth/n_redundant=1",
            "failed_stay_depth/n_redundant=5",
            "failed_stay_depth/n_redundant=9",
        ),
        x_axis_label="Redundant depth",
        ordered=True,
    ),
    MetricSpec(
        key="FUR",
        title="FUR",
        pipeline="failed_update_depth",
        x_labels=("delay=1", "delay=3", "delay=5"),
        bucket_keys=(
            "failed_update_depth/delay_turns=1",
            "failed_update_depth/delay_turns=3",
            "failed_update_depth/delay_turns=5",
        ),
        x_axis_label="Delay turns",
        ordered=True,
    ),
    MetricSpec(
        key="FIR",
        title="FIR",
        pipeline="noise_typology",
        x_labels=("None", "Sycophancy", "Authority", "Stress"),
        bucket_keys=(
            "noise_typology/noise_type=none",
            "noise_typology/noise_type=sycophancy",
            "noise_typology/noise_type=authority",
            "noise_typology/noise_type=stress",
        ),
        x_axis_label="Noise type",
        ordered=False,
    ),
)

TRAINING_CKPTS: Sequence[int] = (260, 390, 520)
TRAINING_DYNAMICS_CKPTS: Sequence[int] = (0, 260, 390, 520)
TRAINING_DYNAMICS_LABELS: Sequence[str] = ("ckpt-0", "260", "390", "520")
TRAINING_X_PAD = 0.22
TRAINING_RUNS: Sequence[TrainingRunSpec] = (
    TrainingRunSpec("train_a", "Train-A", "a", (260, 390, 520)),
    TrainingRunSpec("train_b", "Train-B", "b", (130, 234, 338)),
)
TRAINING_SCENARIOS: Sequence[TrainingScenarioSpec] = (
    TrainingScenarioSpec("task_a", "Rule Discovery", "#4c78a8", "o", "-"),
    TrainingScenarioSpec("task_b", "Circuit Diagnosis", "#f28e2b", "s", "--"),
)
TRAINING_METRICS: Sequence[TrainingMetricSpec] = (
    TrainingMetricSpec("FSR", "FSR", "failed_stay", "failed_stay_depth"),
    TrainingMetricSpec("FUR", "FUR", "failed_update", "failed_update_depth"),
    TrainingMetricSpec("FIR", "FIR", "failed_isolation", "noise_typology"),
)
TRAINING_LINES: Sequence[TrainingLineSpec] = (
    TrainingLineSpec("FSR", "FSR", "failed_stay", "#4c78a8", "o", "-"),
    TrainingLineSpec("FUR", "FUR", "failed_update", "#54a24b", "s", "--"),
    TrainingLineSpec("FIR", "FIR", "failed_isolation", "#f28e2b", "D", ":"),
)


CollectedValues = Dict[str, Dict[str, Dict[str, List[float]]]]


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_percentage_from_payload(payload: Dict[str, object], path: Path) -> float:
    percentages = payload.get("category_percentages")
    if isinstance(percentages, dict):
        return float(percentages.get("belief_failure", 0.0))

    overall = payload.get("overall")
    if isinstance(overall, dict):
        overall_percentages = overall.get("category_percentages")
        if isinstance(overall_percentages, dict):
            return float(overall_percentages.get("belief_failure", 0.0))

    raise ValueError(f"Missing category_percentages in {path}")


def _bucket_valid_percentages(summary_path: Path) -> Dict[str, float]:
    payload = _read_json(summary_path)
    buckets = payload.get("buckets")
    if not isinstance(buckets, dict):
        raise ValueError(f"Missing buckets in {summary_path}")

    out: Dict[str, float] = {}
    for bucket_key, bucket in buckets.items():
        if not isinstance(bucket, dict):
            continue
        percentages = bucket.get("category_percentages")
        if not isinstance(percentages, dict):
            continue
        out[str(bucket_key)] = float(percentages.get("belief_failure", 0.0))
    return out


def _collect_values(outputs_root: Path) -> CollectedValues:
    collected: CollectedValues = {}
    for scenario_key, _scenario_label in SCENARIOS:
        collected[scenario_key] = {}
        for metric in METRICS:
            collected[scenario_key][metric.key] = {}
            for model in MODELS:
                summary_path = (
                    outputs_root
                    / scenario_key
                    / "9B"
                    / metric.pipeline
                    / "eval"
                    / model.folder
                    / "category_breakdown_summary.json"
                )
                valid_by_bucket = _bucket_valid_percentages(summary_path)
                collected[scenario_key][metric.key][model.folder] = [
                    valid_by_bucket[bucket_key] for bucket_key in metric.bucket_keys
                ]
    return collected


def _format_value(value: float, *, signed: bool = False) -> str:
    if abs(value - round(value)) < 1e-6:
        text = f"{int(round(value))}"
    else:
        text = f"{value:.1f}"
    if signed and value > 0:
        return f"+{text}"
    return text


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", dpi=FIG_DPI)
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", dpi=FIG_DPI)


def _add_row_labels(fig: plt.Figure, axes: np.ndarray, row_labels: Sequence[str], x: float = 0.026) -> None:
    fig.canvas.draw()
    for row_idx, label in enumerate(row_labels):
        box = axes[row_idx, 0].get_position()
        fig.text(
            x,
            (box.y0 + box.y1) / 2.0,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="semibold",
        )


def _scenario_labels(scenario_keys: Sequence[str]) -> List[str]:
    label_by_key = {
        "task_a": "Rule Discovery",
        "task_b": "Circuit Diagnosis",
    }
    return [label_by_key.get(key, key.replace("_", " ").title()) for key in scenario_keys]


def _iter_reductions(
    collected: CollectedValues,
    scenario_keys: Sequence[str],
) -> Iterable[float]:
    for scenario_key in scenario_keys:
        for metric in METRICS:
            base = np.asarray(collected[scenario_key][metric.key]["base"], dtype=float)
            for adapter in ADAPTERS:
                adapter_values = np.asarray(collected[scenario_key][metric.key][adapter.folder], dtype=float)
                for value in base - adapter_values:
                    yield float(value)


def _reduction_ylim(values: Iterable[float]) -> Tuple[int, int]:
    values = list(values)
    min_value = min(values, default=0.0)
    max_value = max(values, default=0.0)
    lower = int(math.floor((min_value - 8.0) / 10.0) * 10)
    upper = int(math.ceil((max_value + 8.0) / 10.0) * 10)
    return min(-10, lower), max(20, upper)


def _style_reduction_axis(ax: plt.Axes, y_min: int, y_max: int, show_ylabel: bool) -> None:
    ax.axhspan(0, y_max, color="#f2faf6", zorder=0)
    if y_min < 0:
        ax.axhspan(y_min, 0, color="#fff5f1", zorder=0)
    ax.axhline(0, color="#303030", linewidth=0.95, zorder=2)
    ax.set_ylim(y_min, y_max)
    tick_start = int(math.ceil(y_min / 20.0) * 20)
    ax.set_yticks(np.arange(tick_start, y_max + 1, 20))
    ax.grid(axis="y", linestyle="--", color="#d2d2d2", linewidth=0.7, zorder=1)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3.5)
    ax.set_ylabel("Failure reduction vs. Vanilla (pp)" if show_ylabel else "", fontsize=10, labelpad=8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def _plot_reduction_panel(
    ax: plt.Axes,
    metric: MetricSpec,
    values_by_model: Dict[str, List[float]],
    *,
    y_min: int,
    y_max: int,
    show_title: bool,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    x = np.arange(len(metric.x_labels))
    offsets = np.linspace(-0.10, 0.10, len(ADAPTERS))
    base = np.asarray(values_by_model["base"], dtype=float)

    for adapter, offset in zip(ADAPTERS, offsets):
        adapter_values = np.asarray(values_by_model[adapter.folder], dtype=float)
        reductions = base - adapter_values
        xpos = x + offset
        ax.vlines(xpos, 0, reductions, color=adapter.color, linewidth=2.0, alpha=0.75, zorder=3)
        if metric.ordered:
            ax.plot(
                xpos,
                reductions,
                color=adapter.color,
                marker=adapter.marker,
                markersize=5.7,
                markeredgecolor="#2f2f2f",
                markeredgewidth=0.75,
                linewidth=1.4,
                alpha=0.98,
                zorder=4,
            )
        else:
            ax.scatter(
                xpos,
                reductions,
                s=42,
                marker=adapter.marker,
                facecolor=adapter.color,
                edgecolor="#2f2f2f",
                linewidth=0.75,
                alpha=0.98,
                zorder=4,
            )
        for xi, value in zip(xpos, reductions):
            if value >= 0:
                label_y = value + 2.0
                va = "bottom"
            else:
                label_y = value - 2.0
                va = "top"
            ax.text(
                xi,
                label_y,
                _format_value(float(value), signed=True),
                ha="center",
                va=va,
                fontsize=7.2,
                color="#222222",
                zorder=5,
            )

    rotation = 0
    ha = "center"
    ax.set_xticks(x)
    ax.set_xticklabels(metric.x_labels, rotation=rotation, ha=ha, fontsize=8 if metric.key == "FIR" else 9)
    ax.set_xlabel(metric.x_axis_label if show_xlabel else "", fontsize=10, labelpad=7)
    ax.set_title(metric.title if show_title else "", fontsize=12, fontweight="semibold", pad=8)
    ax.margins(x=0.06)
    _style_reduction_axis(ax, y_min, y_max, show_ylabel)


def _plot_reduction_figure(
    scenario_keys: Sequence[str],
    collected: CollectedValues,
    output_path: Path,
    row_labels: Sequence[str] | None = None,
) -> None:
    nrows = len(scenario_keys)
    ncols = len(METRICS)
    y_min, y_max = _reduction_ylim(_iter_reductions(collected, scenario_keys))
    fig_w = 11.8
    fig_h = 2.85 if nrows == 1 else 5.65

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=FIG_DPI)
    if nrows == 1:
        axes = np.array([axes])
    axes = np.asarray(axes)
    fig.patch.set_facecolor("white")

    for row_idx, scenario_key in enumerate(scenario_keys):
        for col_idx, metric in enumerate(METRICS):
            _plot_reduction_panel(
                axes[row_idx, col_idx],
                metric,
                collected[scenario_key][metric.key],
                y_min=y_min,
                y_max=y_max,
                show_title=row_idx == 0,
                show_ylabel=col_idx == 0,
                show_xlabel=nrows == 1 or row_idx == nrows - 1,
            )

    fig.subplots_adjust(
        left=0.08 if nrows == 1 else 0.10,
        right=0.995,
        bottom=0.22 if nrows == 1 else 0.13,
        top=0.80 if nrows == 1 else 0.86,
        wspace=0.24,
        hspace=0.34,
    )

    if row_labels is None:
        row_labels = _scenario_labels(scenario_keys)
    _add_row_labels(fig, axes, row_labels, x=0.025 if nrows == 1 else 0.031)

    handles = [
        Line2D(
            [0],
            [0],
            color=adapter.color,
            marker=adapter.marker,
            markeredgecolor="#2f2f2f",
            markeredgewidth=0.75,
            linewidth=1.6,
            markersize=6.2,
            label=adapter.label,
        )
        for adapter in ADAPTERS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=1.9,
        columnspacing=1.6,
    )
    fig.text(
        0.992,
        0.955,
        "Higher is better ↑",
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )

    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_reduction_heatmap(
    scenario_keys: Sequence[str],
    collected: CollectedValues,
    output_path: Path,
) -> None:
    rows: List[str] = []
    values: List[List[float]] = []
    group_boundaries: List[int] = []

    for metric in METRICS:
        if rows:
            group_boundaries.append(len(rows) - 0.5)
        for idx, x_label in enumerate(metric.x_labels):
            rows.append(f"{metric.key}: {x_label}")
            row: List[float] = []
            for scenario_key in scenario_keys:
                base = np.asarray(collected[scenario_key][metric.key]["base"], dtype=float)
                for adapter in ADAPTERS:
                    adapter_values = np.asarray(collected[scenario_key][metric.key][adapter.folder], dtype=float)
                    row.append(float(base[idx] - adapter_values[idx]))
            values.append(row)

    matrix = np.asarray(values, dtype=float)
    scenario_labels = _scenario_labels(scenario_keys)
    short_scenario_labels = {
        "Rule Discovery": "Rule",
        "Circuit Diagnosis": "Circuit",
    }
    columns = [
        f"{short_scenario_labels.get(scenario_label, scenario_label)}\n{adapter.label}"
        for scenario_label in scenario_labels
        for adapter in ADAPTERS
    ]

    cmap = LinearSegmentedColormap.from_list("beliefshift_reduction_teal", ["#ffffff", "#dcefee", "#69b8b0"])
    max_abs = max(1.0, float(np.nanmax(matrix)))
    norm = Normalize(vmin=0, vmax=max_abs)

    fig_w = 3.5 if len(columns) <= 4 else 5.0
    fig_h = 3.55
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIG_DPI)
    fig.patch.set_facecolor("white")
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, fontsize=6.8)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.8)
    ax.tick_params(axis="y", length=0, pad=3)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                _format_value(value, signed=True),
                ha="center",
                va="center",
                fontsize=6.6,
                fontweight="semibold" if value >= 50 else "normal",
                color="#111111",
            )

    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for boundary in group_boundaries:
        ax.axhline(boundary, color="#333333", linewidth=0.8)
    for col_boundary in range(2, len(columns), 2):
        ax.axvline(col_boundary - 0.5, color="#333333", linewidth=0.8)

    ax.set_title("Failure Reduction (pp)", loc="left", fontsize=8.0, fontweight="semibold", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=6.0, length=2.0, width=0.6)
    cbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.27, right=0.93, top=0.86, bottom=0.08)
    _save_figure(fig, output_path)
    plt.close(fig)


def _style_absolute_axis(ax: plt.Axes, show_ylabel: bool) -> None:
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", color="#d6d6d6", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3.5)
    ax.set_ylabel("Failure Rate (%)" if show_ylabel else "", fontsize=10, labelpad=8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def _plot_absolute_panel(
    ax: plt.Axes,
    metric: MetricSpec,
    values_by_model: Dict[str, List[float]],
    *,
    show_title: bool,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    x = np.arange(len(metric.x_labels))

    if metric.ordered:
        for model in MODELS:
            values = np.asarray(values_by_model[model.folder], dtype=float)
            ax.plot(
                x,
                values,
                color=model.color,
                marker=model.marker,
                markersize=5.6,
                markeredgecolor="#2f2f2f",
                markeredgewidth=0.75,
                linewidth=1.75 if model.folder == "base" else 1.55,
                alpha=0.98,
                zorder=3 if model.folder == "base" else 4,
            )
            ax.text(
                x[-1] + 0.06,
                values[-1],
                _format_value(float(values[-1])),
                ha="left",
                va="center",
                fontsize=7.5,
                fontweight="semibold",
                color=model.color,
                zorder=5,
            )
    else:
        bar_width = min(0.30, 0.98 / len(MODELS))
        offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2.0) * bar_width
        all_values = np.asarray([values_by_model[model.folder] for model in MODELS], dtype=float)
        best_by_group = all_values.min(axis=0)
        for model_idx, (model, offset) in enumerate(zip(MODELS, offsets)):
            values = np.asarray(values_by_model[model.folder], dtype=float)
            bars = ax.bar(
                x + offset,
                values,
                width=bar_width,
                color=model.color,
                edgecolor="#2f2f2f",
                hatch=model.hatch,
                linewidth=0.75,
                alpha=0.9,
                zorder=3,
            )
            for group_idx, (bar, value) in enumerate(zip(bars, values)):
                is_best = np.isclose(value, best_by_group[group_idx])
                if value >= 96:
                    label_y = value - 3.2
                    va = "top"
                else:
                    label_y = value + 2.0
                    va = "bottom"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    _format_value(float(value)),
                    ha="center",
                    va=va,
                    fontsize=7.1,
                    fontweight="semibold" if is_best else "normal",
                    color="#222222",
                    zorder=5,
                )

    rotation = 0
    ha = "center"
    ax.set_xticks(x)
    ax.set_xticklabels(metric.x_labels, rotation=rotation, ha=ha, fontsize=8 if metric.key == "FIR" else 9)
    ax.set_xlabel(metric.x_axis_label if show_xlabel else "", fontsize=10, labelpad=7)
    ax.set_title(metric.title if show_title else "", fontsize=12, fontweight="semibold", pad=8)
    ax.margins(x=0.035 if metric.ordered else 0.12)
    _style_absolute_axis(ax, show_ylabel=show_ylabel)


def _plot_absolute_figure(
    scenario_keys: Sequence[str],
    collected: CollectedValues,
    output_path: Path,
    row_labels: Sequence[str] | None = None,
) -> None:
    nrows = len(scenario_keys)
    ncols = len(METRICS)
    fig_w = 11.6
    fig_h = 2.35 if nrows == 1 else 4.55

    width_ratios = [0.90, 0.90, 1.30]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=FIG_DPI, gridspec_kw={"width_ratios": width_ratios})
    if nrows == 1:
        axes = np.array([axes])
    axes = np.asarray(axes)
    fig.patch.set_facecolor("white")

    for row_idx, scenario_key in enumerate(scenario_keys):
        for col_idx, metric in enumerate(METRICS):
            _plot_absolute_panel(
                axes[row_idx, col_idx],
                metric,
                collected[scenario_key][metric.key],
                show_title=row_idx == 0,
                show_ylabel=col_idx == 0,
                show_xlabel=nrows == 1 or row_idx == nrows - 1,
            )

    fig.subplots_adjust(
        left=0.075 if nrows == 1 else 0.092,
        right=0.995,
        bottom=0.24 if nrows == 1 else 0.14,
        top=0.74 if nrows == 1 else 0.80,
        wspace=0.18,
        hspace=0.26,
    )

    if nrows > 1:
        if row_labels is None:
            row_labels = _scenario_labels(scenario_keys)
        _add_row_labels(fig, axes, row_labels, x=0.029)

    handles = [
        Line2D(
            [0],
            [0],
            color=model.color,
            marker=model.marker,
            markeredgecolor="#2f2f2f",
            markeredgewidth=0.75,
            linewidth=1.65,
            markersize=6.2,
            label=model.label,
        )
        for model in MODELS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.9,
        columnspacing=1.6,
    )
    fig.text(
        0.992,
        0.935,
        "Lower is better ↓",
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )

    _save_figure(fig, output_path)
    plt.close(fig)


def _style_raw_axis(ax: plt.Axes, show_ylabel: bool) -> None:
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", color="#d6d6d6", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=8.8, width=0.8, length=3.5)
    ax.set_ylabel("Failure Rate (%)" if show_ylabel else "", fontsize=10, labelpad=8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)


def _plot_raw_panel(
    ax: plt.Axes,
    metric: MetricSpec,
    values_by_model: Dict[str, List[float]],
    *,
    show_title: bool,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    x = np.arange(len(metric.x_labels))
    bar_width = min(0.22, 0.78 / len(MODELS))
    all_series = np.asarray([values_by_model[model.folder] for model in MODELS])
    best_by_group = all_series.min(axis=0)

    for model_idx, model in enumerate(MODELS):
        offsets = (model_idx - (len(MODELS) - 1) / 2.0) * bar_width
        series = values_by_model[model.folder]
        bars = ax.bar(
            x + offsets,
            series,
            width=bar_width,
            label=model.label,
            color=model.color,
            edgecolor="#4b4b4b",
            linewidth=0.7,
            alpha=0.9,
            hatch=model.hatch,
            zorder=3,
        )
        for group_idx, (bar, value) in enumerate(zip(bars, series)):
            is_best = np.isclose(value, best_by_group[group_idx])
            if is_best:
                bar.set_edgecolor("#111111")
                bar.set_linewidth(1.15)

            if value >= 98:
                label_y = value - 3.5
                va = "top"
            else:
                label_y = value + 1.5
                va = "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                _format_value(value),
                ha="center",
                va=va,
                fontsize=7.0,
                fontweight="semibold" if is_best else "normal",
                color="#222222",
                zorder=4,
            )

    rotation = 22 if metric.key == "FIR" else 0
    ha = "right" if metric.key == "FIR" else "center"
    ax.set_xticks(x)
    ax.set_xticklabels(metric.x_labels, rotation=rotation, ha=ha, fontsize=8 if metric.key == "FIR" else 8.8)
    ax.set_xlabel(metric.x_axis_label if show_xlabel else "", fontsize=10, labelpad=7)
    ax.set_title(metric.title if show_title else "", fontsize=12, fontweight="semibold", pad=8)
    ax.margins(x=0.045)
    _style_raw_axis(ax, show_ylabel=show_ylabel)


def _plot_raw_figure(
    scenario_keys: Sequence[str],
    collected: CollectedValues,
    output_path: Path,
    row_labels: Sequence[str] | None = None,
) -> None:
    nrows = len(scenario_keys)
    ncols = len(METRICS)
    fig_w = 11.8
    fig_h = 2.85 if nrows == 1 else 5.65

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=FIG_DPI)
    if nrows == 1:
        axes = np.array([axes])
    axes = np.asarray(axes)
    fig.patch.set_facecolor("white")

    for row_idx, scenario_key in enumerate(scenario_keys):
        for col_idx, metric in enumerate(METRICS):
            _plot_raw_panel(
                axes[row_idx, col_idx],
                metric,
                collected[scenario_key][metric.key],
                show_title=row_idx == 0,
                show_ylabel=col_idx == 0,
                show_xlabel=nrows == 1 or row_idx == nrows - 1,
            )

    fig.subplots_adjust(
        left=0.08 if nrows == 1 else 0.10,
        right=0.995,
        bottom=0.22 if nrows == 1 else 0.13,
        top=0.80 if nrows == 1 else 0.86,
        wspace=0.24,
        hspace=0.34,
    )

    if row_labels is None:
        row_labels = _scenario_labels(scenario_keys)
    _add_row_labels(fig, axes, row_labels, x=0.025 if nrows == 1 else 0.031)

    handles = [
        Patch(facecolor=model.color, edgecolor="#4b4b4b", hatch=model.hatch, label=model.label)
        for model in MODELS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.9,
        columnspacing=1.6,
    )
    fig.text(
        0.992,
        0.955,
        "Lower is better ↓",
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )

    _save_figure(fig, output_path)
    plt.close(fig)


def _training_run_dir(scenario_key: str, ckpt: int) -> Path:
    scenario_suffix = "a" if scenario_key == "task_a" else "b"
    outputs_root = REPO_ROOT / scenario_key / "outputs"
    pattern = f"*test_{scenario_suffix}_ckpt_{ckpt}"
    matches = sorted(outputs_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing training checkpoint directory for {scenario_key} ckpt={ckpt}: {outputs_root / pattern}")
    return matches[0]


def _collect_training_values() -> Dict[str, Dict[str, List[float]]]:
    collected: Dict[str, Dict[str, List[float]]] = {}
    for scenario in TRAINING_SCENARIOS:
        collected[scenario.key] = {}
        for metric in TRAINING_METRICS:
            series: List[float] = []
            for ckpt in TRAINING_CKPTS:
                summary_path = _training_run_dir(scenario.key, ckpt) / "lora" / metric.challenge_type / "stats_report.json"
                payload = _read_json(summary_path)
                series.append(_valid_percentage_from_payload(payload, summary_path))
            collected[scenario.key][metric.key] = series
    return collected


def _style_training_axis(ax: plt.Axes, show_ylabel: bool) -> None:
    ax.set_ylim(0, 42)
    ax.set_yticks(np.arange(0, 41, 10))
    ax.grid(axis="y", linestyle="--", color="#d6d6d6", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3.5)
    ax.set_ylabel("Failure Rate (%)" if show_ylabel else "", fontsize=10, labelpad=8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def _plot_training_panel(
    ax: plt.Axes,
    metric: TrainingMetricSpec,
    values_by_scenario: Dict[str, List[float]],
    scenario_specs: Sequence[TrainingScenarioSpec],
    *,
    show_title: bool,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    x = np.arange(len(TRAINING_CKPTS))
    if len(scenario_specs) == 1:
        scenario_offsets = {scenario_specs[0].key: 1.35}
    else:
        scenario_offsets = {
            scenario.key: (1.35 if idx == 0 else -1.7) for idx, scenario in enumerate(scenario_specs)
        }

    for scenario in scenario_specs:
        values = np.asarray(values_by_scenario[scenario.key], dtype=float)
        ax.plot(
            x,
            values,
            color=scenario.color,
            marker=scenario.marker,
            markersize=5.8,
            markeredgecolor="#2f2f2f",
            markeredgewidth=0.75,
            linewidth=1.75,
            linestyle=scenario.linestyle,
            alpha=0.98,
            zorder=3,
        )
        for xi, value in zip(x, values):
            label_y = value + scenario_offsets[scenario.key]
            va = "bottom" if scenario_offsets[scenario.key] > 0 else "top"
            ax.text(
                xi,
                label_y,
                _format_value(float(value)),
                ha="center",
                va=va,
                fontsize=7.2,
                fontweight="semibold",
                color=scenario.color,
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(ckpt) for ckpt in TRAINING_CKPTS], fontsize=8.8)
    ax.set_xlabel("Training Step" if show_xlabel else "", fontsize=10, labelpad=7)
    ax.set_title(metric.title if show_title else "", fontsize=12, fontweight="semibold", pad=8)
    ax.margins(x=0.08)
    _style_training_axis(ax, show_ylabel=show_ylabel)


def _plot_training_figure(output_path: Path, scenario_keys: Sequence[str]) -> None:
    collected = _collect_training_values()
    selected_scenarios = [scenario for scenario in TRAINING_SCENARIOS if scenario.key in scenario_keys]
    fig_w = 11.2
    fig_h = 2.75
    fig, axes = plt.subplots(nrows=1, ncols=len(TRAINING_METRICS), figsize=(fig_w, fig_h), dpi=FIG_DPI)
    axes = np.asarray(axes)
    fig.patch.set_facecolor("white")

    for col_idx, metric in enumerate(TRAINING_METRICS):
        _plot_training_panel(
            axes[col_idx],
            metric,
            {scenario.key: collected[scenario.key][metric.key] for scenario in selected_scenarios},
            selected_scenarios,
            show_title=True,
            show_ylabel=col_idx == 0,
            show_xlabel=False,
        )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.22,
        top=0.77,
        wspace=0.24,
    )

    if len(selected_scenarios) > 1:
        handles = [
            Line2D(
                [0],
                [0],
                color=scenario.color,
                marker=scenario.marker,
                markeredgecolor="#2f2f2f",
                markeredgewidth=0.75,
                linewidth=1.65,
                linestyle=scenario.linestyle,
                markersize=6.0,
                label=scenario.label,
            )
            for scenario in selected_scenarios
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=len(selected_scenarios),
            frameon=False,
            fontsize=9.5,
            handlelength=2.0,
            columnspacing=1.6,
        )
    fig.text(
        0.992,
        0.945,
        "Lower is better \u2193",
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )
    fig.text(0.5, 0.07, "Training checkpoint", ha="center", va="center", fontsize=10)

    _save_figure(fig, output_path)
    plt.close(fig)


def _style_compact_training_axis(ax: plt.Axes, show_ylabel: bool) -> None:
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", color="#d6d6d6", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=5.8, width=0.8, length=3.5)
    ax.set_ylabel("Failure Rate (%)" if show_ylabel else "", fontsize=7.5, labelpad=5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def _plot_compact_training_panel(
    ax: plt.Axes,
    scenario: TrainingScenarioSpec,
    values_by_line: Dict[str, List[float]],
    *,
    show_ylabel: bool,
) -> None:
    x = np.arange(len(TRAINING_DYNAMICS_CKPTS), dtype=float)
    ax.set_title(scenario.label, fontsize=7.6, fontweight="semibold", pad=3)

    for line in TRAINING_LINES:
        values = np.asarray(values_by_line[line.key], dtype=float)
        ax.plot(
            x,
            values,
            color=line.color,
            marker=line.marker,
            markersize=5.2,
            markeredgecolor="#2f2f2f",
            markeredgewidth=0.75,
            linewidth=1.65,
            linestyle=line.linestyle,
            alpha=0.98,
            zorder=3,
            label=line.label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TRAINING_DYNAMICS_LABELS, fontsize=5.2)
    pad = TRAINING_X_PAD
    ax.set_xlim(-pad, x[-1] + pad)
    _style_compact_training_axis(ax, show_ylabel=show_ylabel)
    ax.margins(x=0)


def _compact_base_stats_path(outputs_root: Path, scenario_key: str, metric: TrainingMetricSpec) -> Path:
    return outputs_root / scenario_key / "9B" / metric.pipeline / "eval" / "base" / "base" / "stats_report.json"


def _collect_compact_training_values(outputs_root: Path, scenario_keys: Sequence[str]) -> Dict[str, Dict[str, List[float]]]:
    collected: Dict[str, Dict[str, List[float]]] = {}
    for scenario in TRAINING_SCENARIOS:
        if scenario.key not in scenario_keys:
            continue
        collected[scenario.key] = {}
        for metric in TRAINING_METRICS:
            series: List[float] = []

            base_candidates = (
                _compact_base_stats_path(outputs_root, scenario.key, metric),
                outputs_root / scenario.key / "9B" / metric.pipeline / "eval" / "base" / "category_breakdown_summary.json",
            )
            for candidate in base_candidates:
                if candidate.exists():
                    series.append(_valid_percentage_from_payload(_read_json(candidate), candidate))
                    break
            else:
                raise FileNotFoundError(
                    f"Missing base checkpoint result for {scenario.key} / {metric.pipeline}: {base_candidates[0]}"
                )

            for ckpt in TRAINING_CKPTS:
                summary_path = _training_run_dir(scenario.key, ckpt) / "lora" / metric.challenge_type / "stats_report.json"
                payload = _read_json(summary_path)
                series.append(_valid_percentage_from_payload(payload, summary_path))
            collected[scenario.key][metric.key] = series
    return collected


def _training_rollout_stats_path(test_scenario_key: str, train_run: TrainingRunSpec, ckpt: int, metric: TrainingMetricSpec) -> Path:
    test_suffix = "a" if test_scenario_key == "task_a" else "b"
    run_dir = (
        REPO_ROOT
        / test_scenario_key
        / "outputs"
        / f"swift_train_{train_run.directory_train_key}_with_thinking_rollout_8_test_{test_suffix}_ckpt_{ckpt}"
    )
    return run_dir / "lora" / metric.challenge_type / "stats_report.json"


def _collect_cross_training_values(
    outputs_root: Path,
    scenario_keys: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    collected: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for train_run in TRAINING_RUNS:
        collected[train_run.key] = {}
        for scenario in TRAINING_SCENARIOS:
            if scenario.key not in scenario_keys:
                continue
            collected[train_run.key][scenario.key] = {}
            for metric in TRAINING_METRICS:
                series: List[float] = []
                base_candidates = (
                    _compact_base_stats_path(outputs_root, scenario.key, metric),
                    outputs_root / scenario.key / "9B" / metric.pipeline / "eval" / "base" / "category_breakdown_summary.json",
                )
                for candidate in base_candidates:
                    if candidate.exists():
                        series.append(_valid_percentage_from_payload(_read_json(candidate), candidate))
                        break
                else:
                    raise FileNotFoundError(
                        f"Missing base checkpoint result for {scenario.key} / {metric.pipeline}: {base_candidates[0]}"
                    )

                for ckpt in train_run.ckpts:
                    summary_path = _training_rollout_stats_path(scenario.key, train_run, ckpt, metric)
                    if not summary_path.exists():
                        raise FileNotFoundError(f"Missing rollout checkpoint result: {summary_path}")
                    series.append(_valid_percentage_from_payload(_read_json(summary_path), summary_path))
                collected[train_run.key][scenario.key][metric.key] = series
    return collected


def _plot_compact_training_figure(output_path: Path, outputs_root: Path, scenario_keys: Sequence[str]) -> None:
    selected_scenarios = [scenario for scenario in TRAINING_SCENARIOS if scenario.key in scenario_keys]
    collected = _collect_compact_training_values(outputs_root, scenario_keys)
    ncols = len(selected_scenarios)
    fig_w = 4.30 if ncols == 2 else 3.25
    fig_h = 2.25

    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(fig_w, fig_h), dpi=FIG_DPI)
    if ncols == 1:
        axes = np.array([axes])
    axes = np.asarray(axes)
    fig.patch.set_facecolor("white")

    metric_labels = [line.label for line in TRAINING_LINES]
    cmap = LinearSegmentedColormap.from_list("training_failure_rate", ["#6ab8ae", "#edf2f1", "#e4a56e"])
    norm = Normalize(vmin=0, vmax=100)
    image = None
    for idx, scenario in enumerate(selected_scenarios):
        matrix = np.asarray([collected[scenario.key][line.key] for line in TRAINING_LINES], dtype=float)
        ax = axes[idx]
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(scenario.label, fontsize=7.8, fontweight="semibold", pad=4)
        ax.set_xticks(np.arange(len(TRAINING_DYNAMICS_LABELS)))
        ax.set_xticklabels(TRAINING_DYNAMICS_LABELS, fontsize=5.8)
        ax.set_yticks(np.arange(len(metric_labels)))
        ax.set_yticklabels(metric_labels if idx == 0 else [], fontsize=6.6)
        ax.tick_params(axis="both", length=0, pad=2.5)
        ax.set_xticks(np.arange(-0.5, len(TRAINING_DYNAMICS_LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = float(matrix[row_idx, col_idx])
                ax.text(
                    col_idx,
                    row_idx,
                    _format_value(value),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    fontweight="semibold" if col_idx in {0, matrix.shape[1] - 1} else "normal",
                    color="#111111",
                )
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(
        left=0.11 if ncols == 1 else 0.07,
        right=0.90 if ncols == 1 else 0.93,
        bottom=0.22,
        top=0.78,
        wspace=0.18,
    )

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.025)
        cbar.ax.tick_params(labelsize=5.8, length=2.0, width=0.55)
        cbar.outline.set_linewidth(0.5)
    fig.text(
        0.93,
        0.93,
        "Lower is better \u2193",
        ha="right",
        va="top",
        fontsize=6.4,
        color="#444444",
    )
    fig.text(0.5, 0.10, "Training checkpoint", ha="center", va="center", fontsize=7.2)

    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_checkpoint_slope_figure(output_path: Path, outputs_root: Path, scenario_keys: Sequence[str]) -> None:
    selected_scenarios = [scenario for scenario in TRAINING_SCENARIOS if scenario.key in scenario_keys]
    selected_train_runs = list(TRAINING_RUNS)
    collected = _collect_cross_training_values(outputs_root, scenario_keys)
    ncols = len(selected_scenarios)
    nrows = len(selected_train_runs)
    fig_w = 4.85 if ncols == 2 else 3.05
    fig_h = 3.35

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=FIG_DPI, sharey=True)
    axes = np.asarray(axes)
    if nrows == 1 and ncols == 1:
        axes = axes.reshape(1, 1)
    elif nrows == 1:
        axes = axes.reshape(1, ncols)
    elif ncols == 1:
        axes = axes.reshape(nrows, 1)
    fig.patch.set_facecolor("white")
    final_label_offsets = {"FSR": -3.2, "FUR": 3.2, "FIR": 0.0}

    for row_idx, train_run in enumerate(selected_train_runs):
        x_labels = ("ckpt-0",) + tuple(str(ckpt) for ckpt in train_run.ckpts)
        x = np.arange(len(x_labels), dtype=float)
        for col_idx, scenario in enumerate(selected_scenarios):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(scenario.label, fontsize=7.2, fontweight="semibold", pad=3)
            if col_idx == 0:
                ax.text(
                    -0.23,
                    0.5,
                    train_run.label,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=6.8,
                    fontweight="semibold",
                )

            for line in TRAINING_LINES:
                values = np.asarray(collected[train_run.key][scenario.key][line.key], dtype=float)
                ax.fill_between(x, values, 0, color=line.color, alpha=0.045, linewidth=0, zorder=1)
                ax.plot(
                    x,
                    values,
                    color=line.color,
                    marker=line.marker,
                    markersize=3.7,
                    markerfacecolor=line.color,
                    markeredgecolor="#2f2f2f",
                    markeredgewidth=0.6,
                    linewidth=1.25,
                    linestyle=line.linestyle,
                    alpha=0.98,
                    zorder=3,
                    label=line.label,
                )
                ax.scatter(
                    [x[-1]],
                    [values[-1]],
                    s=30,
                    marker=line.marker,
                    facecolor=line.color,
                    edgecolor="white",
                    linewidth=0.85,
                    zorder=4,
                )
                ax.text(
                    x[-1] + 0.06,
                    min(98.0, max(2.0, values[-1] + final_label_offsets.get(line.key, 0.0))),
                    _format_value(float(values[-1])),
                    ha="left",
                    va="center",
                    fontsize=5.1,
                    fontweight="semibold",
                    color=line.color,
                    zorder=5,
                )

            ax.set_ylim(0, 100)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=4.9)
            ax.set_xlim(-0.16, x[-1] + 0.45)
            ax.grid(axis="y", linestyle="--", color="#d6d6d6", linewidth=0.55, zorder=0)
            ax.tick_params(axis="both", labelsize=5.2, width=0.65, length=2.5)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.75)
            ax.spines["bottom"].set_linewidth(0.75)
            ax.spines["left"].set_color("#333333")
            ax.spines["bottom"].set_color("#333333")
            if col_idx == 0:
                ax.set_ylabel("Failure Rate (%)", fontsize=6.1, labelpad=3)

    handles = [
        Line2D(
            [0],
            [0],
            color=line.color,
            marker=line.marker,
            markeredgecolor="#2f2f2f",
            markeredgewidth=0.65,
            linewidth=1.35,
            linestyle=line.linestyle,
            markersize=4.2,
            label=line.label,
        )
        for line in TRAINING_LINES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=6.7,
        handlelength=1.6,
        columnspacing=0.85,
    )
    fig.text(0.955, 0.955, "Lower is better \u2193", ha="right", va="top", fontsize=5.8, color="#444444")
    fig.text(0.5, 0.075, "Training checkpoint", ha="center", va="center", fontsize=6.5)
    fig.subplots_adjust(
        left=0.12 if ncols == 2 else 0.18,
        right=0.965,
        bottom=0.17,
        top=0.84,
        hspace=0.34,
        wspace=0.16,
    )

    _save_figure(fig, output_path)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot BELIEFSHIFT failure-rate analysis figures")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Root directory containing analysis/outputs",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for the generated figures",
    )
    parser.add_argument(
        "--scenario",
        choices=["a", "b", "combined", "all"],
        default="all",
        help="Which scenario figure set to generate",
    )
    parser.add_argument(
        "--plot-type",
        choices=["absolute", "reduction", "raw", "heatmap", "dynamics", "all"],
        default="all",
        help="Generate absolute-rate, Vanilla-RL reduction, raw bar, heatmap, dynamics, or all figure types",
    )
    return parser


def _scenario_sets(selection: str) -> Sequence[Tuple[str, Sequence[str]]]:
    if selection == "a":
        return (("task_a", ("task_a",)),)
    if selection == "b":
        return (("task_b", ("task_b",)),)
    if selection == "combined":
        return (("combined", ("task_a", "task_b")),)
    return (
        ("task_a", ("task_a",)),
        ("task_b", ("task_b",)),
        ("combined", ("task_a", "task_b")),
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    collected = _collect_values(args.outputs_root)

    for suffix, scenario_keys in _scenario_sets(args.scenario):
        if args.plot_type in {"absolute", "all"}:
            _plot_absolute_figure(
                scenario_keys,
                collected,
                args.figures_dir / f"beliefshift_absolute_rates_{suffix}.png",
            )
        if args.plot_type in {"reduction", "all"}:
            _plot_reduction_figure(
                scenario_keys,
                collected,
                args.figures_dir / f"beliefshift_failure_reduction_{suffix}.png",
            )
        if args.plot_type in {"heatmap", "all"}:
            _plot_reduction_heatmap(
                scenario_keys,
                collected,
                args.figures_dir / f"beliefshift_failure_reduction_heatmap_{suffix}.png",
            )
        if args.plot_type in {"raw", "all"}:
            _plot_raw_figure(
                scenario_keys,
                collected,
                args.figures_dir / f"beliefshift_analysis_results_{suffix}.png",
            )
        if args.plot_type in {"dynamics", "all"}:
            _plot_checkpoint_slope_figure(
                args.figures_dir / f"beliefshift_checkpoint_dynamics_{suffix}.png",
                args.outputs_root,
                scenario_keys,
            )
            _plot_checkpoint_slope_figure(
                args.figures_dir / f"beliefshift_checkpoint_slope_{suffix}.png",
                args.outputs_root,
                scenario_keys,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
