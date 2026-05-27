#!/usr/bin/env python3
"""Plot the combined absolute-rate figure with the BT-Prompt baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.scripts import plot_beliefshift_results as base_plot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot BELIEFSHIFT absolute rates with BT-Prompt.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=base_plot.DEFAULT_OUTPUTS_ROOT / "analysis_results",
        help="Root directory containing task_a/task_b analysis results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_plot.DEFAULT_FIGURES_DIR / "beliefshift_absolute_rates_combined_bt_prompt.png",
        help="Output PNG path. A matching PDF is also written.",
    )
    return parser


def _plot_absolute_panel_with_bt_spacing(
    ax: plt.Axes,
    metric: base_plot.MetricSpec,
    values_by_model: dict[str, list[float]],
    *,
    show_title: bool,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    if metric.ordered:
        base_plot._plot_absolute_panel(
            ax,
            metric,
            values_by_model,
            show_title=show_title,
            show_ylabel=show_ylabel,
            show_xlabel=show_xlabel,
        )
        return

    group_spacing = 1.14
    x = np.arange(len(metric.x_labels)) * group_spacing
    bar_width = 0.245
    offset_step = bar_width
    offsets = (np.arange(len(base_plot.MODELS)) - (len(base_plot.MODELS) - 1) / 2.0) * offset_step
    all_values = np.asarray([values_by_model[model.folder] for model in base_plot.MODELS], dtype=float)
    best_by_group = all_values.min(axis=0)

    for model, offset in zip(base_plot.MODELS, offsets):
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
                base_plot._format_value(float(value)),
                ha="center",
                va=va,
                fontsize=7.1,
                fontweight="semibold" if is_best else "normal",
                color="#222222",
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric.x_labels, rotation=0, ha="center", fontsize=8)
    ax.set_xlabel(metric.x_axis_label if show_xlabel else "", fontsize=10, labelpad=7)
    ax.set_title(metric.title if show_title else "", fontsize=12, fontweight="semibold", pad=8)
    half_group = (len(base_plot.MODELS) - 1) * offset_step / 2.0 + bar_width / 2.0
    side_pad = 0.10
    ax.set_xlim(x[0] - half_group - side_pad, x[-1] + half_group + side_pad)
    base_plot._style_absolute_axis(ax, show_ylabel=show_ylabel)


def _plot_combined_absolute_figure(collected: base_plot.CollectedValues, output: Path) -> None:
    scenario_keys = ("task_a", "task_b")
    nrows = len(scenario_keys)
    ncols = len(base_plot.METRICS)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12.1, 4.55),
        dpi=base_plot.FIG_DPI,
        gridspec_kw={"width_ratios": [0.90, 0.90, 1.30]},
    )
    axes = np.asarray(axes)
    fig.patch.set_facecolor("white")

    for row_idx, scenario_key in enumerate(scenario_keys):
        for col_idx, metric in enumerate(base_plot.METRICS):
            _plot_absolute_panel_with_bt_spacing(
                axes[row_idx, col_idx],
                metric,
                collected[scenario_key][metric.key],
                show_title=row_idx == 0,
                show_ylabel=col_idx == 0,
                show_xlabel=row_idx == nrows - 1,
            )

    fig.subplots_adjust(
        left=0.088,
        right=0.995,
        bottom=0.14,
        top=0.80,
        wspace=0.18,
        hspace=0.26,
    )
    base_plot._add_row_labels(fig, axes, ("Rule Discovery", "Circuit Diagnosis"), x=0.026)

    handles = []
    labels = []
    for model in base_plot.MODELS:
        line_handle = Line2D(
            [0, 1],
            [0, 0],
            color=model.color,
            marker=model.marker,
            markeredgecolor="#2f2f2f",
            markeredgewidth=0.75,
            linewidth=1.65,
            markersize=5.6,
        )
        patch_handle = Patch(
            facecolor=model.color,
            edgecolor="#2f2f2f",
            hatch=model.hatch,
            linewidth=0.75,
            alpha=0.9,
        )
        handles.append((line_handle, patch_handle))
        labels.append(model.label)

    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.955),
        ncol=len(base_plot.MODELS),
        frameon=False,
        fontsize=9.5,
        handlelength=2.4,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)},
        columnspacing=1.45,
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

    base_plot._save_figure(fig, output)
    plt.close(fig)


def main() -> int:
    args = build_arg_parser().parse_args()
    base_plot.MODELS = (
        base_plot.MODELS[0],
        base_plot.ModelSpec("base_prompt_enhanced", "BT-Prompt", "#9b7ac9", "\\\\", "^"),
        *base_plot.MODELS[1:],
    )
    collected = base_plot._collect_values(args.outputs_root)
    _plot_combined_absolute_figure(collected, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
