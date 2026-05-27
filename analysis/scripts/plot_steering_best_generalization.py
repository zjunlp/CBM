#!/usr/bin/env python3
"""Plot best steering failure rates and cross-scenario generalization."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_ROOT.parent
_MPLCONFIGDIR = ANALYSIS_ROOT / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": False,
        "hatch.linewidth": 0.55,
    }
)


DEFAULT_INPUT = REPO_ROOT / "steering" / "outputs" / "task_a_b_strict_summary_compact.csv"
DEFAULT_OUTPUT = ANALYSIS_ROOT / "figures" / "steering_best_generalization_failure_rate.png"

DATA_TYPES = ("failed_stay", "failed_update", "failed_isolation")
TASK_LABELS = {
    "failed_stay": "FSR",
    "failed_update": "FUR",
    "failed_isolation": "FIR",
}

BASE_COLOR = "#6d8fc6"
STEER_COLOR = "#68b7ae"
EDGE_COLOR = "#333333"
GRID_COLOR = "#d4d4d4"


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_cell(row: Dict[str, str], key: str) -> float:
    return float(str(row[key]).replace("+", "").strip())


def _format_value(value: float, *, signed: bool = False) -> str:
    if abs(value - round(value)) < 1e-6:
        text = f"{int(round(value))}"
    else:
        text = f"{value:.1f}"
    if signed and value > 0:
        return f"+{text}"
    return text


def _best_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    best: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row.get("scenario") == "task_a" and row.get("best_combo_in_task_a") == "BEST":
            best[row["data_type"]] = row
    return best


def _transfer_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    transfer: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row.get("scenario") == "task_b":
            transfer[row["data_type"]] = row
    return transfer


def _style_axis(ax: plt.Axes, *, ylabel: str) -> None:
    ax.set_ylim(0, 122)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=8.5, width=0.8, length=3.5)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="semibold", labelpad=8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)


def _plot_task_panel(ax: plt.Axes, data_type: str, task_a: Dict[str, str], task_b: Dict[str, str]) -> None:
    rows = [task_a, task_b]
    labels = ["Scenario A", "Scenario B"]
    baseline = [100.0 - _float_cell(row, "baseline_strict_accuracy") for row in rows]
    steered = [100.0 - _float_cell(row, "intervention_strict_accuracy") for row in rows]
    failure_changes = [
        ((steer - base) / base * 100.0) if base else 0.0
        for base, steer in zip(baseline, steered)
    ]

    x = np.arange(len(rows))
    width = 0.28
    base_bars = ax.bar(
        x - width / 2,
        baseline,
        width,
        color=BASE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.75,
        zorder=3,
    )
    steer_bars = ax.bar(
        x + width / 2,
        steered,
        width,
        color=STEER_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.75,
        hatch="//",
        zorder=3,
    )

    for bars, values in ((base_bars, baseline), (steer_bars, steered)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.3,
                _format_value(value),
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#222222",
            )

    for xi, base, steer, change_pct in zip(x, baseline, steered, failure_changes):
        delta_color = "#226b5e" if change_pct <= 0 else "#9b341f"
        arrow_y = min(max(base, steer) + 15.0, 116.0)
        ax.annotate(
            "",
            xy=(xi + width / 2 - 0.03, arrow_y),
            xytext=(xi - width / 2 + 0.03, arrow_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": delta_color,
                "linewidth": 1.35,
                "mutation_scale": 10,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=5,
        )
        ax.text(
            xi,
            arrow_y + 5.0,
            f"{_format_value(change_pct, signed=True)}%",
            ha="center",
            va="center",
            fontsize=7.7,
            color=delta_color,
            fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.margins(x=0.10)


def plot(input_path: Path, output_path: Path) -> None:
    rows = _read_rows(input_path)
    best = _best_rows(rows)
    transfer = _transfer_rows(rows)
    missing = [key for key in DATA_TYPES if key not in best or key not in transfer]
    if missing:
        raise ValueError(f"missing rows for: {', '.join(missing)}")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8.8, 2.35), dpi=300, sharey=True)
    fig.patch.set_facecolor("white")

    for idx, data_type in enumerate(DATA_TYPES):
        _plot_task_panel(axes[idx], data_type, best[data_type], transfer[data_type])
        _style_axis(axes[idx], ylabel=TASK_LABELS[data_type])

    handles = [
        Patch(facecolor=BASE_COLOR, edgecolor=EDGE_COLOR, label="Vanilla"),
        Patch(facecolor=STEER_COLOR, edgecolor=EDGE_COLOR, hatch="//", label="Steered"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=1.7,
        columnspacing=1.8,
    )
    fig.text(
        0.992,
        1.01,
        "Lower is better ↓",
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.80, wspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot best steering failure rates and transfer results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.input, args.output)
    print(f"[plot] wrote {args.output}")
    print(f"[plot] wrote {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
