#!/usr/bin/env python3
"""Plot probing failure modes and steering transfer as a two-panel figure."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_ROOT.parent
_MPLCONFIGDIR = ANALYSIS_ROOT / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": False,
        "hatch.linewidth": 0.55,
    }
)


DEFAULT_STEERING_CSV = REPO_ROOT / "steering" / "outputs" / "task_a_b_strict_summary_compact.csv"
DEFAULT_OUTPUT = ANALYSIS_ROOT / "figures" / "probe_steering_combined_ab.png"
FIG_DPI = 600

VANILLA = "#6d8fc6"
STEERED = "#68b7ae"
RL_CD_ORANGE = "#d9a06b"
BLUE = VANILLA
CORRECT = STEERED
WRONG = RL_CD_ORANGE
EDGE = "#333333"
GRID = "#d4d4d4"

DATA_TYPES = ("failed_stay", "failed_update", "failed_isolation")
RATE_LABELS = {
    "failed_stay": "FSR",
    "failed_update": "FUR",
    "failed_isolation": "FIR",
}


@dataclass(frozen=True)
class ProbePanel:
    title: str
    xs: Sequence[int]
    ranks: Sequence[float]
    correct: Sequence[bool]
    note: str
    note_xy: tuple[float, float]
    arrow_xy: tuple[float, float]


PROBE_PANELS: Sequence[ProbePanel] = (
    ProbePanel(
        title="Evidence-State FailedStay",
        xs=(0, 1, 2, 3),
        ranks=(1, 1, 6, 6),
        correct=(True, True, False, False),
        note="Mis-tracks\nevidence",
        note_xy=(1.20, 8.7),
        arrow_xy=(2.0, 6.0),
    ),
    ProbePanel(
        title="Backtracking Failure",
        xs=(0, 1, 2, 3),
        ranks=(3, 1, 1, 5),
        correct=(True, True, True, False),
        note="Fails to recover\ndiscarded rule",
        note_xy=(0.56, 8.2),
        arrow_xy=(3.0, 5.0),
    ),
    ProbePanel(
        title="Context Hijacking",
        xs=(0, 1, 2, 3, 4),
        ranks=(1, 7, 7, 7, 7),
        correct=(True, False, False, False, False),
        note="Misleading comment\noverrides tracking",
        note_xy=(1.18, 5.5),
        arrow_xy=(1.0, 7.0),
    ),
    ProbePanel(
        title="Latent-Output Gap",
        xs=(0, 1, 2, 3),
        ranks=(1, 1, 1, 1),
        correct=(True, False, True, False),
        note="Correct latent belief;\nwrong final answer",
        note_xy=(0.58, 6.7),
        arrow_xy=(3.0, 1.0),
    ),
)


def _format_value(value: float, *, signed: bool = False) -> str:
    if abs(value - round(value)) < 1e-6:
        text = f"{int(round(value))}"
    else:
        text = f"{value:.1f}"
    if signed and value > 0:
        return f"+{text}"
    return text


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_cell(row: Dict[str, str], key: str) -> float:
    return float(str(row[key]).replace("+", "").strip())


def _best_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {
        row["data_type"]: row
        for row in rows
        if row.get("scenario") == "task_a" and row.get("best_combo_in_task_a") == "BEST"
    }


def _transfer_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {row["data_type"]: row for row in rows if row.get("scenario") == "task_b"}


def _plot_probe_panel(ax: plt.Axes, panel: ProbePanel, *, show_ylabel: bool) -> None:
    xs = np.asarray(panel.xs, dtype=float)
    ranks = np.asarray(panel.ranks, dtype=float)
    colors = [CORRECT if ok else WRONG for ok in panel.correct]

    ax.plot(xs, ranks, color=BLUE, linewidth=2.2, zorder=2)
    ax.scatter(xs, ranks, s=45, c=colors, edgecolor="white", linewidth=0.9, zorder=3)
    ax.axhline(5, color="#cfcfcf", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_xlim(-0.18, 4.18)
    ax.set_ylim(10.6, 0.4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([1, 4, 8, 10])
    ax.grid(axis="y", color="#e2e2e2", linestyle="--", linewidth=0.75, zorder=0)
    ax.tick_params(axis="both", labelsize=8.1, width=0.9, length=3.2)
    ax.set_xlabel("Turn", fontsize=9.0, labelpad=4)
    ax.set_ylabel("Oracle Rank" if show_ylabel else "", fontsize=9.8, labelpad=7)
    ax.set_title(panel.title, fontsize=9.2, pad=5)

    ax.annotate(
        panel.note,
        xy=panel.arrow_xy,
        xytext=panel.note_xy,
        fontsize=6.8,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.17", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.88},
        arrowprops={"arrowstyle": "-|>", "color": "#444444", "linewidth": 1.1, "mutation_scale": 10},
        zorder=4,
    )

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.05)
        ax.spines[side].set_color("#222222")


def _plot_steering_panel(ax: plt.Axes, data_type: str, task_a: Dict[str, str], task_b: Dict[str, str]) -> None:
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
        color=VANILLA,
        edgecolor=EDGE,
        linewidth=0.7,
        zorder=3,
    )
    steered_bars = ax.bar(
        x + width / 2,
        steered,
        width,
        color=STEERED,
        edgecolor=EDGE,
        linewidth=0.7,
        hatch="//",
        zorder=3,
    )
    for bars, values in ((base_bars, baseline), (steered_bars, steered)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.3,
                _format_value(value),
                ha="center",
                va="bottom",
                fontsize=7.1,
                color="#222222",
            )

    for xi, base, steer, change_pct in zip(x, baseline, steered, failure_changes):
        arrow_y = min(max(base, steer) + 15.0, 116.0)
        change_color = "#226b5e" if change_pct <= 0 else "#9b341f"
        ax.annotate(
            "",
            xy=(xi + width / 2 - 0.03, arrow_y),
            xytext=(xi - width / 2 + 0.03, arrow_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": change_color,
                "linewidth": 1.3,
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
            fontsize=7.6,
            color=change_color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
        )

    ax.set_ylim(0, 122)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", color=GRID, linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=8.2, width=0.8, length=3.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(RATE_LABELS[data_type], fontsize=12, labelpad=8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color(EDGE)


def plot(steering_csv: Path, output_path: Path) -> None:
    rows = _read_rows(steering_csv)
    best = _best_rows(rows)
    transfer = _transfer_rows(rows)
    missing = [key for key in DATA_TYPES if key not in best or key not in transfer]
    if missing:
        raise ValueError(f"missing steering rows for: {', '.join(missing)}")

    fig = plt.figure(figsize=(10.8, 4.0), dpi=FIG_DPI)
    outer = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[0.82, 1.0],
        hspace=0.42,
        left=0.07,
        right=0.992,
        bottom=0.075,
        top=0.915,
    )
    probe_grid = outer[0].subgridspec(1, 4, wspace=0.25)
    steering_grid = outer[1].subgridspec(1, 3, wspace=0.22)

    probe_axes = [fig.add_subplot(probe_grid[0, i]) for i in range(4)]
    for idx, panel in enumerate(PROBE_PANELS):
        _plot_probe_panel(probe_axes[idx], panel, show_ylabel=idx == 0)

    steering_axes = [fig.add_subplot(steering_grid[0, i]) for i in range(3)]
    for idx, data_type in enumerate(DATA_TYPES):
        _plot_steering_panel(steering_axes[idx], data_type, best[data_type], transfer[data_type])

    probe_handles = [
        Line2D([0], [0], color=BLUE, marker="o", markersize=5.0, linewidth=2.0, label="Oracle rank"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=CORRECT, markeredgecolor="white", markersize=5.5, label="Correct"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=WRONG, markeredgecolor="white", markersize=5.5, label="Wrong"),
    ]
    steering_handles = [
        Patch(facecolor=VANILLA, edgecolor=EDGE, label="Vanilla"),
        Patch(facecolor=STEERED, edgecolor=EDGE, hatch="//", label="Steered"),
    ]
    fig.legend(
        handles=probe_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=3,
        frameon=False,
        fontsize=9.2,
        handlelength=1.6,
        columnspacing=1.6,
    )
    fig.legend(
        handles=steering_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.485),
        ncol=2,
        frameon=False,
        fontsize=9.4,
        handlelength=1.6,
        columnspacing=1.8,
    )
    fig.text(0.012, 0.982, "(a)", fontsize=13, ha="left", va="center")
    fig.text(0.012, 0.485, "(b)", fontsize=13, ha="left", va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", dpi=FIG_DPI)
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot probing and steering as a combined a/b figure.")
    parser.add_argument("--steering-csv", type=Path, default=DEFAULT_STEERING_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.steering_csv, args.output)
    print(f"[plot] wrote {args.output}")
    print(f"[plot] wrote {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
