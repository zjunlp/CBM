#!/usr/bin/env python3
"""Plot one belief-probe case at a time."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "task_a" / "9B" / "probing" / "base" / "probe_ranking_simple_cases.json"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "probing" / "cases"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_case_figure(case: Dict[str, Any], output: Path) -> None:
    turns = list(case.get("turn_probes") or [])
    if not turns:
        return

    def shorten(text: Any, limit: int = 84) -> str:
        value = str(text or "").replace("<think>", "").replace("</think>", "").replace("\n", " ").strip()
        return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."

    def wrap_with_prefix(prefix: str, text: str, width: int, indent: str = "   ") -> List[str]:
        if not text:
            return [prefix.rstrip()]
        available = max(width - len(prefix), 10)
        wrapped = textwrap.wrap(text, width=available) or [""]
        lines = [prefix + wrapped[0]]
        lines.extend(indent + line for line in wrapped[1:])
        return lines

    xs = [int(turn.get("turn_index", idx) or idx) for idx, turn in enumerate(turns)]
    ranks = [float(turn.get("oracle_rank", 0) or 0) for turn in turns]
    original_wrong = [not bool(turn.get("original_correct")) for turn in turns]
    max_rank = max([max(ranks), 1.0]) + 1.0

    traj_lines: List[str] = []
    def load_model_hypotheses() -> Dict[int, List[str]]:
        traj = case.get("original_trajectory") or {}
        mapping: Dict[int, List[str]] = {}
        turns_src = traj.get("turns") if isinstance(traj.get("turns"), list) else []
        if turns_src:
            for turn in turns_src:
                if not isinstance(turn, dict):
                    continue
                idx = int(turn.get("turn_index", 0) or 0)
                mapping[idx] = [str(item) for item in (turn.get("model_hypotheses") or []) if str(item)]
            return mapping

        turn_idx = 0
        for message in traj.get("messages") or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            if role == "assistant":
                mapping[turn_idx] = [str(item) for item in (message.get("hypotheses") or []) if str(item)]
                turn_idx += 1
            elif role == "user":
                if turn_idx not in mapping:
                    mapping[turn_idx] = []
        return mapping

    model_hypotheses = load_model_hypotheses()
    for turn in turns:
        idx = int(turn.get("turn_index", 0) or 0)
        hypotheses = model_hypotheses.get(idx, [])
        if hypotheses:
            hyp_text = ", ".join(hypotheses[:4])
            traj_lines.extend(wrap_with_prefix(f"T{idx} Model's Belief State: ", hyp_text, 70))
        else:
            traj_lines.append(f"T{idx} Model's Belief State: (missing)")

        ranking = turn.get("completed_ranking") or turn.get("parsed_ranking") or []
        ranking = [str(item) for item in ranking if str(item)]
        if not ranking:
            traj_lines.append("   R: (missing)")
            traj_lines.append("")
            continue

        top = [f"{rank + 1}:{item}" for rank, item in enumerate(ranking[:3])]
        suffix = " ..." if len(ranking) > 3 else ""
        traj_lines.extend(wrap_with_prefix("   Rank: ", " ".join(top) + suffix, 70))
        traj_lines.append("")

    if traj_lines and traj_lines[-1] == "":
        traj_lines.pop()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, (ax, ax_text) = plt.subplots(
        1,
        2,
        figsize=(5.6, 2.2),
        gridspec_kw={"width_ratios": [1.25, 0.75]},
    )
    ax.plot(xs, ranks, color="#2563eb", marker="o", linewidth=1.8, markersize=4.0)
    for x, rank, wrong in zip(xs, ranks, original_wrong):
        ax.scatter([x], [rank], s=34, color="#dc2626" if wrong else "#16a34a", zorder=3)
        ax.text(x, rank + 0.18, f"{int(rank)}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Turn")
    ax.set_ylabel("Oracle rank")
    ax.set_xticks(xs)
    ax.set_ylim(0.5, max_rank)
    ax.grid(axis="y", color="#d1d5db", linestyle="--", linewidth=0.7, alpha=0.8)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markersize=6, label="Original correct"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", markersize=6, label="Original wrong"),
        plt.Line2D([0], [0], color="#2563eb", marker="o", label="Probe oracle rank"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(traj_lines),
        ha="left",
        va="top",
        family="DejaVu Sans Mono",
        fontsize=7.0,
        linespacing=1.0,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.16, wspace=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def select_cases(cases: Sequence[Dict[str, Any]], final_wrong_only: bool, case_id: str | None, repeat_index: int | None) -> List[Dict[str, Any]]:
    selected = list(cases)
    if final_wrong_only:
        selected = [case for case in selected if bool(case.get("final_wrong"))]
    if case_id is not None:
        selected = [case for case in selected if str(case.get("case_id")) == case_id]
    if repeat_index is not None:
        selected = [case for case in selected if int(case.get("repeat_index", -1)) == repeat_index]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot belief-probe cases")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--repeat-index", type=int, default=None)
    parser.add_argument("--all-cases", action="store_true")
    args = parser.parse_args()

    cases = read_json(args.input)
    if not isinstance(cases, list):
        raise ValueError(f"expected case list: {args.input}")

    selected = select_cases(cases, final_wrong_only=not args.all_cases, case_id=args.case_id, repeat_index=args.repeat_index)
    if not selected:
        print("[plot-probe-cases] no case selected")
        return 0

    for case in selected:
        suffix = "final_wrong" if case.get("final_wrong") else "all"
        name = f"{case.get('case_id')}_rep{case.get('repeat_index')}_{suffix}"
        output = args.output_dir / f"{name}.png"
        write_case_figure(case, output)
        print(f"[plot-probe-cases] wrote {output}")
        print(f"[plot-probe-cases] wrote {output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
