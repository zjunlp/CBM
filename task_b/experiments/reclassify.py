"""Reclassify existing Scenario B experiment outputs using the current rules."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List

from task_b.experiments.challenge_metrics import (
    CATEGORIES,
    build_category_dirs,
    build_stats_report,
    classify_trajectory,
)
from task_b.experiments.belief_stats import aggregate_repeat_categories
from utils.io import save_json


def _iter_result_files(category_dirs: Dict[str, str]) -> Iterable[str]:
    for category_dir in category_dirs.values():
        if not os.path.isdir(category_dir):
            continue
        for filename in os.listdir(category_dir):
            if filename.endswith(".json"):
                yield os.path.join(category_dir, filename)


def _load_report_metadata(output_dir: str) -> Dict[str, Any]:
    report_path = os.path.join(output_dir, "stats_report.json")
    if not os.path.exists(report_path):
        return {}
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reclassify_output_dir(output_dir: str) -> None:
    category_dirs = build_category_dirs(output_dir)
    for category_dir in category_dirs.values():
        os.makedirs(category_dir, exist_ok=True)

    all_files = sorted(_iter_result_files(category_dirs))
    moved = Counter()
    stayed = Counter()
    per_task_counts: Dict[str, Dict[str, int]] = {}
    all_results: List[Dict[str, Any]] = []

    for file_path in all_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        challenge_dict = data["challenge"]
        task_label = f"{data['circuit_type']}:{data['oracle']}"

        new_per_run_categories: List[str] = []
        new_repeat_trajectories: List[Dict[str, Any]] = []
        for repeat in data["repeat_trajectories"]:
            trajectory = repeat["trajectory"]
            repeat_category = classify_trajectory(trajectory, challenge_dict)
            new_per_run_categories.append(repeat_category)
            new_repeat_trajectories.append({**repeat, "category": repeat_category})

        new_category = aggregate_repeat_categories(new_per_run_categories)
        data["category"] = new_category
        data["per_run_categories"] = new_per_run_categories
        data["repeat_trajectories"] = new_repeat_trajectories

        destination_dir = category_dirs[new_category]
        destination_path = os.path.join(destination_dir, os.path.basename(file_path))
        if os.path.dirname(file_path) != destination_dir:
            shutil.move(file_path, destination_path)
            moved[new_category] += 1
        else:
            stayed[new_category] += 1

        save_json(destination_path, data)

        task_counts = per_task_counts.setdefault(
            task_label,
            {category: 0 for category in CATEGORIES},
        )
        task_counts[new_category] += 1

        all_results.append({
            "experiment_id": data["experiment_id"],
            "circuit_type": data["circuit_type"],
            "oracle": data["oracle"],
            "template_idx": data["template_idx"],
            "run_idx": data["run_idx"],
            "category": new_category,
            "per_run_categories": new_per_run_categories,
        })

    previous_report = _load_report_metadata(output_dir)
    report = build_stats_report(
        model=previous_report.get("model"),
        prompt_style=previous_report.get("prompt_style"),
        num_runs_per_task=previous_report.get("num_runs_per_task"),
        run_semantics=previous_report.get("run_semantics"),
        repeats=previous_report.get("repeats"),
        gpus=previous_report.get("gpus"),
        circuit_types=previous_report.get("circuit_types"),
        faults=previous_report.get("faults"),
        per_task_counts=per_task_counts,
    )
    save_json(os.path.join(output_dir, "stats_report.json"), report)
    save_json(os.path.join(output_dir, "all_results.json"), all_results)

    total_counts = report["total"]
    grand_total = total_counts["total"]
    print(f"\n=== {os.path.basename(output_dir)} ===")
    print(f"  Total files processed: {len(all_files)}")
    print(f"  Files moved: {sum(moved.values())}  (stayed: {sum(stayed.values())})")
    print("\n  New distribution:")
    for category in CATEGORIES:
        count = total_counts[category]
        pct = count / max(grand_total, 1) * 100
        print(f"    {category}: {count} ({pct:.1f}%)")

    print("\n  Per-task breakdown:")
    for task_label, counts in sorted(per_task_counts.items()):
        task_total = sum(counts.values())
        if task_total == 0:
            continue
        parts = "  ".join(
            f"{category[:16]}"
            f"={counts[category]}({counts[category] / task_total * 100:.0f}%)"
            for category in CATEGORIES
            if counts[category] > 0
        )
        print(f"    {task_label:<28}  {parts}")


def main() -> None:
    dirs = sys.argv[1:] or [
        "task_b/outputs/v3_binary_test_split",
        "task_b/outputs/v3_binary_pure_retraction_test_split",
    ]
    for output_dir in dirs:
        reclassify_output_dir(output_dir)


if __name__ == "__main__":
    main()
