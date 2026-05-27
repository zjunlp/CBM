"""Utilities for running Scenario B GRPO with the verl framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_b.runtime.orchestrator import parse_hypotheses  # noqa: E402


def _jaccard_from_response(response: str, gt_values: Iterable[str]) -> dict[str, Any]:
    gt_set = set(gt_values)
    hypotheses = parse_hypotheses(response)
    if hypotheses is None:
        return {
            "score": 0.0,
            "jaccard": 0.0,
            "exact_match": 0.0,
            "prediction_count": 0.0,
            "parse_failed": 1.0,
        }

    pred_list = list(hypotheses)
    pred_set = set(pred_list)
    union = pred_set | gt_set
    jaccard = 1.0 if not union else len(pred_set & gt_set) / len(union)
    exact_match = 1.0 if pred_set == gt_set else 0.0
    return {
        "score": float(jaccard),
        "jaccard": float(jaccard),
        "exact_match": float(exact_match),
        "prediction_count": float(len(pred_list)),
        "parse_failed": 0.0,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: list[str] | str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """verl custom reward function entrypoint."""
    del data_source
    del extra_info
    gt_values = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    return _jaccard_from_response(solution_str, gt_values)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for msg in messages:
        role = str(msg["role"])
        content = msg["content"]
        if not isinstance(content, str):
            raise TypeError(f"Expected string message content, got {type(content)!r}")
        normalized.append({"role": role, "content": content})
    return normalized


def convert_record(row: dict[str, Any]) -> dict[str, Any]:
    gt_survivors = row.get("gt_survivors")
    if not isinstance(gt_survivors, list):
        raise TypeError(f"gt_survivors must be a list, got {type(gt_survivors)!r}")
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages)!r}")

    return {
        "messages": _normalize_messages(messages),
        "data_source": "task_b_belief",
        "reward_model": {
            "ground_truth": gt_survivors,
        },
        "extra_info": {
            "case_id": row.get("case_id"),
            "challenge_type": row.get("challenge_type"),
            "oracle": row.get("oracle"),
            "target_set": row.get("target_set"),
            "selected_turn": row.get("selected_turn"),
            "source_category": row.get("source_category"),
            "source_file": row.get("source_file"),
            "source_repeat_index": row.get("source_repeat_index"),
        },
    }


def prepare_dataset(input_path: Path, output_path: Path) -> None:
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected top-level list in {input_path}, got {type(rows)!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            converted = convert_record(row)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"Converted {len(rows)} rows to verl dataset: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Scenario B belief dataset for verl GRPO.")
    parser.add_argument("--input", required=True, help="Path to the original Scenario B GRPO JSON dataset.")
    parser.add_argument("--output", required=True, help="Path to the verl JSONL output dataset.")
    args = parser.parse_args()

    prepare_dataset(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
