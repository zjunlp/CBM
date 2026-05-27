"""Convert heldout belief-stat case directories into evaluate.py-compatible cases JSON.

Input:
- failed_stay_dir: directory containing heldout failed_stay case json files
- failed_update_dir: directory containing heldout failed_update case json files

Output:
- a single JSON list in the same schema as training/data/cases/*.json
  expected by task_a.training.evaluate / cases_dataset.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _iter_case_files(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    files: list[Path] = []
    for file in sorted(path.rglob("*.json")):
        if file.name in {"all_results.json", "summary.json", "eval_report.json", "cases.json"}:
            continue
        files.append(file)
    return files


def _infer_mode_from_payload(payload: dict[str, Any], fallback: str) -> str:
    challenge_type = str(payload.get("challenge_sequence", {}).get("challenge_type", "")).lower()
    if "failed_stay" in challenge_type:
        return "failed_stay"
    if "failed_update" in challenge_type:
        return "failed_update"
    return fallback


def _extract_case_from_payload(payload: dict[str, Any], mode: str, index_within_rule: int) -> dict[str, Any]:
    repeat_trajectories = payload.get("repeat_trajectories") or []
    if not repeat_trajectories:
        raise ValueError(f"Missing repeat_trajectories in {payload.get('experiment_id')}")

    trajectory = repeat_trajectories[0].get("trajectory") or {}
    conversation = trajectory.get("conversation") or []
    if len(conversation) < 6:
        raise ValueError(f"Conversation too short in {payload.get('experiment_id')}: {len(conversation)}")

    system_prompt = conversation[0]["content"]
    user_turns = [msg["content"] for msg in conversation if msg.get("role") == "user"]
    if len(user_turns) != 3:
        raise ValueError(
            f"Expected 3 user turns in {payload.get('experiment_id')}, got {len(user_turns)}"
        )

    ground_truth = payload.get("challenge_sequence", {}).get("ground_truth") or []
    if len(ground_truth) != 3:
        raise ValueError(
            f"Expected 3 ground-truth turns in {payload.get('experiment_id')}, got {len(ground_truth)}"
        )

    turns = []
    for turn_idx in range(3):
        golden = ground_truth[turn_idx].get("survivors")
        if not isinstance(golden, list):
            raise ValueError(
                f"Invalid golden survivors at turn {turn_idx} in {payload.get('experiment_id')}"
            )
        turns.append(
            {
                "turn": turn_idx,
                "prompt": user_turns[turn_idx],
                "golden": golden,
            }
        )

    return {
        "case_id": payload["experiment_id"],
        "oracle": payload["rule_name"],
        "challenge_type": mode,
        "system_prompt": system_prompt,
        "turns": turns,
        "split": "heldout_eval",
        "index_within_rule": index_within_rule,
        "source_experiment_id": payload["experiment_id"],
        "source_category": payload.get("category"),
    }


def convert_heldout_case_dirs_to_eval_cases(
    failed_stay_dir: str,
    failed_update_dir: str,
    output_json: str,
) -> list[dict[str, Any]]:
    grouped_counts: dict[tuple[str, str], int] = defaultdict(int)
    cases: list[dict[str, Any]] = []

    for fallback_mode, input_path in (("failed_stay", failed_stay_dir), ("failed_update", failed_update_dir)):
        for file in _iter_case_files(input_path):
            payload = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            if "repeat_trajectories" not in payload or "challenge_sequence" not in payload:
                continue

            mode = _infer_mode_from_payload(payload, fallback_mode)
            oracle = str(payload.get("rule_name", ""))
            grouped_counts[(mode, oracle)] += 1
            case = _extract_case_from_payload(payload, mode, grouped_counts[(mode, oracle)] - 1)
            cases.append(case)

    if not cases:
        raise ValueError("No heldout case files were converted.")

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evaluate.py-compatible cases JSON from heldout case dirs")
    parser.add_argument("--failed_stay-dir", type=str, required=True)
    parser.add_argument("--failed_update-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = convert_heldout_case_dirs_to_eval_cases(
        failed_stay_dir=args.failed_stay_dir,
        failed_update_dir=args.failed_update_dir,
        output_json=args.output_json,
    )
    print(f"Wrote {len(cases)} cases to {args.output_json}")


if __name__ == "__main__":
    main()
