"""Summarize analysis eval categories by augmentation depth/noise bucket."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.belieftrack_constants import (
    canonicalize_category_counts,
    percentages_from_counts,
)

CATEGORIES = [
    "insufficient_capability",
    "belief_failure",
    "oracle_match",
    "parse_error",
]

SKIP_NAMES = {
    "stats_report.json",
    "summary.json",
    "comparison.json",
    "augment_summary.json",
    "category_breakdown_summary.json",
    "failed_stay_depth_curve_summary.json",
    "failed_update_depth_curve_summary.json",
}


def _iter_eval_case_files(eval_dir: Path) -> Iterable[Path]:
    for path in sorted(eval_dir.rglob("*.json")):
        if path.name in SKIP_NAMES:
            continue
        yield path


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_augmentation(payload: Dict[str, Any], cases_dir: Path | None) -> Dict[str, Any]:
    augmentation = payload.get("augmentation") or {}
    if augmentation or cases_dir is None:
        return augmentation
    case_id = str(payload.get("case_id", ""))
    if not case_id:
        return {}
    case_path = cases_dir / f"{case_id}.json"
    if not case_path.exists():
        return {}
    case_payload = _read_json(case_path) or {}
    return case_payload.get("augmentation") or {}


def _bucket_key(augmentation: Dict[str, Any]) -> str:
    pipeline = str(augmentation.get("pipeline", "unknown"))
    if pipeline in {"rd_failed_stay_depth", "cd_failed_stay_depth"}:
        return f"failed_stay_depth/n_redundant={augmentation.get('n_redundant')}"
    if pipeline in {"rd_failed_update_depth", "cd_failed_update_depth"}:
        return f"failed_update_depth/delay_turns={augmentation.get('delay_turns')}"
    if pipeline == "noise_typology":
        return f"noise_typology/noise_type={augmentation.get('noise_type')}"
    return pipeline


def _aggregate_repeat_categories(per_run_categories: List[str]) -> str:
    if any(category == "belief_failure" for category in per_run_categories):
        return "belief_failure"
    if any(category == "insufficient_capability" for category in per_run_categories):
        return "insufficient_capability"
    return "oracle_match"


def _classify_projected_failed_stay_repeat(turns: List[Dict[str, Any]]) -> str:
    matches = [bool(turn.get("model_matches_golden")) for turn in turns]
    if len(matches) < 2:
        return "insufficient_capability"
    if not matches[0]:
        return "insufficient_capability"
    if not matches[1]:
        first_turn_gold = set(turns[0].get("golden_hypotheses") or [])
        second_turn_model = set(turns[1].get("model_hypotheses") or [])
        reintroduced = any(rule_id not in first_turn_gold for rule_id in second_turn_model)
        return "belief_failure" if reintroduced else "insufficient_capability"
    return "belief_failure" if any(not match for match in matches[2:]) else "oracle_match"


def _project_failed_stay_turns(turns: List[Dict[str, Any]], augmentation: Dict[str, Any], depth: int) -> List[Dict[str, Any]]:
    lock_idx = int(augmentation.get("lock_idx", -1))
    max_depth = int(augmentation.get("n_redundant", depth))
    if lock_idx < 0 or depth >= max_depth:
        return turns
    prefix_end = lock_idx + 1
    added_end = prefix_end + depth
    suffix_start = prefix_end + max_depth
    return list(turns[:added_end]) + list(turns[suffix_start:])


def _failed_stay_category_for_depth(payload: Dict[str, Any], augmentation: Dict[str, Any], depth: int) -> str:
    per_run_categories: List[str] = []
    for item in payload.get("repeat_trajectories") or []:
        trajectory = item.get("trajectory") or {}
        turns = list(trajectory.get("turns") or [])
        projected = _project_failed_stay_turns(turns, augmentation, depth)
        per_run_categories.append(_classify_projected_failed_stay_repeat(projected))
    if not per_run_categories:
        return "insufficient_capability"
    return _aggregate_repeat_categories(per_run_categories)


def _failed_stay_depths(augmentation: Dict[str, Any]) -> List[int]:
    depths = augmentation.get("eval_depths")
    if isinstance(depths, list) and depths:
        return sorted({int(depth) for depth in depths})
    return [int(augmentation.get("n_redundant", 0))]


def _empty_counts() -> Dict[str, int]:
    return {category: 0 for category in CATEGORIES}


def _format_bucket(counts: Dict[str, int]) -> Dict[str, Any]:
    total = sum(counts.values())
    cbm_counts = canonicalize_category_counts(counts)
    return {
        "num_cases": total,
        "cbm_category_counts": cbm_counts,
        "cbm_category_percentages": percentages_from_counts(cbm_counts),
    }


def summarize(eval_dir: Path, cases_dir: Path | None = None) -> Dict[str, Any]:
    bucket_counts: Dict[str, Dict[str, int]] = defaultdict(_empty_counts)
    overall_counts = _empty_counts()
    unknown_category_count = 0

    for eval_path in _iter_eval_case_files(eval_dir):
        payload = _read_json(eval_path)
        if payload is None or "repeat_trajectories" not in payload:
            continue
        category = str(payload.get("category", ""))
        if category not in overall_counts:
            unknown_category_count += 1
            continue

        augmentation = _load_augmentation(payload, cases_dir)
        pipeline = str(augmentation.get("pipeline", ""))
        if pipeline in {"rd_failed_stay_depth", "cd_failed_stay_depth"}:
            for depth in _failed_stay_depths(augmentation):
                bucket = f"failed_stay_depth/n_redundant={depth}"
                derived_category = _failed_stay_category_for_depth(payload, augmentation, depth)
                bucket_counts[bucket][derived_category] += 1
        else:
            bucket = _bucket_key(augmentation)
            bucket_counts[bucket][category] += 1

    for counts in bucket_counts.values():
        for category, count in counts.items():
            overall_counts[category] += count

    return {
        "eval_dir": str(eval_dir),
        "augmented_cases_dir": str(cases_dir) if cases_dir is not None else None,
        "overall": _format_bucket(overall_counts),
        "buckets": {
            bucket: _format_bucket(counts)
            for bucket, counts in sorted(bucket_counts.items())
        },
        "unknown_category_count": unknown_category_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize analysis eval categories by augmentation bucket")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--augmented-cases-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary = summarize(
        Path(args.eval_dir),
        Path(args.augmented_cases_dir) if args.augmented_cases_dir else None,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
