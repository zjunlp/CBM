"""Aggregate per-turn eval outputs into FSR/FUR depth curves."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _iter_eval_case_files(eval_dir: Path) -> Iterable[Path]:
    skip_names = {
        "stats_report.json",
        "augment_summary.json",
        "summary.json",
        "comparison.json",
        "failed_stay_depth_curve_summary.json",
    }
    for path in sorted(eval_dir.rglob("*.json")):
        if path.name in skip_names:
            continue
        yield path


def _load_eval_payload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "repeat_trajectories" in payload:
        return payload
    return None


def _majority_category(categories: List[str]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for cat in categories:
        counts[cat] += 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _turn_matches_from_payload(payload: Dict[str, Any]) -> List[List[bool]]:
    trajectories = payload.get("repeat_trajectories") or []
    all_runs: List[List[bool]] = []
    for item in trajectories:
        session = item.get("trajectory") or {}
        turns = session.get("turns") or []
        all_runs.append([bool(turn.get("model_matches_golden")) for turn in turns])
    return all_runs


def analyze_failed_stay_depth(payload: Dict[str, Any], augmentation: Dict[str, Any], effective_depth: Optional[int] = None) -> Dict[str, Any]:
    lock_idx = int(augmentation.get("lock_idx", -1))
    n_redundant = int(effective_depth if effective_depth is not None else augmentation.get("n_redundant", 0))
    if lock_idx < 0:
        return {"status": "missing_lock_idx"}

    runs = _turn_matches_from_payload(payload)
    if not runs:
        return {"status": "no_runs"}

    first_fail_depths: List[Optional[int]] = []
    for matches in runs:
        depth: Optional[int] = None
        for offset in range(1, n_redundant + 1):
            turn_idx = lock_idx + offset
            if turn_idx >= len(matches):
                break
            if not matches[turn_idx]:
                depth = offset
                break
        first_fail_depths.append(depth)

    failed = [d for d in first_fail_depths if d is not None]
    survived_all = sum(1 for d in first_fail_depths if d is None)
    return {
        "status": "ok",
        "lock_idx": lock_idx,
        "n_redundant": n_redundant,
        "generated_n_redundant": int(augmentation.get("n_redundant", n_redundant)),
        "num_runs": len(runs),
        "fsr_any_redundant": round(len(failed) / max(len(runs), 1) * 100, 2),
        "survived_all_redundant": survived_all,
        "first_fail_depth_histogram": {
            str(d): sum(1 for x in first_fail_depths if x == d) for d in range(1, n_redundant + 1)
        },
    }


def analyze_failed_update_depth(payload: Dict[str, Any], augmentation: Dict[str, Any]) -> Dict[str, Any]:
    corr_idx = int(augmentation.get("corr_idx", -1))
    delay_turns = int(augmentation.get("delay_turns", 0))
    if corr_idx < 0:
        return {"status": "missing_corr_idx"}

    correction_turn = corr_idx + delay_turns
    runs = _turn_matches_from_payload(payload)
    if not runs:
        return {"status": "no_runs"}

    prefix_ok = []
    correction_ok = []
    for matches in runs:
        prefix = matches[:corr_idx] if corr_idx <= len(matches) else matches
        prefix_ok.append(all(prefix) if prefix else False)
        correction_ok.append(
            bool(matches[correction_turn]) if correction_turn < len(matches) else False
        )

    eligible = [i for i, ok in enumerate(prefix_ok) if ok]
    if not eligible:
        return {"status": "no_eligible_prefix"}

    fur = sum(1 for i in eligible if not correction_ok[i]) / len(eligible)
    return {
        "status": "ok",
        "corr_idx": corr_idx,
        "delay_turns": delay_turns,
        "correction_turn": correction_turn,
        "num_runs": len(runs),
        "eligible_runs": len(eligible),
        "fur_at_correction": round(fur * 100, 2),
    }


def analyze_eval_dir(eval_dir: Path, augmented_cases_dir: Optional[Path] = None) -> Dict[str, Any]:
    per_case: List[Dict[str, Any]] = []
    failed_stay_buckets: Dict[str, List[float]] = defaultdict(list)
    failed_update_buckets: Dict[str, List[float]] = defaultdict(list)

    for eval_path in _iter_eval_case_files(eval_dir):
        payload = _load_eval_payload(eval_path)
        if payload is None:
            continue
        case_id = str(payload.get("case_id", eval_path.stem))
        augmentation = payload.get("augmentation") or {}
        if not augmentation and augmented_cases_dir is not None:
            aug_path = augmented_cases_dir / f"{case_id}.json"
            if aug_path.exists():
                aug_case = json.loads(aug_path.read_text(encoding="utf-8"))
                augmentation = aug_case.get("augmentation") or {}

        pipeline = str(augmentation.get("pipeline", ""))
        item: Dict[str, Any] = {"case_id": case_id, "pipeline": pipeline, "eval_path": str(eval_path)}
        if pipeline in {"rd_failed_stay_depth", "cd_failed_stay_depth"}:
            depths = augmentation.get("eval_depths")
            if not isinstance(depths, list) or not depths:
                depths = [augmentation.get("n_redundant")]
            item["failed_stay_depth"] = {}
            for depth in sorted({int(depth) for depth in depths}):
                stats = analyze_failed_stay_depth(payload, augmentation, effective_depth=depth)
                item["failed_stay_depth"][str(depth)] = stats
                if stats.get("status") == "ok":
                    key = f"n={depth}"
                    failed_stay_buckets[key].append(float(stats["fsr_any_redundant"]))
        elif pipeline in {"rd_failed_update_depth", "cd_failed_update_depth"}:
            stats = analyze_failed_update_depth(payload, augmentation)
            item["failed_update_depth"] = stats
            if stats.get("status") == "ok":
                key = f"delay={augmentation.get('delay_turns')}"
                failed_update_buckets[key].append(float(stats["fur_at_correction"]))
        per_case.append(item)

    summary = {
        "eval_dir": str(eval_dir),
        "num_cases": len(per_case),
        "failed_stay_depth_means": {
            bucket: round(sum(vals) / len(vals), 2) for bucket, vals in failed_stay_buckets.items() if vals
        },
        "failed_update_depth_means": {
            bucket: round(sum(vals) / len(vals), 2) for bucket, vals in failed_update_buckets.items() if vals
        },
        "cases": per_case,
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze per-turn FSR/FUR depth curves")
    parser.add_argument("--eval-dir", required=True, help="Directory with eval per-case JSON outputs")
    parser.add_argument("--augmented-cases-dir", default=None, help="Original augmented cases for metadata")
    parser.add_argument("--output", default=None, help="Write summary JSON to this path")
    args = parser.parse_args(argv)

    summary = analyze_eval_dir(Path(args.eval_dir), Path(args.augmented_cases_dir) if args.augmented_cases_dir else None)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
