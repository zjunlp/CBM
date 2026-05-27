#!/usr/bin/env python3
"""Summarize held-out activation-steering intervention results."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steering.common import read_json, write_json  # noqa: E402


def key_for(item: Dict[str, Any]) -> Tuple[str, Any, float]:
    return (
        str(item.get("condition")),
        item.get("layer"),
        float(item.get("alpha") or 0.0),
    )


def pct(num: int, den: int) -> float:
    return round(num / den * 100.0, 2) if den else 0.0


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in rows:
        if item.get("condition") == "no_steer":
            baseline_items[str(item["record_id"])].append(item)
    baseline_wrong = {
        rid for rid, items in baseline_items.items() if not all(bool(item.get("correct")) for item in items)
    }
    baseline_correct = {
        rid for rid, items in baseline_items.items() if items and all(bool(item.get("correct")) for item in items)
    }

    grouped: Dict[Tuple[str, Any, float], List[Dict[str, Any]]] = defaultdict(list)
    for item in rows:
        grouped[key_for(item)].append(item)

    condition_summaries: List[Dict[str, Any]] = []
    for (condition, layer, alpha), items in sorted(grouped.items(), key=lambda x: (str(x[0][1]), x[0][0], x[0][2])):
        record_ids = {str(item["record_id"]) for item in items}
        rescued_records = {
            str(item["record_id"])
            for item in items
            if str(item["record_id"]) in baseline_wrong and bool(item.get("correct"))
        }
        preserved_records = {
            str(item["record_id"])
            for item in items
            if str(item["record_id"]) in baseline_correct and bool(item.get("correct"))
        }
        correct = [item for item in items if bool(item.get("correct"))]
        parse_ok = [item for item in items if bool(item.get("parse_ok"))]
        condition_summaries.append(
            {
                "condition": condition,
                "layer": layer,
                "alpha": alpha,
                "records": len(items),
                "unique_records": len(record_ids),
                "correct": len(correct),
                "correct_rate": pct(len(correct), len(items)),
                "parse_ok": len(parse_ok),
                "parse_ok_rate": pct(len(parse_ok), len(items)),
                "baseline_wrong_records": len(baseline_wrong & record_ids),
                "rescued": len(rescued_records),
                "rescue_rate": pct(len(rescued_records), len(baseline_wrong & record_ids)),
                "baseline_correct_records": len(baseline_correct & record_ids),
                "preserved": len(preserved_records),
                "preservation_rate": pct(len(preserved_records), len(baseline_correct & record_ids)),
            }
        )

    strict_condition_summaries = _summarize_strict(rows)

    best = sorted(
        [item for item in condition_summaries if item["condition"] != "no_steer"],
        key=lambda item: (item["rescue_rate"], item["correct_rate"], item["parse_ok_rate"]),
        reverse=True,
    )[:10]
    strict_best = sorted(
        [item for item in strict_condition_summaries if item["condition"] != "no_steer"],
        key=lambda item: (item["strict_rescue_rate"], -item["strict_error_rate"], item["strict_parse_ok_rate"]),
        reverse=True,
    )[:10]

    return {
        "overall": {
            "result_rows": len(rows),
            "baseline_records": len(baseline_items),
            "baseline_wrong": len(baseline_wrong),
            "baseline_correct": len(baseline_correct),
            "baseline_success_rate": pct(len(baseline_correct), len(baseline_items)),
        },
        "conditions": condition_summaries,
        "best_conditions": best,
        "strict_conditions": strict_condition_summaries,
        "strict_best_conditions": strict_best,
    }


def _condition_record_key(item: Dict[str, Any]) -> Tuple[str, Any, float, str]:
    condition, layer, alpha = key_for(item)
    return condition, layer, alpha, str(item["record_id"])


def _summarize_strict(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, Any, float, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in rows:
        grouped[_condition_record_key(item)].append(item)

    record_status: Dict[Tuple[str, Any, float], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (condition, layer, alpha, record_id), items in grouped.items():
        repeats = {int(item.get("generation_repeat", 0)) for item in items}
        strict_parse_ok = bool(items) and all(bool(item.get("parse_ok")) for item in items)
        strict_correct = bool(items) and all(bool(item.get("correct")) for item in items)
        any_correct = any(bool(item.get("correct")) for item in items)
        record_status[(condition, layer, alpha)][record_id] = {
            "repeats": len(repeats),
            "strict_parse_ok": strict_parse_ok,
            "strict_correct": strict_correct,
            "any_correct": any_correct,
        }

    baseline_records = record_status.get(("no_steer", None, 0.0), {})
    baseline_wrong = {rid for rid, item in baseline_records.items() if not item["strict_correct"]}
    baseline_correct = {rid for rid, item in baseline_records.items() if item["strict_correct"]}

    summaries: List[Dict[str, Any]] = []
    for (condition, layer, alpha), items_by_record in sorted(
        record_status.items(), key=lambda x: (str(x[0][1]), x[0][0], x[0][2])
    ):
        record_ids = set(items_by_record)
        strict_correct = {rid for rid, item in items_by_record.items() if item["strict_correct"]}
        strict_parse_ok = {rid for rid, item in items_by_record.items() if item["strict_parse_ok"]}
        rescued = strict_correct & baseline_wrong
        preserved = strict_correct & baseline_correct
        summaries.append(
            {
                "condition": condition,
                "layer": layer,
                "alpha": alpha,
                "records": len(record_ids),
                "strict_correct": len(strict_correct),
                "strict_correct_rate": pct(len(strict_correct), len(record_ids)),
                "strict_error_rate": round(100.0 - pct(len(strict_correct), len(record_ids)), 2) if record_ids else 0.0,
                "strict_parse_ok": len(strict_parse_ok),
                "strict_parse_ok_rate": pct(len(strict_parse_ok), len(record_ids)),
                "baseline_strict_wrong_records": len(baseline_wrong & record_ids),
                "strict_rescued": len(rescued),
                "strict_rescue_rate": pct(len(rescued), len(baseline_wrong & record_ids)),
                "baseline_strict_correct_records": len(baseline_correct & record_ids),
                "strict_preserved": len(preserved),
                "strict_preservation_rate": pct(len(preserved), len(baseline_correct & record_ids)),
            }
        )
    return summaries


def summarize(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(results)
    summary = _summarize_rows(rows)
    by_challenge: Dict[str, Dict[str, Any]] = {}
    for challenge in sorted({str(item.get("challenge_type")) for item in rows}):
        subset = [item for item in rows if str(item.get("challenge_type")) == challenge]
        if subset:
            by_challenge[challenge] = _summarize_rows(subset)["overall"]
    summary["by_challenge_overall"] = by_challenge
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("steering/outputs/task_a_train_a520/intervention_results.json"))
    parser.add_argument("--output", type=Path, default=Path("steering/outputs/task_a_train_a520/intervention_summary.json"))
    parser.add_argument("--csv", type=Path, default=Path("steering/outputs/task_a_train_a520/intervention_summary.csv"))
    args = parser.parse_args()

    results = read_json(args.input)
    if not isinstance(results, list):
        raise ValueError(f"{args.input} is not a result list")
    summary = summarize(results)
    write_json(args.output, summary)
    write_csv(args.csv, summary["conditions"])
    strict_csv = args.csv.with_name(args.csv.stem + "_strict" + args.csv.suffix)
    write_csv(strict_csv, summary["strict_conditions"])
    print(f"[summary] wrote {args.output}")
    print(f"[summary] wrote {args.csv}")
    print(f"[summary] wrote {strict_csv}")
    print(f"[summary] overall={summary['overall']}")
    if summary["best_conditions"]:
        print("[summary] best:")
        for item in summary["best_conditions"][:5]:
            print(
                f"  {item['condition']} L{item['layer']} alpha={item['alpha']}: "
                f"rescue={item['rescue_rate']}% correct={item['correct_rate']}% "
                f"parse={item['parse_ok_rate']}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
