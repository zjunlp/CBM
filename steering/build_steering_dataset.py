#!/usr/bin/env python3
"""Screen repeat-level or case-level divergence cases for belief-vector steering."""

from __future__ import annotations

import argparse
import copy
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONTAMINATION_EVAL_ROOTS = [
    REPO_ROOT / "task_a/outputs/swift_train_a_with_thinking_rollout_8_test_a_ckpt_520",
    REPO_ROOT / "task_a/outputs/swift_train_a_9B_ckpt_520_test_a_failed_isolation_train",
]

from steering.common import (  # noqa: E402
    CATEGORY_BELIEF,
    CATEGORY_DIRS,
    CATEGORY_VALID,
    CHALLENGE_TYPES,
    DEFAULT_BASE_EVAL_ROOT,
    DEFAULT_EVAL_ROOT,
    DEFAULT_RL_EVAL_ROOT,
    assistant_hypothesis_only_messages,
    build_base_context_messages,
    build_canonical_messages,
    extract_hypothesis_block,
    final_turn,
    first_failure_turn,
    get_repeat,
    hypotheses_match,
    index_case_payloads,
    load_model_samples,
    read_json,
    short_record,
    turn_records,
    write_json,
)


def target_turn_for_divergence(
    base_repeat: Dict[str, Any],
    challenge_type: str,
    *,
    failed_stay_target_turn: int | None = None,
) -> int:
    """Choose the intervention/extraction turn within the challenge-specific window."""
    turns = turn_records(base_repeat)
    if not turns:
        return 0
    last_turn = len(turns) - 1
    first_failure = first_failure_turn(base_repeat)
    if challenge_type == "failed_stay":
        allowed = [max(0, last_turn - 1), last_turn]
        if failed_stay_target_turn in allowed:
            return int(failed_stay_target_turn)
        window_failure = first_failure_turn_in_window(base_repeat, set(allowed))
        return window_failure if window_failure is not None else allowed[0]
    if challenge_type == "failed_update":
        return last_turn
    return first_failure


def first_failure_turn_in_window(repeat: Dict[str, Any], allowed_turns: set[int]) -> int | None:
    for idx, turn in enumerate(turn_records(repeat)):
        if idx in allowed_turns and not bool(turn.get("model_matches_golden")):
            return idx
    return None


def choose_failed_stay_target_turn(candidates: List[Dict[str, Any]]) -> int | None:
    counts: Counter[int] = Counter()
    fallback: int | None = None
    for item in candidates:
        if item.get("challenge_type") != "failed_stay":
            continue
        repeat = item["base_repeat"]
        turns = turn_records(repeat)
        if not turns:
            continue
        allowed = {max(0, len(turns) - 2), len(turns) - 1}
        fallback = min(allowed)
        failure_turn = first_failure_turn_in_window(repeat, allowed)
        if failure_turn is not None:
            counts[failure_turn] += 1
    if counts:
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return fallback


def build_record(
    *,
    split: str,
    case_id: str,
    challenge_type: str,
    repeat_index: int,
    base_payload: Dict[str, Any],
    rl_payload: Dict[str, Any],
    target_turn: int,
) -> Dict[str, Any]:
    base_repeat = get_repeat(base_payload, repeat_index)
    rl_repeat = get_repeat(rl_payload, repeat_index)
    base_turns = turn_records(base_repeat)
    rl_turns = turn_records(rl_repeat)
    if target_turn >= len(base_turns) or target_turn >= len(rl_turns):
        raise IndexError(f"{case_id}/rep{repeat_index}: target_turn={target_turn} out of range")

    golden = list(base_turns[target_turn].get("golden_hypotheses") or [])
    base_hyp = list(base_turns[target_turn].get("model_hypotheses") or [])
    rl_hyp = list(rl_turns[target_turn].get("model_hypotheses") or [])
    return {
        "record_id": f"{challenge_type}__{case_id}__rep{repeat_index}__t{target_turn}",
        "split": split,
        "case_id": case_id,
        "challenge_type": challenge_type,
        "repeat_index": int(repeat_index),
        "target_turn": int(target_turn),
        "oracle": base_payload.get("oracle"),
        "golden_hypotheses": golden,
        "base_original_hypotheses": base_hyp,
        "rl_original_hypotheses": rl_hyp,
        "base_original_correct": hypotheses_match(base_hyp, golden),
        "rl_original_correct": hypotheses_match(rl_hyp, golden),
        "messages_common": build_canonical_messages(base_repeat, target_turn),
        "messages_base_context": build_base_context_messages(base_repeat, target_turn),
        "messages_rl_context": build_base_context_messages(rl_repeat, target_turn),
        "base_repeat_trajectory": base_repeat.get("trajectory"),
        "rl_repeat_trajectory": rl_repeat.get("trajectory"),
        "source": {
            "eval_root": None,
            "base_source_file": base_payload.get("source_file"),
            "base_category": base_payload.get("category"),
            "rl_source_file": rl_payload.get("source_file"),
            "rl_category": rl_payload.get("category"),
        },
    }


def _candidate_repeat_indices(
    base_cats: List[str],
    rl_cats: List[str],
    *,
    kind: str,
) -> List[int]:
    indices: List[int] = []
    for repeat_index, (base_cat, rl_cat) in enumerate(zip(base_cats, rl_cats, strict=False)):
        if kind == "divergence" and base_cat == CATEGORY_VALID and rl_cat == CATEGORY_BELIEF:
            indices.append(repeat_index)
        elif kind == "control" and base_cat == CATEGORY_BELIEF and rl_cat == CATEGORY_BELIEF:
            indices.append(repeat_index)
    return indices


def _choose_repeat_index(
    repeat_indices: List[int],
    *,
    priority: str = "min_repeat",
) -> int:
    if not repeat_indices:
        raise ValueError("empty repeat_indices")
    if priority == "min_repeat":
        return min(repeat_indices)
    if priority == "max_repeat":
        return max(repeat_indices)
    return repeat_indices[0]


def screen_repeat_pairs(
    base_eval_root: Path,
    rl_eval_root: Path,
    seed: int,
    challenge_types: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    base_samples = load_model_samples(base_eval_root, "base")
    rl_samples = load_model_samples(rl_eval_root, "lora")
    base_payloads = index_case_payloads(base_eval_root, "base")
    rl_payloads = index_case_payloads(rl_eval_root, "lora")

    divergence: List[Dict[str, Any]] = []
    control: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    divergence_candidates: List[Dict[str, Any]] = []
    control_candidates: List[Dict[str, Any]] = []

    allowed_challenge_types = set(challenge_types or CHALLENGE_TYPES)
    for case_id in sorted(set(base_samples) & set(rl_samples)):
        base_sample = base_samples[case_id]
        rl_sample = rl_samples[case_id]
        challenge_type = str(base_sample.get("challenge_type") or "")
        if challenge_type not in allowed_challenge_types:
            continue
        if case_id not in base_payloads or case_id not in rl_payloads:
            skipped.append({"case_id": case_id, "reason": "missing_payload"})
            continue

        base_cats = list(base_sample.get("repeat_categories") or [])
        rl_cats = list(rl_sample.get("repeat_categories") or [])
        for repeat_index, (base_cat, rl_cat) in enumerate(zip(base_cats, rl_cats, strict=False)):
            base_payload = base_payloads[case_id]
            rl_payload = rl_payloads[case_id]
            if base_cat == CATEGORY_VALID and rl_cat == CATEGORY_BELIEF:
                divergence_candidates.append(
                    {
                        "case_id": case_id,
                        "challenge_type": challenge_type,
                        "repeat_index": repeat_index,
                        "base_payload": base_payload,
                        "rl_payload": rl_payload,
                        "base_repeat": get_repeat(base_payload, repeat_index),
                    }
                )
            elif base_cat == CATEGORY_BELIEF and rl_cat == CATEGORY_BELIEF:
                control_candidates.append(
                    {
                        "case_id": case_id,
                        "challenge_type": challenge_type,
                        "repeat_index": repeat_index,
                        "base_payload": base_payload,
                        "rl_payload": rl_payload,
                        "base_repeat": get_repeat(base_payload, repeat_index),
                    }
                )

    for item in divergence_candidates:
        target_turn = target_turn_for_divergence(
            item["base_repeat"],
            item["challenge_type"],
        )
        divergence.append(
            build_record(
                split="divergence",
                case_id=item["case_id"],
                challenge_type=item["challenge_type"],
                repeat_index=item["repeat_index"],
                base_payload=item["base_payload"],
                rl_payload=item["rl_payload"],
                target_turn=target_turn,
            )
        )
    for item in control_candidates:
        target_turn = final_turn(item["base_repeat"])
        control.append(
            build_record(
                split="control",
                case_id=item["case_id"],
                challenge_type=item["challenge_type"],
                repeat_index=item["repeat_index"],
                base_payload=item["base_payload"],
                rl_payload=item["rl_payload"],
                target_turn=target_turn,
            )
        )

    rng = random.Random(seed)
    rng.shuffle(divergence)
    rng.shuffle(control)
    return {"divergence_pool": divergence, "control_pool": control, "skipped": skipped}


def screen_case_level(
    base_eval_root: Path,
    rl_eval_root: Path,
    seed: int,
    challenge_types: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    base_samples = load_model_samples(base_eval_root, "base")
    rl_samples = load_model_samples(rl_eval_root, "lora")
    base_payloads = index_case_payloads(base_eval_root, "base")
    rl_payloads = index_case_payloads(rl_eval_root, "lora")

    entries: Dict[str, Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []

    allowed_challenge_types = set(challenge_types or CHALLENGE_TYPES)
    for case_id in sorted(set(base_samples) & set(rl_samples)):
        base_sample = base_samples[case_id]
        rl_sample = rl_samples[case_id]
        challenge_type = str(base_sample.get("challenge_type") or "")
        if challenge_type not in allowed_challenge_types:
            continue
        if case_id not in base_payloads or case_id not in rl_payloads:
            skipped.append({"case_id": case_id, "reason": "missing_payload"})
            continue

        base_cats = list(base_sample.get("repeat_categories") or [])
        rl_cats = list(rl_sample.get("repeat_categories") or [])
        divergence_repeats = _candidate_repeat_indices(base_cats, rl_cats, kind="divergence")
        control_repeats = _candidate_repeat_indices(base_cats, rl_cats, kind="control")
        if not divergence_repeats and not control_repeats:
            continue
        entries[case_id] = {
            "case_id": case_id,
            "challenge_type": challenge_type,
            "divergence_repeats": divergence_repeats,
            "control_repeats": control_repeats,
            "base_payload": base_payloads[case_id],
            "rl_payload": rl_payloads[case_id],
        }

    return {"entries": list(entries.values()), "skipped": skipped}


def take(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return list(records)
    return list(records[:limit])


def parse_challenge_types(text: str) -> List[str]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    invalid = sorted(set(values) - set(CHALLENGE_TYPES))
    if invalid:
        raise ValueError(f"unknown challenge types: {invalid}; expected subset of {CHALLENGE_TYPES}")
    return values


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "source"


def load_failed_isolation_samples(eval_root: Path, model_label: str) -> Dict[str, Dict[str, Any]]:
    stats_path = eval_root / model_label / "failed_isolation" / "stats_report.json"
    stats = read_json(stats_path)
    return {str(item["case_id"]): item for item in stats.get("sample_results", [])}


def index_failed_isolation_payloads(eval_root: Path, model_label: str) -> Dict[str, Dict[str, Any]]:
    root = eval_root / model_label / "failed_isolation"
    index: Dict[str, Dict[str, Any]] = {}
    for category_dir in list(CATEGORY_DIRS.values()) + ["parse_error"]:
        case_dir = root / category_dir
        if not case_dir.exists():
            continue
        for path in sorted(case_dir.glob("*.json")):
            if path.name in {"stats_report.json", "summary.json", "comparison.json"}:
                continue
            payload = read_json(path)
            case_id = str(payload.get("case_id") or path.stem)
            index[case_id] = payload
    return index


def repeat_final_correct(payload: Dict[str, Any], repeat_index: int) -> bool:
    repeat = get_repeat(payload, repeat_index)
    turns = turn_records(repeat)
    if not turns:
        return False
    final = turns[-1]
    return hypotheses_match(final.get("model_hypotheses"), final.get("golden_hypotheses") or [])


def failed_isolation_turns(base_repeat: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for turn in turn_records(base_repeat):
        turns.append(
            {
                "turn": int(turn.get("turn", len(turns))),
                "prompt": str(turn.get("prompt") or ""),
                "golden_hypotheses": list(turn.get("golden_hypotheses") or []),
            }
        )
    return turns


def build_hypothesis_only_context_messages(repeat: Dict[str, Any], target_turn: int) -> List[Dict[str, str]]:
    return assistant_hypothesis_only_messages(build_base_context_messages(repeat, target_turn))


def has_required_hypothesis_history(repeat: Dict[str, Any], target_turn: int) -> bool:
    messages = build_base_context_messages(repeat, target_turn)
    for message in messages:
        if message.get("role") == "assistant" and not extract_hypothesis_block(str(message.get("content") or "")):
            return False
    return True


def build_failed_isolation_record(
    *,
    source_index: int,
    source_tag: str,
    eval_root: Path,
    case_id: str,
    repeat_index: int,
    base_payload: Dict[str, Any],
    rl_payload: Dict[str, Any],
) -> Dict[str, Any]:
    base_repeat = get_repeat(base_payload, repeat_index)
    rl_repeat = get_repeat(rl_payload, repeat_index)
    base_turns = turn_records(base_repeat)
    rl_turns = turn_records(rl_repeat)
    if not base_turns or len(base_turns) != len(rl_turns):
        raise ValueError(
            f"{case_id}/rep{repeat_index}: mismatched turn count base={len(base_turns)} rl={len(rl_turns)}"
        )

    target_turn = len(base_turns) - 1
    if not has_required_hypothesis_history(base_repeat, target_turn):
        raise ValueError("base_history_missing_hypothesis")
    if not has_required_hypothesis_history(rl_repeat, target_turn):
        raise ValueError("rl_history_missing_hypothesis")

    golden = list(base_turns[target_turn].get("golden_hypotheses") or [])
    base_hyp = list(base_turns[target_turn].get("model_hypotheses") or [])
    rl_hyp = list(rl_turns[target_turn].get("model_hypotheses") or [])

    return {
        "record_id": f"failed_isolation__{source_tag}__{case_id}__rep{repeat_index}",
        "split": "extract_divergence",
        "case_id": case_id,
        "source_index": int(source_index),
        "source_tag": source_tag,
        "challenge_type": "failed_isolation",
        "repeat_index": int(repeat_index),
        "target_turn": int(target_turn),
        "num_turns": len(base_turns),
        "oracle": base_payload.get("oracle"),
        "golden_hypotheses": golden,
        "base_original_hypotheses": base_hyp,
        "rl_original_hypotheses": rl_hyp,
        "base_original_correct": hypotheses_match(base_hyp, golden),
        "rl_original_correct": hypotheses_match(rl_hyp, golden),
        "system_prompt": str((base_repeat.get("trajectory") or {}).get("system_prompt") or base_payload.get("system_prompt") or ""),
        "turns": failed_isolation_turns(base_repeat),
        "messages_base_by_turn": [
            build_hypothesis_only_context_messages(base_repeat, turn_index) for turn_index in range(len(base_turns))
        ],
        "messages_rl_by_turn": [
            build_hypothesis_only_context_messages(rl_repeat, turn_index) for turn_index in range(len(rl_turns))
        ],
        "messages_base_context": build_hypothesis_only_context_messages(base_repeat, target_turn),
        "messages_rl_context": build_hypothesis_only_context_messages(rl_repeat, target_turn),
        "source": {
            "eval_root": str(eval_root),
            "base_source_file": base_payload.get("source_file"),
            "base_category": base_payload.get("category"),
            "rl_source_file": rl_payload.get("source_file"),
            "rl_category": rl_payload.get("category"),
        },
    }


def screen_failed_isolation_eval_root(eval_root: Path, source_index: int) -> Dict[str, Any]:
    source_tag = safe_name(eval_root.name)
    base_samples = load_failed_isolation_samples(eval_root, "base")
    rl_samples = load_failed_isolation_samples(eval_root, "lora")
    base_payloads = index_failed_isolation_payloads(eval_root, "base")
    rl_payloads = index_failed_isolation_payloads(eval_root, "lora")

    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    repeat_count_by_case: Counter[int] = Counter()
    turn_count_pairs: Counter[str] = Counter()
    eligible_repeat_total = 0

    for case_id in sorted(set(base_samples) & set(rl_samples)):
        if case_id not in base_payloads or case_id not in rl_payloads:
            skipped.append({"case_id": case_id, "reason": "missing_payload"})
            continue
        base_cats = list(base_samples[case_id].get("repeat_categories") or [])
        rl_cats = list(rl_samples[case_id].get("repeat_categories") or [])
        case_candidates: List[Dict[str, Any]] = []
        for repeat_index, (base_cat, rl_cat) in enumerate(zip(base_cats, rl_cats, strict=False)):
            if base_cat != CATEGORY_VALID or rl_cat != CATEGORY_BELIEF:
                continue
            if repeat_final_correct(base_payloads[case_id], repeat_index):
                skipped.append({"case_id": case_id, "repeat_index": repeat_index, "reason": "base_final_correct"})
                continue
            if not repeat_final_correct(rl_payloads[case_id], repeat_index):
                skipped.append({"case_id": case_id, "repeat_index": repeat_index, "reason": "rl_final_wrong"})
                continue
            try:
                record = build_failed_isolation_record(
                    source_index=source_index,
                    source_tag=source_tag,
                    eval_root=eval_root,
                    case_id=case_id,
                    repeat_index=repeat_index,
                    base_payload=base_payloads[case_id],
                    rl_payload=rl_payloads[case_id],
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"case_id": case_id, "repeat_index": repeat_index, "reason": str(exc)})
                continue
            case_candidates.append(record)
        if case_candidates:
            eligible_repeat_total += len(case_candidates)
            repeat_count_by_case[len(case_candidates)] += 1
            selected = sorted(case_candidates, key=lambda item: int(item["repeat_index"]))[0]
            turn_count_pairs[f"{selected['num_turns']}:{selected['num_turns']}"] += 1
            records.append(selected)

    return {
        "source_tag": source_tag,
        "eval_root": str(eval_root),
        "records": records,
        "skipped": skipped,
        "summary": {
            "overlap_cases": len(set(base_samples) & set(rl_samples)),
            "case_level_base_wrong_rl_right": sum(repeat_count_by_case.values()),
            "eligible_repeat_level_base_wrong_rl_right": eligible_repeat_total,
            "selected_case_level_records": len(records),
            "repeat_per_case_distribution": dict(sorted(repeat_count_by_case.items())),
            "turn_count_pairs": dict(sorted(turn_count_pairs.items())),
            "skipped": len(skipped),
        },
    }


def compact_failed_isolation_heldout_record(record: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "record_id",
        "split",
        "case_id",
        "source_index",
        "source_tag",
        "challenge_type",
        "repeat_index",
        "target_turn",
        "num_turns",
        "oracle",
        "golden_hypotheses",
        "base_original_hypotheses",
        "rl_original_hypotheses",
        "base_original_correct",
        "rl_original_correct",
        "system_prompt",
        "turns",
        "source",
    ]
    return {key: copy.deepcopy(record[key]) for key in keep_keys if key in record}


def build_failed_isolation_dataset(
    eval_roots: Sequence[Path],
    seed: int,
    max_cases: int,
) -> Dict[str, Any]:
    source_results = [screen_failed_isolation_eval_root(root, idx) for idx, root in enumerate(eval_roots)]
    all_records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    per_source_summary: Dict[str, Any] = {}
    for item in source_results:
        all_records.extend(item["records"])
        skipped.extend({"source_tag": item["source_tag"], **skip} for skip in item["skipped"])
        per_source_summary[item["source_tag"]] = item["summary"]

    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    if max_cases > 0:
        for item in source_results:
            source_records = list(item["records"])
            rng.shuffle(source_records)
            remaining = max_cases - len(records)
            if remaining <= 0:
                break
            records.extend(source_records[:remaining])
    else:
        for item in source_results:
            source_records = list(item["records"])
            rng.shuffle(source_records)
            records.extend(source_records)
    heldout = [compact_failed_isolation_heldout_record(record) for record in records]
    for record in heldout:
        record["split"] = "heldout"

    num_turns = Counter(int(record["num_turns"]) for record in records)
    selected_by_source = Counter(str(record.get("source_tag") or "unknown") for record in records)
    payload = {
        "config": {
            "eval_roots": [str(path) for path in eval_roots],
            "seed": seed,
            "max_cases": max_cases,
            "challenge_types": ["failed_isolation"],
            "unit": "case-level failed_isolation record",
            "selection": "one minimum-repeat trajectory per case where Base final is wrong, RL final is right, and all assistant history has a parseable hypothesis block",
            "source_selection": "fill from eval_roots in argument order; shuffle within each source by seed",
            "assistant_context": "hypothesis_block_only",
            "extraction_prompt": "stored online Base/RL trajectories for every turn, with assistant history truncated to <hypothesis>...</hypothesis>",
            "intervention_prompt": "fresh online Base generation for every turn, with generated assistant history truncated to <hypothesis>...</hypothesis>",
            "vector_formula": "mean_over_records(mean_over_turns(H_RL_turn - H_Base_turn))",
            "heldout": "same records as extraction",
        },
        "summary": {
            "extract_divergence": len(records),
            "heldout": len(heldout),
            "control": 0,
            "eligible_case_level_base_wrong_rl_right": len(all_records),
            "eligible_repeat_level_base_wrong_rl_right": sum(
                int(item["summary"]["eligible_repeat_level_base_wrong_rl_right"]) for item in source_results
            ),
            "selected_case_level_base_wrong_rl_right": len(records),
            "selected_by_source": dict(sorted(selected_by_source.items())),
            "case_level_base_wrong_rl_right_before_cap": sum(
                int(item["summary"]["case_level_base_wrong_rl_right"]) for item in source_results
            ),
            "num_turns_distribution": dict(sorted(num_turns.items())),
            "skipped": len(skipped),
            "per_source": per_source_summary,
        },
        "records": {
            "extract_divergence": records,
            "control": [],
            "heldout": heldout,
        },
        "screening_preview": {
            "extract_divergence": [short_record(record) for record in records[:5]],
            "heldout": [short_record(record) for record in heldout[:5]],
        },
        "skipped": skipped[:200],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["task_a", "failed_isolation"], default="task_a")
    parser.add_argument("--eval-root", type=Path, default=None, help="Override both eval roots.")
    parser.add_argument("--base-eval-root", type=Path, default=DEFAULT_BASE_EVAL_ROOT)
    parser.add_argument("--rl-eval-root", type=Path, default=DEFAULT_RL_EVAL_ROOT)
    parser.add_argument("--eval-roots", type=Path, nargs="+", default=DEFAULT_CONTAMINATION_EVAL_ROOTS)
    parser.add_argument("--output", type=Path, default=Path("steering/outputs/task_a_train_a520/steering_cases.json"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--extract-n", type=int, default=60, help="Divergence records used to extract the vector.")
    parser.add_argument("--heldout-n", type=int, default=30, help="Remaining divergence records used for intervention.")
    parser.add_argument("--control-n", type=int, default=60, help="Base-correct/RL-correct nuisance controls.")
    parser.add_argument("--challenge-types", default=",".join(CHALLENGE_TYPES), help="Comma-separated challenge types to include.")
    parser.add_argument("--split-level", choices=["repeat", "case"], default="repeat")
    parser.add_argument("--max-cases", type=int, default=100, help="Maximum case-level records to keep; <=0 keeps all.")
    parser.add_argument(
        "--reuse-divergence-for-heldout",
        action="store_true",
        help="Use the same divergence records for vector extraction and held-out intervention.",
    )
    parser.add_argument("--no-control", action="store_true", help="Do not include control records.")
    args = parser.parse_args()

    if args.mode == "failed_isolation":
        payload = build_failed_isolation_dataset(args.eval_roots, args.seed, args.max_cases)
        write_json(args.output, payload)
        print(f"[failed_isolation:screen] wrote {args.output}")
        print(f"[failed_isolation:screen] summary={payload['summary']}")
        return 0

    if args.eval_root is not None:
        base_eval_root = args.eval_root
        rl_eval_root = args.eval_root
    else:
        base_eval_root = args.base_eval_root
        rl_eval_root = args.rl_eval_root
    challenge_types = parse_challenge_types(args.challenge_types)

    if args.split_level == "repeat":
        screened = screen_repeat_pairs(base_eval_root, rl_eval_root, args.seed, challenge_types)
        divergence_pool = screened["divergence_pool"]
        control_pool = screened["control_pool"]
        extract_div = take(divergence_pool, args.extract_n)
        if args.reuse_divergence_for_heldout:
            heldout = [copy.deepcopy(record) for record in extract_div]
        else:
            heldout_start = len(extract_div)
            heldout = take(divergence_pool[heldout_start:], args.heldout_n)
        control = [] if args.no_control else take(control_pool, args.control_n)

        for record in extract_div:
            record["split"] = "extract_divergence"
        for record in heldout:
            record["split"] = "heldout"
        for record in control:
            record["split"] = "control"

        payload = {
            "config": {
                "base_eval_root": str(base_eval_root),
                "rl_eval_root": str(rl_eval_root),
                "seed": args.seed,
                "extract_n": args.extract_n,
                "heldout_n": args.heldout_n,
                "control_n": args.control_n,
                "split_level": args.split_level,
                "challenge_types": challenge_types,
                "reuse_divergence_for_heldout": bool(args.reuse_divergence_for_heldout),
                "no_control": bool(args.no_control),
                "unit": "repeat-level record",
                "extraction_prompt": "canonical gold-history prefix",
                "rl_extraction_prompt": "rl original history prefix",
                "intervention_prompt": "base original history up to target user turn",
            },
            "summary": {
                "divergence_pool": len(divergence_pool),
                "control_pool": len(control_pool),
                "extract_divergence": len(extract_div),
                "heldout": len(heldout),
                "control": len(control),
                "skipped": len(screened["skipped"]),
            },
            "records": {
                "extract_divergence": extract_div,
                "control": control,
                "heldout": heldout,
            },
            "screening_preview": {
                "extract_divergence": [short_record(r) for r in extract_div[:5]],
                "control": [short_record(r) for r in control[:5]],
                "heldout": [short_record(r) for r in heldout[:5]],
            },
        }
    else:
        screened = screen_case_level(base_eval_root, rl_eval_root, args.seed, challenge_types)
        entries = {item["case_id"]: item for item in screened["entries"]}
        rng = random.Random(args.seed)
        divergence_case_ids = [case_id for case_id, item in entries.items() if item["divergence_repeats"]]
        control_case_ids = [case_id for case_id, item in entries.items() if item["control_repeats"]]
        rng.shuffle(divergence_case_ids)
        rng.shuffle(control_case_ids)
        extract_case_ids = divergence_case_ids[: min(args.extract_n, len(divergence_case_ids))]
        if args.reuse_divergence_for_heldout:
            heldout_case_ids = list(extract_case_ids)
        else:
            heldout_case_ids = divergence_case_ids[len(extract_case_ids) : len(extract_case_ids) + args.heldout_n]
        exclude = set(extract_case_ids) | set(heldout_case_ids)
        control_case_ids = [] if args.no_control else [cid for cid in control_case_ids if cid not in exclude][: args.control_n]

        def build_selected(case_ids: List[str], kind: str) -> List[Dict[str, Any]]:
            selected: List[Dict[str, Any]] = []
            for case_id in case_ids:
                entry = entries[case_id]
                repeat_indices = entry["divergence_repeats"] if kind != "control" else entry["control_repeats"]
                if not repeat_indices:
                    continue
                if kind == "control":
                    repeat_index = _choose_repeat_index(repeat_indices, priority="min_repeat")
                    target_turn = final_turn(get_repeat(entry["base_payload"], repeat_index))
                else:
                    repeat_index = _choose_repeat_index(repeat_indices, priority="min_repeat")
                    target_turn = target_turn_for_divergence(
                        get_repeat(entry["base_payload"], repeat_index),
                        entry["challenge_type"],
                    )
                record = build_record(
                    split=kind,
                    case_id=case_id,
                    challenge_type=entry["challenge_type"],
                    repeat_index=repeat_index,
                    base_payload=entry["base_payload"],
                    rl_payload=entry["rl_payload"],
                    target_turn=target_turn,
                )
                record["split"] = kind
                selected.append(record)
            return selected

        extract_div = build_selected(extract_case_ids, "extract_divergence")
        heldout = build_selected(heldout_case_ids, "heldout")
        control = build_selected(control_case_ids, "control")

        payload = {
            "config": {
                "base_eval_root": str(base_eval_root),
                "rl_eval_root": str(rl_eval_root),
                "seed": args.seed,
                "extract_n": args.extract_n,
                "heldout_n": args.heldout_n,
                "control_n": args.control_n,
                "split_level": args.split_level,
                "challenge_types": challenge_types,
                "reuse_divergence_for_heldout": bool(args.reuse_divergence_for_heldout),
                "no_control": bool(args.no_control),
                "unit": "case-level representative record",
                "extraction_prompt": "canonical gold-history prefix",
                "rl_extraction_prompt": "rl original history prefix",
                "intervention_prompt": "base original history up to target user turn",
            },
            "summary": {
                "divergence_pool": len(divergence_case_ids),
                "control_pool": len(control_case_ids),
                "extract_divergence": len(extract_div),
                "heldout": len(heldout),
                "control": len(control),
                "skipped": len(screened["skipped"]),
            },
            "records": {
                "extract_divergence": extract_div,
                "control": control,
                "heldout": heldout,
            },
            "screening_preview": {
                "extract_divergence": [short_record(r) for r in extract_div[:5]],
                "control": [short_record(r) for r in control[:5]],
                "heldout": [short_record(r) for r in heldout[:5]],
            },
        }
    write_json(args.output, payload)
    print(f"[screen] wrote {args.output}")
    print(f"[screen] summary={payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
