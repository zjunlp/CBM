"""Shared trajectory metrics and reporting utilities for Scenario B experiments."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


CATEGORIES = ["insufficient_capability", "oracle_match", "belief_failure", "unstable", "illegal"]

CATEGORY_DIR_NAMES = {
    "insufficient_capability": "insufficient_capability",
    "oracle_match": "oracle_match",
    "belief_failure": "belief_failure",
    "unstable": "unstable",
    "illegal": "illegal",
}

STRICT_TURN_COUNT = 3


def build_category_dirs(output_dir: str) -> Dict[str, str]:
    return {
        category: os.path.join(output_dir, dirname)
        for category, dirname in CATEGORY_DIR_NAMES.items()
    }


def _normalize_turn_hypotheses(turn: Mapping[str, Any]) -> Set[str]:
    return set(turn.get("hypotheses") or [])


def _extract_turn_hypothesis_sets(
    trajectory: Mapping[str, Any],
    *,
    required_turns: int = STRICT_TURN_COUNT,
    capability_turn_count: int = 2,
) -> Optional[List[Optional[Set[str]]]]:
    turns = trajectory.get("turns", [])
    if len(turns) < required_turns:
        return None
    # Parse errors only invalidate the capability/convergence prefix. Later
    # parse errors are treated as wrong challenge answers, which can be valid
    # failed_stay/failed_update signals.
    capability_turns = turns[: min(capability_turn_count, required_turns)]
    if any(turn.get("parse_error") for turn in capability_turns):
        return None
    sets: List[Optional[Set[str]]] = []
    for turn in turns[:required_turns]:
        if turn.get("parse_error") or turn.get("hypotheses") is None:
            sets.append(None)
        else:
            sets.append(_normalize_turn_hypotheses(turn))
    return sets


def _extract_ground_truth_sets(
    challenge_dict: Mapping[str, Any],
    *,
    required_turns: int = STRICT_TURN_COUNT,
) -> Optional[List[Set[str]]]:
    ground_truth = challenge_dict.get("ground_truth", [])
    if len(ground_truth) < required_turns:
        return None
    return [set(step.get("survivors", [])) for step in ground_truth[:required_turns]]


def _classify_noise_trajectory(
    trajectory: Mapping[str, Any],
    challenge_dict: Mapping[str, Any],
) -> str:
    if trajectory.get("termination_reason") == "format_error_dropped":
        return "illegal"
    turns = list(trajectory.get("turns", []))
    if not turns:
        return "illegal"

    finalize_turn = next(
        (turn for turn in turns if turn.get("action") == "finalize_fault"),
        None,
    )
    if finalize_turn is None:
        # The agent kept asking measurements but did not converge, so this is a valid failure case.
        if any(turn.get("action") == "ask_measure" for turn in turns):
            return "belief_failure"
        return "illegal"

    final_guess = str(finalize_turn.get("final_guess", "")).upper()
    oracle = str(challenge_dict.get("oracle", "")).upper()
    if final_guess == oracle:
        return "oracle_match"
    return "belief_failure"


def _classify_failed_isolation_trajectory(
    trajectory: Mapping[str, Any],
    challenge_dict: Mapping[str, Any],
) -> str:
    if trajectory.get("termination_reason") == "format_error_dropped":
        return "illegal"

    turns = list(trajectory.get("turns", []))
    if not turns:
        return "illegal"

    if any(bool(turn.get("parse_error")) for turn in turns):
        return "illegal"

    strict_prefix_turns = int(challenge_dict.get("strict_failed_isolation_prefix_turns") or 0)
    if strict_prefix_turns > 0:
        ground_truth = challenge_dict.get("ground_truth", [])
        if len(turns) <= strict_prefix_turns or len(ground_truth) <= strict_prefix_turns:
            return "insufficient_capability"

        for idx in range(strict_prefix_turns):
            expected = set((ground_truth[idx] or {}).get("survivors") or [])
            actual = _normalize_turn_hypotheses(turns[idx])
            if actual != expected:
                return "insufficient_capability"

        post_turns = turns[strict_prefix_turns:]
        post_truth = ground_truth[strict_prefix_turns:]
        if not post_turns or not post_truth:
            return "insufficient_capability"
        for turn, truth in zip(post_turns, post_truth):
            actual = _normalize_turn_hypotheses(turn)
            expected = set((truth or {}).get("survivors") or [])
            if actual != expected:
                return "belief_failure"
        return "oracle_match"

    final_hyp_list = turns[-1].get("hypotheses")
    if final_hyp_list is None:
        return "illegal"
    if len(final_hyp_list) == 0:
        return "belief_failure"

    final_hyp = set(final_hyp_list)
    oracle = str(challenge_dict.get("oracle", "")).upper()
    if final_hyp == {oracle}:
        return "oracle_match"
    return "belief_failure"


def classify_trajectory(
    trajectory: Dict[str, Any],
    challenge_dict: Dict[str, Any],
) -> str:
    """Classify a trajectory using strict turn-level exact matches."""
    if challenge_dict.get("mode") == "noise" or trajectory.get("mode") == "noise":
        return _classify_noise_trajectory(trajectory, challenge_dict)

    if challenge_dict.get("mode") == "failed_isolation" or trajectory.get("mode") == "failed_isolation":
        return _classify_failed_isolation_trajectory(trajectory, challenge_dict)

    ground_truth = challenge_dict.get("ground_truth", [])
    required_turns = len(ground_truth) if len(ground_truth) > STRICT_TURN_COUNT else STRICT_TURN_COUNT
    ground_truth_sets = _extract_ground_truth_sets(
        challenge_dict,
        required_turns=required_turns,
    )
    if ground_truth_sets is None:
        return "insufficient_capability"

    if challenge_dict.get("direct_converge"):
        turn_hypotheses = _extract_turn_hypothesis_sets(
            trajectory,
            required_turns=required_turns,
            capability_turn_count=1,
        )
        if turn_hypotheses is None:
            return "insufficient_capability"
        # Direct-converge failed_stay: T0 establishes singleton capability, and
        # T1/T2 are both interference turns. A wrong answer on either
        # interference turn is the failed_stay signal.
        if turn_hypotheses[0] != ground_truth_sets[0]:
            return "insufficient_capability"
        for hyps, gt in zip(turn_hypotheses[1:], ground_truth_sets[1:]):
            if hyps is None or hyps != gt:
                return "belief_failure"
        return "oracle_match"

    if len(ground_truth_sets) > STRICT_TURN_COUNT:
        # Multi-turn failed_stay: prefix turns establish capability/convergence.
        # Only errors at or after failed_stay_eval_start_turn count as failed_stay.
        failed_stay_eval_start_turn = int(challenge_dict.get("failed_stay_eval_start_turn", 2))
        if failed_stay_eval_start_turn < 1:
            failed_stay_eval_start_turn = 1
        if failed_stay_eval_start_turn >= len(ground_truth_sets):
            failed_stay_eval_start_turn = len(ground_truth_sets) - 1
        turn_hypotheses = _extract_turn_hypothesis_sets(
            trajectory,
            required_turns=required_turns,
            capability_turn_count=failed_stay_eval_start_turn,
        )
        if turn_hypotheses is None:
            return "insufficient_capability"

        for hyps, gt in zip(
            turn_hypotheses[:failed_stay_eval_start_turn],
            ground_truth_sets[:failed_stay_eval_start_turn],
        ):
            if hyps != gt:
                return "insufficient_capability"
        for hyps, gt in zip(
            turn_hypotheses[failed_stay_eval_start_turn:],
            ground_truth_sets[failed_stay_eval_start_turn:],
        ):
            if hyps is None or hyps != gt:
                return "belief_failure"
        return "oracle_match"

    turn_hypotheses = _extract_turn_hypothesis_sets(
        trajectory,
        required_turns=required_turns,
        capability_turn_count=2,
    )
    if turn_hypotheses is None:
        return "insufficient_capability"
    hyps_t0, hyps_t1, hyps_t2 = turn_hypotheses
    gt_t0, gt_t1, gt_t2 = ground_truth_sets

    if hyps_t0 != gt_t0 or hyps_t1 != gt_t1:
        return "insufficient_capability"
    if hyps_t2 is None:
        # T2 parse error / no hypothesis produced — model failed to commit to a
        # belief at correction time. This is a belief_failure failed_update signal.
        return "belief_failure"
    if hyps_t2 == gt_t2:
        return "oracle_match"
    return "belief_failure"


def sanitize_trajectory_for_export(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep compact turn data and preserve full conversation for audit/debug."""
    cleaned_turns = [
        {
            "turn": turn.get("turn"),
            "hypotheses": turn.get("hypotheses"),
            "gt_survivors": turn.get("gt_survivors"),
            "parse_error": turn.get("parse_error"),
            "finish_reason": turn.get("finish_reason"),
            "output_truncated": turn.get("output_truncated"),
            "action": turn.get("action"),
            "query_key": turn.get("query_key"),
            "query_valid": turn.get("query_valid"),
            "env_answer": turn.get("env_answer"),
            "host_comment": turn.get("host_comment"),
            "final_guess": turn.get("final_guess"),
            "finalize_success": turn.get("finalize_success"),
        }
        for turn in trajectory.get("turns", [])
    ]

    return {
        "mode": trajectory.get("mode"),
        "circuit_type": trajectory.get("circuit_type"),
        "oracle": trajectory.get("oracle"),
        "noise_success": trajectory.get("noise_success"),
        "termination_reason": trajectory.get("termination_reason"),
        "n_turns_played": trajectory.get("n_turns_played", len(cleaned_turns)),
        "turns": cleaned_turns,
        "conversation": trajectory.get("conversation", []),
    }


def compact_case_export(
    *,
    challenge_dict: Mapping[str, Any],
    repeat_trajectories: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Export only conversations and side-by-side golden/sampled survivors."""
    is_noise = challenge_dict.get("mode") == "noise"
    # For non-noise: golden survivors come from the challenge ground_truth (fixed).
    # For noise: golden survivors are per-trajectory (dynamic, based on evidence gathered).
    static_golden_survivors: Optional[List[Dict[str, Any]]] = None
    if not is_noise:
        static_golden_survivors = [
            {
                "turn": step.get("turn", index),
                "survivors": sorted(step.get("survivors", [])),
            }
            for index, step in enumerate(challenge_dict.get("ground_truth", []))
        ]
    conversation = []
    for item in repeat_trajectories:
        trajectory = item["trajectory"]
        if is_noise:
            # Per-trajectory dynamic GT: survivors consistent with evidence accumulated so far
            golden_survivors = [
                {
                    "turn": turn.get("turn", index),
                    "survivors": turn.get("gt_survivors") or [],
                }
                for index, turn in enumerate(trajectory.get("turns", []))
            ]
        else:
            golden_survivors = static_golden_survivors or []
        sampled_survivors = [
            {
                "turn": turn.get("turn", index),
                "survivors": turn.get("hypotheses"),
            }
            for index, turn in enumerate(trajectory.get("turns", []))
        ]
        n_turns = max(len(golden_survivors), len(sampled_survivors))
        turn_survivors = []
        for index in range(n_turns):
            golden = golden_survivors[index] if index < len(golden_survivors) else {}
            sampled = sampled_survivors[index] if index < len(sampled_survivors) else {}
            turn_survivors.append({
                "turn": golden.get("turn", sampled.get("turn", index)),
                "golden_survivors": golden.get("survivors"),
                "sampled_survivors": sampled.get("survivors"),
            })
        conversation.append({
            "conversation": trajectory.get("conversation", []),
            "turn_survivors": turn_survivors,
        })
    return {
        "conversation": conversation,
    }


def compute_belief_metrics(
    trajectory: Dict[str, Any],
    challenge_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute trajectory-level belief metrics."""
    mode = challenge_dict.get("mode") or trajectory.get("mode")
    turns = trajectory.get("turns", [])

    if mode == "noise":
        ask_turns = [turn for turn in turns if turn.get("action") == "ask_measure"]
        finalize_turn = next(
            (turn for turn in turns if turn.get("action") == "finalize_fault"),
            None,
        )
        success = bool(trajectory.get("noise_success"))
        if finalize_turn is not None and not success:
            success = str(finalize_turn.get("final_guess", "")).upper() == str(challenge_dict.get("oracle", "")).upper()

        parse_errors = sum(1 for turn in turns if turn.get("parse_error"))
        return {
            "noise_success": success,
            "asked_measurements": len(ask_turns),
            "finalized": finalize_turn is not None,
            "turns_to_finalize": (int(finalize_turn.get("turn", 0)) + 1) if finalize_turn is not None else None,
            "max_turn_exceeded": trajectory.get("termination_reason") == "max_turns_exceeded",
            "comment_misled": (not success) and any(int(turn.get("turn", -1)) >= 2 for turn in ask_turns),
            "format_error_rate": round(parse_errors / len(turns), 4) if turns else None,
        }

    oracle = challenge_dict["oracle"]
    misleading_target = challenge_dict.get("misleading_target")
    valid_turns = [turn for turn in turns if turn.get("hypotheses") is not None]

    if not valid_turns:
        return {"all_parse_errors": True}

    n_valid = len(valid_turns)
    oracle_in_trajectory = [
        oracle in _normalize_turn_hypotheses(turn)
        for turn in valid_turns
    ]
    oracle_retention_rate = sum(oracle_in_trajectory) / n_valid

    misleading_in_trajectory = [
        misleading_target in _normalize_turn_hypotheses(turn)
        for turn in valid_turns
    ] if misleading_target else []

    exact_matches = sum(
        1 for turn in valid_turns
        if turn.get("belief_metrics", {}).get("exact_match", False)
    )
    exact_match_rate = exact_matches / n_valid

    return {
        "oracle_retention_rate": round(oracle_retention_rate, 4),
        "oracle_in_trajectory": oracle_in_trajectory,
        "misleading_in_trajectory": misleading_in_trajectory,
        "exact_match_rate": round(exact_match_rate, 4),
        "format_success_rate": round(n_valid / len(turns), 4) if turns else None,
    }


def summarize_category_counts(counts: Mapping[str, int]) -> Dict[str, Any]:
    total = sum(int(counts.get(category, 0)) for category in CATEGORIES)
    summary: Dict[str, Any] = {
        "total": total,
        **{category: int(counts.get(category, 0)) for category in CATEGORIES},
    }
    summary.update({
        f"{category}_pct": round(summary[category] / max(total, 1) * 100, 2)
        for category in CATEGORIES
    })
    return summary


def build_stats_report(
    *,
    per_task_counts: Mapping[str, Mapping[str, int]],
    template_counts: Optional[Mapping[str, int]] = None,
    model: Optional[str] = None,
    prompt_style: Optional[str] = None,
    num_runs_per_task: Optional[int] = None,
    run_semantics: Optional[str] = None,
    repeats: Optional[int] = None,
    gpus: Optional[Sequence[int]] = None,
    circuit_types: Optional[Sequence[str]] = None,
    faults: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if model is not None:
        report["model"] = model
    if prompt_style is not None:
        report["prompt_style"] = prompt_style
    if num_runs_per_task is not None:
        report["num_runs_per_task"] = num_runs_per_task
    if run_semantics is not None:
        report["run_semantics"] = run_semantics
    if repeats is not None:
        report["repeats"] = repeats
    if gpus is not None:
        report["gpus"] = list(gpus)
    if circuit_types is not None:
        report["circuit_types"] = list(circuit_types)
    if faults is not None:
        report["faults"] = list(faults)

    template_counts = template_counts or {}
    total_counts: Counter[str] = Counter()
    per_task_report: Dict[str, Any] = {}
    for task_label, counts in sorted(per_task_counts.items()):
        total_counts.update(counts)
        task_summary = summarize_category_counts(counts)
        if task_label in template_counts:
            task_summary = {
                "template_count": int(template_counts[task_label]),
                **task_summary,
            }
        per_task_report[task_label] = task_summary

    report["per_task"] = per_task_report
    report["total"] = summarize_category_counts(total_counts)
    return report
