"""Challenge-specific trajectory annotations for Scenario A experiments."""

from typing import Any, Dict, List

from task_a.core.config import (
    CHALLENGE_MISRECORD_CORRECTION,
    CHALLENGE_SHRINK_THEN_HOLD,
    challenge_display_name,
    normalize_challenge_type,
)


def annotate_challenge_trajectory(
    trajectory: Dict[str, Any],
    challenge_sequence: Dict[str, Any],
) -> Dict[str, Any]:
    """Annotate a trajectory with challenge-specific metrics."""
    challenge_type = normalize_challenge_type(challenge_sequence["challenge_type"])
    challenge_turns = set(challenge_sequence["challenge_turns"])
    convergence_turn = challenge_sequence["convergence_turn"]
    ground_truth_steps = challenge_sequence["ground_truth"]

    turns = trajectory.get("turns", [])

    phase1_converged = False
    if len(turns) > convergence_turn:
        turn_data = turns[convergence_turn]
        agent_hyps = set(turn_data.get("hypotheses") or [])
        gt_surv = ground_truth_steps[convergence_turn]["survivors"]
        phase1_converged = (agent_hyps == gt_surv)

    challenge_results = []
    all_exact_matches = []
    for turn_index, turn_data in enumerate(turns):
        agent_hyps = set(turn_data.get("hypotheses") or [])
        if turn_index < len(ground_truth_steps):
            gt_surv = ground_truth_steps[turn_index]["survivors"]
        else:
            gt_surv = ground_truth_steps[-1]["survivors"]

        exact = (agent_hyps == gt_surv)
        all_exact_matches.append(exact)

        false_ret = agent_hyps - gt_surv
        false_elim = gt_surv - agent_hyps

        entry = {
            "turn": turn_index,
            "exact_match": exact,
            "false_retention": sorted(false_ret),
            "false_elimination": sorted(false_elim),
            "is_challenge_turn": turn_index in challenge_turns,
        }
        if turn_index in challenge_turns:
            challenge_results.append(entry)

    challenge_pass = (
        all(result["exact_match"] for result in challenge_results)
        if challenge_results else False
    )

    type_metrics: Dict[str, Any] = {}
    if challenge_type == CHALLENGE_MISRECORD_CORRECTION and challenge_results:
        type_metrics["correction_pass"] = challenge_results[0]["exact_match"]
    elif challenge_type == CHALLENGE_SHRINK_THEN_HOLD and len(challenge_results) >= 2:
        type_metrics["shrink_pass"] = challenge_results[0]["exact_match"]
        type_metrics["hold_pass"] = challenge_results[1]["exact_match"]

    return {
        "challenge_type": challenge_type,
        "phase1_converged": phase1_converged,
        "challenge_pass": challenge_pass,
        "challenge_results": challenge_results,
        "type_metrics": type_metrics,
        "exact_match_rate": (
            sum(all_exact_matches) / len(all_exact_matches)
            if all_exact_matches else 0.0
        ),
        "oracle_in_all_turns": all(
            trajectory.get("rule_name") in set(turn.get("hypotheses") or [])
            for turn in turns
        ),
    }


def summarize_trajectory(
    trajectory: Dict[str, Any],
    annotation: Dict[str, Any],
    challenge_type: str,
    seed: int,
) -> Dict[str, Any]:
    """Compact summary row for one trajectory."""
    challenge_type = normalize_challenge_type(challenge_type)
    return {
        "experiment_id": trajectory.get("experiment_id"),
        "rule_name": trajectory.get("rule_name"),
        "challenge_type": challenge_type,
        "challenge_label": challenge_display_name(challenge_type),
        "seed": seed,
        "phase1_converged": annotation["phase1_converged"],
        "challenge_pass": annotation["challenge_pass"],
        "exact_match_rate": round(annotation["exact_match_rate"], 4),
        "oracle_in_all_turns": annotation["oracle_in_all_turns"],
        "type_metrics": annotation.get("type_metrics", {}),
        "n_turns": trajectory.get("n_turns_played"),
    }
