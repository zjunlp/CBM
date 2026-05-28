"""Render Rule Discovery turn prompts for augmented cases."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from task_a.experiments.belief_stats import Triple, triple_text
from task_a.core.rules import get_rule as get_extended_rule


def _turn_prefix(turn: int) -> str:
    return "Let's begin.\n\n" if turn == 0 else ""


def format_rule_predictions(
    candidate_names: List[str],
    triple: Triple,
    *,
    result: str,
    annotate: bool = True,
) -> str:
    expected_yes = str(result).upper() == "YES"
    lines = []
    for name in candidate_names:
        pred_yes = bool(get_extended_rule(name).validate(triple))
        pred = "YES" if pred_yes else "NO"
        suffix = ""
        if annotate:
            suffix = " (consistent)" if pred_yes == expected_yes else " (CONTRADICTS evidence → eliminated)"
        lines.append(f"  - {name} → {pred}{suffix}")
    return "\n".join(lines)


def is_9b_rd_case(case: Dict[str, Any]) -> bool:
    """9B RD drops rule-prediction tables after the model has converged."""
    system_prompt = str(case.get("system_prompt", ""))
    if "For some non-corrected turns" in system_prompt:
        return True
    turns = case.get("turns") or []
    for turn in turns[1:]:
        if "Rule predictions" not in str(turn.get("prompt", "")):
            return True
    return False


def format_evidence_turn_9b_post_lock(
    *,
    turn: int,
    triple: Triple,
    result: str,
) -> str:
    """Post-convergence 9B turn: numbered triple only, no rule-prediction table."""
    closing = (
        "Please update your hypotheses based on this evidence."
        if turn == 0
        else "Please update your hypotheses using all currently active evidence."
    )
    return (
        f"{_turn_prefix(turn)}"
        f"**Turn {turn} evidence:**\n"
        f"1. Triple {triple_text(triple)}: **{result}**\n\n"
        f"{closing}"
    )


def format_evidence_turn(
    *,
    turn: int,
    triple: Triple,
    result: str,
    candidate_names: List[str],
    previous_hypotheses: Optional[List[str]] = None,
) -> str:
    body = (
        f"{_turn_prefix(turn)}"
        f"**Turn {turn} evidence:**\n"
        f"Triple {triple_text(triple)}: **{result}**\n\n"
        f"Rule predictions for this triple:\n"
        f"{format_rule_predictions(candidate_names, triple, result=result, annotate=True)}\n\n"
    )
    if turn > 0 and previous_hypotheses is not None:
        expected_yes = str(result).upper() == "YES"
        matching = [
            name
            for name in candidate_names
            if bool(get_extended_rule(name).validate(triple)) == expected_yes
        ]
        body += (
            f"Previous hypothesis: {', '.join(previous_hypotheses) if previous_hypotheses else 'none'}\n"
            f"Current matching rule IDs: {', '.join(matching) if matching else 'none'}\n"
            "Update rule: keep only rule IDs that are in BOTH lists above. Do not add new rule IDs.\n\n"
        )
    if turn == 0:
        body += "Please update your hypotheses based on this evidence."
    else:
        body += "Please update your hypotheses using all currently active evidence."
    return body
