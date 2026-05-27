"""CD failed_stay depth: insert redundant hold measurements after lock."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from analysis.augment.rd_failed_stay_depth import find_lock_index
from analysis.augment.schema import attach_augmentation, build_augmented_case_id

_TURN_HEADER_RE = re.compile(r"^Turn (\d+)( \(CORRECTION\))?:", re.MULTILINE)


def _turn_header(turn_idx: int, *, correction: bool = False) -> str:
    if correction:
        return f"Turn {turn_idx} (CORRECTION):"
    return f"Turn {turn_idx}:"


def _clone_cd_turn_prompt(template_prompt: str, template_idx: int, new_idx: int) -> str:
    correction = "CORRECTION" in template_prompt
    old = _turn_header(template_idx, correction=correction)
    new = _turn_header(new_idx, correction=correction)
    return template_prompt.replace(old, new, 1)


def _renumber_cd_prompt(prompt: str, turn_map: Dict[int, int]) -> str:
    updated = prompt
    for old in sorted(turn_map.keys(), reverse=True):
        new = turn_map[old]
        if f"Turn {old} (CORRECTION):" in updated:
            updated = updated.replace(f"Turn {old} (CORRECTION):", f"Turn {new} (CORRECTION):")
        updated = re.sub(rf"\bTurn {old}\b", f"Turn {new}", updated)
    return updated


def augment_cd_failed_stay_depth(
    case: Dict[str, Any],
    *,
    n_redundant: int,
    seed: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if str(case.get("challenge_type", "")).lower() != "failed_stay":
        return None, "not_failed_stay"
    if n_redundant <= 0:
        return None, "n_redundant_non_positive"

    turns: List[Dict[str, Any]] = list(case.get("turns") or [])
    lock_idx = find_lock_index(case)
    if lock_idx is None:
        return None, "no_lock_index"

    lock_golden = sorted(turns[lock_idx].get("golden") or [])
    template_idx = len(turns) - 1
    if set(turns[template_idx].get("golden") or []) != set(lock_golden):
        template_idx = lock_idx

    new_turns = list(turns[: lock_idx + 1])
    for offset in range(n_redundant):
        turn_num = lock_idx + 1 + offset
        prompt = _clone_cd_turn_prompt(str(turns[template_idx]["prompt"]), template_idx, turn_num)
        new_turns.append({"prompt": prompt, "golden": list(lock_golden)})

    shift = n_redundant
    for orig_idx in range(lock_idx + 1, len(turns)):
        turn_map = {orig_idx: orig_idx + shift}
        prompt = _renumber_cd_prompt(str(turns[orig_idx]["prompt"]), turn_map)
        new_turns.append({"prompt": prompt, "golden": list(turns[orig_idx].get("golden") or [])})

    suffix = f"n{n_redundant}_s{seed}"
    new_case_id = build_augmented_case_id(str(case["case_id"]), "cdfailed_stay", suffix)
    params = {"n_redundant": n_redundant, "seed": seed, "lock_idx": lock_idx, "case_id": new_case_id}
    augmented = attach_augmentation(
        {**case, "turns": new_turns, "case_id": new_case_id},
        pipeline="cd_failed_stay_depth",
        params=params,
        source_case_id=str(case["case_id"]),
        extra={"lock_idx": lock_idx, "n_redundant": n_redundant},
    )
    return augmented, None
