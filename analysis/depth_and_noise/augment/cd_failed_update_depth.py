"""CD failed_update depth: delay CORRECTION by cloning the last pre-correction measurement turn."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from analysis.depth_and_noise.augment.cd_failed_stay_depth import _clone_cd_turn_prompt
from analysis.depth_and_noise.augment.schema import attach_augmentation, build_augmented_case_id


def find_cd_correction_index(turns: List[Dict[str, Any]]) -> Optional[int]:
    for idx, turn in enumerate(turns):
        if "CORRECTION" in str(turn.get("prompt", "")):
            return idx
    return None


def _renumber_cd_prompt(prompt: str, turn_map: Dict[int, int]) -> str:
    updated = prompt
    for old in sorted(turn_map.keys(), reverse=True):
        new = turn_map[old]
        if f"Turn {old} (CORRECTION):" in updated:
            updated = updated.replace(f"Turn {old} (CORRECTION):", f"Turn {new} (CORRECTION):")
        updated = re.sub(rf"\bTurn {old}\b", f"Turn {new}", updated)
    return updated


def augment_cd_failed_update_depth(
    case: Dict[str, Any],
    *,
    delay_turns: int,
    seed: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if str(case.get("cbm_challenge_type", "")).lower() != "failed_update":
        return None, "not_failed_update"
    if delay_turns <= 0:
        return None, "delay_turns_non_positive"

    turns: List[Dict[str, Any]] = list(case.get("turns") or [])
    corr_idx = find_cd_correction_index(turns)
    if corr_idx is None or corr_idx < 1:
        return None, "no_correction_turn"

    wrong_golden = sorted(turns[corr_idx - 1].get("golden") or [])
    if not wrong_golden:
        return None, "empty_wrong_golden"
    restored_golden = sorted(turns[corr_idx].get("golden") or case.get("target_set") or [case.get("oracle")])
    if not restored_golden:
        return None, "empty_restored_golden"

    template_idx = corr_idx - 1
    template_prompt = str(turns[template_idx].get("prompt", ""))

    prefix = turns[:corr_idx]
    suffix = turns[corr_idx:]

    delay_records: List[Dict[str, Any]] = []
    for offset in range(delay_turns):
        turn_num = corr_idx + offset
        prompt = _clone_cd_turn_prompt(template_prompt, template_idx, turn_num)
        delay_records.append({"prompt": prompt, "golden": list(wrong_golden)})

    new_turns = list(prefix) + delay_records
    turn_map = {old: old + delay_turns for old in range(corr_idx, len(turns))}
    for turn in suffix:
        prompt = _renumber_cd_prompt(str(turn["prompt"]), turn_map)
        new_turns.append({"prompt": prompt, "golden": list(turn.get("golden") or [])})

    suffix_tag = f"d{delay_turns}_s{seed}"
    new_case_id = build_augmented_case_id(str(case["case_id"]), "cdfailed_update", suffix_tag)
    params = {
        "delay_turns": delay_turns,
        "seed": seed,
        "corr_idx": corr_idx,
        "restored_golden": restored_golden,
        "case_id": new_case_id,
    }
    augmented = attach_augmentation(
        {**case, "turns": new_turns, "case_id": new_case_id},
        pipeline="cd_failed_update_depth",
        params=params,
        source_case_id=str(case["case_id"]),
        extra={"corr_idx": corr_idx, "delay_turns": delay_turns, "restored_golden": restored_golden},
    )
    return augmented, None
