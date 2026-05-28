"""FailedUpdate depth augmentation: delay CORRECTION with oracle-neutral evidence triples."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from analysis.depth_and_noise.augment.domain_bridge import (
    build_rules_map,
    oracle_label,
    parse_candidate_rules_from_system_prompt,
    sample_redundant_triple,
)
from analysis.depth_and_noise.augment.format_rd_turn import (
    format_evidence_turn,
    format_evidence_turn_9b_post_lock,
    is_9b_rd_case,
)
from analysis.depth_and_noise.augment.prompt_parse import renumber_rd_turn_prompt, rebuild_active_evidence
from analysis.depth_and_noise.augment.schema import attach_augmentation, build_augmented_case_id


def find_correction_index(turns: List[Dict[str, Any]]) -> Optional[int]:
    for idx, turn in enumerate(turns):
        if "CORRECTION" in str(turn.get("prompt", "")):
            return idx
    return None


def augment_rd_failed_update_depth(
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
    corr_idx = find_correction_index(turns)
    if corr_idx is None or corr_idx < 1:
        return None, "no_correction_turn"

    wrong_golden = sorted(turns[corr_idx - 1].get("golden") or [])
    if not wrong_golden:
        return None, "empty_wrong_golden"
    restored_golden = sorted(turns[corr_idx].get("golden") or case.get("target_set") or [case.get("oracle")])
    restored_survivors = {str(rule_id) for rule_id in restored_golden if str(rule_id)}
    if not restored_survivors:
        return None, "empty_restored_golden"

    oracle = str(case.get("oracle", "")).strip()
    if not oracle:
        return None, "missing_oracle"

    candidate_names = parse_candidate_rules_from_system_prompt(str(case.get("system_prompt", "")))
    if not candidate_names:
        return None, "no_candidate_rules"
    rules = build_rules_map(candidate_names)
    wrong_survivors: Set[str] = set(wrong_golden)

    prefix = turns[:corr_idx]
    suffix = turns[corr_idx:]

    rng = random.Random(seed)
    active_evidence = rebuild_active_evidence(turns, corr_idx - 1)
    post_lock_9b = is_9b_rd_case(case)

    delay_turn_records: List[Dict[str, Any]] = []
    for offset in range(delay_turns):
        triple = sample_redundant_triple(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            target_survivors=wrong_survivors,
            compatible_survivors=restored_survivors,
        )
        if triple is None:
            return None, "sample_failed_update_delay_triple_failed"
        result = oracle_label(rules, oracle, triple)
        active_evidence.append((triple, result))
        turn_num = corr_idx + offset
        if post_lock_9b:
            prompt = format_evidence_turn_9b_post_lock(turn=turn_num, triple=triple, result=result)
        else:
            prompt = format_evidence_turn(
                turn=turn_num,
                triple=triple,
                result=result,
                candidate_names=candidate_names,
            )
        delay_turn_records.append({"prompt": prompt, "golden": list(wrong_golden)})

    new_turns = list(prefix) + delay_turn_records

    turn_map = {}
    for old_idx in range(corr_idx, len(turns)):
        turn_map[old_idx] = old_idx + delay_turns

    for turn in suffix:
        prompt = renumber_rd_turn_prompt(str(turn["prompt"]), turn_map)
        new_turns.append({"prompt": prompt, "golden": list(turn.get("golden") or [])})

    suffix_tag = f"d{delay_turns}_s{seed}"
    new_case_id = build_augmented_case_id(str(case["case_id"]), "rdfailed_update", suffix_tag)
    params = {
        "delay_turns": delay_turns,
        "seed": seed,
        "corr_idx": corr_idx,
        "restored_golden": restored_golden,
        "case_id": new_case_id,
    }
    augmented = attach_augmentation(
        {**case, "turns": new_turns, "case_id": new_case_id},
        pipeline="rd_failed_update_depth",
        params=params,
        source_case_id=str(case["case_id"]),
        extra={"corr_idx": corr_idx, "delay_turns": delay_turns, "restored_golden": restored_golden},
    )
    return augmented, None
