"""FailedStay depth augmentation: insert redundant consistent evidence after lock."""

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
from analysis.depth_and_noise.augment.prompt_parse import has_correction, rebuild_active_evidence
from analysis.depth_and_noise.augment.schema import attach_augmentation, build_augmented_case_id

Triple = Tuple[int, int, int]


def find_lock_index(case: Dict[str, Any]) -> Optional[int]:
    oracle = str(case.get("oracle", ""))
    target = set(case.get("target_set") or [oracle])
    for idx, turn in enumerate(case.get("turns", [])):
        golden = set(turn.get("golden") or [])
        if golden == target or golden == {oracle}:
            return idx
    return None


def augment_rd_failed_stay_depth(
    case: Dict[str, Any],
    *,
    n_redundant: int,
    seed: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if str(case.get("cbm_challenge_type", "")).lower() != "failed_stay":
        return None, "not_failed_stay"
    if n_redundant <= 0:
        return None, "n_redundant_non_positive"

    lock_idx = find_lock_index(case)
    if lock_idx is None:
        return None, "no_lock_index"

    turns: List[Dict[str, Any]] = list(case.get("turns") or [])
    for idx in range(lock_idx + 1):
        if has_correction(str(turns[idx].get("prompt", ""))):
            return None, "correction_before_lock"

    oracle = str(case["oracle"])
    target_survivors: Set[str] = {oracle}
    candidate_names = parse_candidate_rules_from_system_prompt(str(case.get("system_prompt", "")))
    if not candidate_names:
        return None, "no_candidate_rules"
    rules = build_rules_map(candidate_names)

    rng = random.Random(seed)
    active_evidence = rebuild_active_evidence(turns, lock_idx)
    lock_golden = sorted(turns[lock_idx].get("golden") or [oracle])
    post_lock_9b = is_9b_rd_case(case)

    new_turns = list(turns[: lock_idx + 1])
    insert_at = lock_idx + 1
    for offset in range(n_redundant):
        triple = sample_redundant_triple(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            target_survivors=target_survivors,
        )
        if triple is None:
            return None, "sample_redundant_triple_failed"
        result = oracle_label(rules, oracle, triple)
        active_evidence.append((triple, result))
        turn_num = insert_at + offset
        if post_lock_9b:
            prompt = format_evidence_turn_9b_post_lock(
                turn=turn_num,
                triple=triple,
                result=result,
            )
        else:
            prompt = format_evidence_turn(
                turn=turn_num,
                triple=triple,
                result=result,
                candidate_names=candidate_names,
            )
        new_turns.append({"prompt": prompt, "golden": list(lock_golden)})

    shift = n_redundant
    for orig_idx in range(lock_idx + 1, len(turns)):
        orig_turn = turns[orig_idx]
        prompt = str(orig_turn["prompt"])
        new_turn_num = orig_idx + shift
        prompt = prompt.replace(f"**Turn {orig_idx} evidence:**", f"**Turn {new_turn_num} evidence:**")
        new_turns.append({"prompt": prompt, "golden": list(orig_turn.get("golden") or [])})

    suffix = f"n{n_redundant}_s{seed}"
    new_case_id = build_augmented_case_id(str(case["case_id"]), "rdfailed_stay", suffix)
    params = {"n_redundant": n_redundant, "seed": seed, "lock_idx": lock_idx, "case_id": new_case_id}
    augmented = attach_augmentation(
        {**case, "turns": new_turns, "case_id": new_case_id},
        pipeline="rd_failed_stay_depth",
        params=params,
        source_case_id=str(case["case_id"]),
        extra={"lock_idx": lock_idx, "n_redundant": n_redundant},
    )
    return augmented, None
