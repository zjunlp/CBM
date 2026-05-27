"""Host-driven sequence generators for larger candidate spaces.

This module provides explicit failed_stay/failed_update generators that support arbitrary
candidate spaces, including 5-rule and 10-rule (benchmark + heldout) setups.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from task_a.core.config import ChallengeSequence
from task_a.core.rules import EXTENDED_RULES as RULES, Triple, get_rule, resolve_heldout_rules

TRIPLE_RANGE = (-20, 20)
MAX_SAMPLE_ATTEMPTS = 12000
SYSTEM_RNG = random.SystemRandom()


def _random_triple(rng: random.Random) -> Triple:
    lo, hi = TRIPLE_RANGE
    return (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))


def _sample_label_unseeded() -> str:
    """Sample YES/NO from OS randomness (not tied to experiment seed)."""
    return "YES" if SYSTEM_RNG.random() < 0.5 else "NO"


def _sample_label(rng: random.Random) -> str:
    """Sample YES/NO from the experiment RNG so generated cases are reproducible."""
    return "YES" if rng.random() < 0.5 else "NO"


def _all_rules(heldout_set: str = "easy") -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(RULES)
    merged.update(resolve_heldout_rules(heldout_set))
    return merged


def _compute_survivors(
    rules: Dict[str, Any],
    evidence: List[Tuple[Triple, str]],
) -> Set[str]:
    survivors = set(rules.keys())
    for triple, label in evidence:
        expected = (label == "YES")
        survivors = {r for r in survivors if rules[r].validate(triple) == expected}
    return survivors


def _compute_retraction_ground_truth_v2(
    *,
    oracle: str,
    rules: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute per-turn ground truth with misrecord + retraction events."""
    oracle_rule = rules[oracle]
    active_evidence: List[Tuple[Triple, str]] = []
    event_to_active: Dict[int, int] = {}
    steps: List[Dict[str, Any]] = []

    for i, event in enumerate(events):
        etype = event["type"]
        if etype == "evidence":
            triple = tuple(event["triple"])
            true_result = "YES" if oracle_rule.validate(triple) else "NO"
            result = event.get("recorded_result", true_result)
            event_to_active[i] = len(active_evidence)
            active_evidence.append((triple, result))
            survivors = _compute_survivors(rules, active_evidence)
            steps.append(
                {
                    "turn": i,
                    "event_type": "evidence",
                    "triple": triple,
                    "result": result,
                    "survivors": sorted(survivors),
                }
            )
            continue

        if etype == "retraction":
            raw_turns = event.get("retract_turns")
            if raw_turns is not None:
                retract_turns = [int(rt) for rt in raw_turns]
            else:
                retract_turns = [int(event["retract_turn"])]

            for rt in retract_turns:
                if rt not in event_to_active:
                    raise ValueError(f"Invalid retract_turn={rt}")

            # Remove in descending active-index order so earlier indices stay stable.
            pairs = sorted(
                ((rt, event_to_active[rt]) for rt in retract_turns),
                key=lambda pair: -pair[1],
            )
            removed: List[Tuple[int, Triple]] = []
            for rt, idx in pairs:
                triple_removed, _ = active_evidence.pop(idx)
                removed.append((rt, triple_removed))
                del event_to_active[rt]
                for k, v in list(event_to_active.items()):
                    if v > idx:
                        event_to_active[k] = v - 1

            has_replacement = ("new_triple" in event) and len(retract_turns) == 1
            if has_replacement:
                single_rt = retract_turns[0]
                original_triple = next(tr for (rt, tr) in removed if rt == single_rt)
                use_retracted_triple = bool(event.get("use_retracted_triple", False))
                triple = original_triple if use_retracted_triple else tuple(event["new_triple"])
                true_result = "YES" if oracle_rule.validate(triple) else "NO"
                result = event.get("new_result", true_result)
                event_to_active[i] = len(active_evidence)
                active_evidence.append((triple, result))
            else:
                # Pure retraction (possibly multi-turn) without replacement evidence.
                triple = removed[0][1] if removed else (0, 0, 0)
                result = ""

            survivors = _compute_survivors(rules, active_evidence)
            step_record: Dict[str, Any] = {
                "turn": i,
                "event_type": "retraction",
                "triple": triple,
                "result": result,
                "survivors": sorted(survivors),
            }
            if len(retract_turns) == 1:
                step_record["retract_turn"] = retract_turns[0]
            else:
                step_record["retract_turns"] = retract_turns
            steps.append(step_record)
            continue

        raise ValueError(f"Unknown event type: {etype}")

    return steps


def _find_true_step(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    target_size: int,
) -> Optional[Tuple[Triple, str, Set[str]]]:
    """Find one true-feedback evidence step reaching exact target size."""
    oracle_rule = rules[oracle]
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        desired_result = _sample_label(rng)
        if (desired_result == "YES") != oracle_rule.validate(t):
            continue
        result = desired_result
        survivors = _compute_survivors(rules, active_evidence + [(t, result)])
        if oracle in survivors and len(survivors) == target_size:
            return t, result, survivors
    return None


def _find_true_step_between_sizes(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    min_size: int,
    max_size: int,
) -> Optional[Tuple[Triple, str, Set[str]]]:
    """Find a true-feedback step whose survivor size is within [min_size, max_size]."""
    if min_size > max_size:
        return None
    oracle_rule = rules[oracle]
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        desired_result = _sample_label(rng)
        if (desired_result == "YES") != oracle_rule.validate(t):
            continue
        result = desired_result
        survivors = _compute_survivors(rules, active_evidence + [(t, result)])
        if oracle in survivors and min_size <= len(survivors) <= max_size:
            return t, result, survivors
    return None


def _find_misrecord_pair(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    wrong_target_size: int,
    corrected_target_size: int,
    corrected_must_be_oracle_singleton: bool,
) -> Optional[Tuple[Triple, str, str, Set[str], Set[str]]]:
    """Find one triple where flipped result gives wrong size, true result gives corrected size."""
    oracle_rule = rules[oracle]
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        true_result = "YES" if oracle_rule.validate(t) else "NO"
        wrong_result = "NO" if true_result == "YES" else "YES"

        wrong_survivors = _compute_survivors(rules, active_evidence + [(t, wrong_result)])
        corrected_survivors = _compute_survivors(rules, active_evidence + [(t, true_result)])

        if len(wrong_survivors) != wrong_target_size:
            continue
        if len(corrected_survivors) != corrected_target_size:
            continue
        if oracle not in corrected_survivors:
            continue
        if corrected_must_be_oracle_singleton and corrected_survivors != {oracle}:
            continue

        return t, wrong_result, true_result, wrong_survivors, corrected_survivors
    return None


def _find_no_elimination_step(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    current_survivors: Set[str],
) -> Optional[Tuple[Triple, str]]:
    """Find an evidence triple that keeps survivors unchanged."""
    oracle_rule = rules[oracle]
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        desired_result = _sample_label(rng)
        if (desired_result == "YES") != oracle_rule.validate(t):
            continue
        result = desired_result
        survivors = _compute_survivors(rules, active_evidence + [(t, result)])
        if survivors == current_survivors:
            return t, result
    return None


def _find_misrecord_eliminate_oracle_and_one(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    current_survivors: Set[str],
) -> Optional[Tuple[Triple, str, str, Set[str], Set[str]]]:
    """Find a misrecord step that eliminates exactly 2 rules including oracle.

    From a 5-rule survivor set, wrong-recorded result leaves 3 survivors and
    removes exactly two rules including oracle.
    """
    oracle_rule = rules[oracle]
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        true_result = "YES" if oracle_rule.validate(t) else "NO"
        wrong_result = "NO" if true_result == "YES" else "YES"

        wrong_survivors = _compute_survivors(rules, active_evidence + [(t, wrong_result)])
        true_survivors = _compute_survivors(rules, active_evidence + [(t, true_result)])

        if not wrong_survivors.issubset(current_survivors):
            continue
        eliminated = current_survivors - wrong_survivors
        if len(eliminated) != 2:
            continue
        if oracle not in eliminated:
            continue
        if len(wrong_survivors) != len(current_survivors) - 2:
            continue

        return t, wrong_result, true_result, wrong_survivors, true_survivors
    return None


def generate_failed_update_sequence_v3_ordered_four_evidence(
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    target_sizes: Optional[List[int]] = None,
    heldout_set: str = "easy",
) -> Optional[ChallengeSequence]:
    """Generate failed_update-v3 with ordered four-evidence collapse before correction.

    Target trajectory (10-rule setup):
    10 -> 8 -> 5 -> 5 -> 3 -> 3 -> 1 -> 2

    After reaching 5 survivors, four evidence turns are ordered as:
    1) no elimination,
    2) eliminate oracle side (misrecorded; removes exactly 2 including oracle),
    3) no elimination,
    4) eliminate the other two (reaches 1).
    Final turn retracts step 2 and should recover to 2 survivors including oracle.
    """
    all_rules = _all_rules(heldout_set)
    for name in candidate_names:
        get_rule(name)

    if oracle not in all_rules:
        return None

    rules = {n: all_rules[n] for n in candidate_names}
    sizes = list(target_sizes) if target_sizes is not None else [8, 5, 5, 3, 3, 1, 2]
    if sizes != [8, 5, 5, 3, 3, 1, 2]:
        return None

    for _ in range(300):
        events: List[Dict[str, Any]] = []
        active_evidence: List[Tuple[Triple, str]] = []

        step0 = _find_true_step(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            target_size=8,
        )
        if step0 is None:
            continue
        t0, r0, _ = step0
        events.append({"type": "evidence", "triple": t0})
        active_evidence.append((t0, r0))

        step1 = _find_true_step(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            target_size=5,
        )
        if step1 is None:
            continue
        t1, r1, s5 = step1
        events.append({"type": "evidence", "triple": t1})
        active_evidence.append((t1, r1))

        # Evidence A: no elimination (5 -> 5)
        step2 = _find_no_elimination_step(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            current_survivors=s5,
        )
        if step2 is None:
            continue
        t2, r2 = step2
        events.append({"type": "evidence", "triple": t2})
        active_evidence.append((t2, r2))

        # Evidence B (misrecorded): eliminate oracle side (5 -> 3)
        step3 = _find_misrecord_eliminate_oracle_and_one(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            current_survivors=s5,
        )
        if step3 is None:
            continue
        t3, wrong3, true3, s3_wrong, _s_true_after3 = step3
        events.append({"type": "evidence", "triple": t3, "recorded_result": wrong3})
        active_evidence.append((t3, wrong3))

        # Evidence C: no elimination on wrong path (3 -> 3)
        step4 = _find_no_elimination_step(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            current_survivors=s3_wrong,
        )
        if step4 is None:
            continue
        t4, r4 = step4
        events.append({"type": "evidence", "triple": t4})
        active_evidence.append((t4, r4))

        # Evidence D: eliminate other two (3 -> 1), still on wrong path
        step5 = _find_true_step(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            target_size=1,
        )
        if step5 is None:
            continue
        t5, r5, s1 = step5
        if oracle in s1:
            continue
        events.append({"type": "evidence", "triple": t5})
        active_evidence.append((t5, r5))

        # Final correction: retract Evidence B (misrecord) and flip it back.
        events.append(
            {
                "type": "retraction",
                "retract_turn": 3,
                "new_triple": t3,
                "use_retracted_triple": True,
            }
        )

        ground_truth = _compute_retraction_ground_truth_v2(
            oracle=oracle,
            rules=rules,
            events=events,
        )
        size_trace = [len(step["survivors"]) for step in ground_truth]
        if size_trace != sizes:
            continue
        if oracle not in set(ground_truth[-1]["survivors"]):
            continue

        return {
            "challenge_type": "failed_update_v3_ordered_four_evidence",
            "oracle": oracle,
            "events": events,
            "triples": None,
            "ground_truth": ground_truth,
            "challenge_turns": [6],
            "convergence_turn": 5,
            "total_turns": len(events),
            "target_sizes": sizes,
        }

    return None


def generate_failed_update_sequence_v2(
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    target_sizes: Optional[List[int]] = None,
    heldout_set: str = "easy",
) -> Optional[ChallengeSequence]:
    """Generate failed_update-v2 with one misrecord/correction cycle.

    Default trajectory in a 10-rule setup is:
    10 -> 8 -> 5 -> 1 (wrong) -> 4 (corrected, includes oracle)
    represented by target sizes [8, 5, 1, 4] after each event.
    """
    all_rules = _all_rules(heldout_set)
    for name in candidate_names:
        get_rule(name)

    if oracle not in all_rules:
        return None

    rules = {n: all_rules[n] for n in candidate_names}
    sizes = list(target_sizes) if target_sizes is not None else [8, 5, 1, 4]
    if len(sizes) not in {3, 4}:
        return None

    if len(sizes) == 3:
        s_first, w1, c4 = sizes
        if w1 != 1 or c4 != 4:
            return None
        if s_first > len(candidate_names):
            return None

        events: List[Dict[str, Any]] = []
        active_evidence: List[Tuple[Triple, str]] = []

        step0 = _find_true_step(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            target_size=s_first,
        )
        if step0 is None:
            return None
        t0, r0, _ = step0
        events.append({"type": "evidence", "triple": t0})
        active_evidence.append((t0, r0))

        mis1 = _find_misrecord_pair(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            wrong_target_size=w1,
            corrected_target_size=c4,
            corrected_must_be_oracle_singleton=False,
        )
        if mis1 is None:
            return None
        t1, wrong1, true1, wrong1_survivors, corrected4_survivors = mis1
        events.append({"type": "evidence", "triple": t1, "recorded_result": wrong1})
        active_evidence.append((t1, wrong1))

        events.append(
            {
                "type": "retraction",
                "retract_turn": 1,
                "new_triple": t1,
                "use_retracted_triple": True,
            }
        )
        active_evidence.pop()
        active_evidence.append((t1, true1))

        ground_truth = _compute_retraction_ground_truth_v2(
            oracle=oracle,
            rules=rules,
            events=events,
        )

        if len(ground_truth) != 3:
            return None
        size_trace = [len(step["survivors"]) for step in ground_truth]
        if size_trace != sizes:
            return None
        if oracle not in set(ground_truth[2]["survivors"]):
            return None

        return {
            "challenge_type": "failed_update_v2_single_correction",
            "oracle": oracle,
            "events": events,
            "triples": None,
            "ground_truth": ground_truth,
            "challenge_turns": [2],
            "convergence_turn": 1,
            "total_turns": len(events),
            "target_sizes": sizes,
            "wrong_survivors_before_corrections": [sorted(wrong1_survivors)],
            "corrected_survivors": [sorted(corrected4_survivors)],
        }

    s8, s5, w1, c4 = sizes
    if w1 != 1 or c4 != 4:
        return None
    if s8 > len(candidate_names) or s5 > s8:
        return None

    events: List[Dict[str, Any]] = []
    active_evidence: List[Tuple[Triple, str]] = []

    step0 = _find_true_step(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        target_size=s8,
    )
    if step0 is None:
        return None
    t0, r0, _ = step0
    events.append({"type": "evidence", "triple": t0})
    active_evidence.append((t0, r0))

    step1 = _find_true_step(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        target_size=s5,
    )
    if step1 is None:
        return None
    t1, r1, _ = step1
    events.append({"type": "evidence", "triple": t1})
    active_evidence.append((t1, r1))

    mis1 = _find_misrecord_pair(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        wrong_target_size=w1,
        corrected_target_size=c4,
        corrected_must_be_oracle_singleton=False,
    )
    if mis1 is None:
        return None
    t2, wrong2, true2, wrong1_survivors, corrected4_survivors = mis1
    events.append({"type": "evidence", "triple": t2, "recorded_result": wrong2})
    active_evidence.append((t2, wrong2))

    events.append(
        {
            "type": "retraction",
            "retract_turn": 2,
            "new_triple": t2,
            "use_retracted_triple": True,
        }
    )
    active_evidence.pop()
    active_evidence.append((t2, true2))

    ground_truth = _compute_retraction_ground_truth_v2(
        oracle=oracle,
        rules=rules,
        events=events,
    )

    if len(ground_truth) != 4:
        return None
    size_trace = [len(step["survivors"]) for step in ground_truth]
    if size_trace != sizes:
        return None
    if oracle not in set(ground_truth[3]["survivors"]):
        return None

    return {
        "challenge_type": "failed_update_v2_single_correction",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [3],
        "convergence_turn": 2,
        "total_turns": len(events),
        "target_sizes": sizes,
        "wrong_survivors_before_corrections": [sorted(wrong1_survivors)],
        "corrected_survivors": [sorted(corrected4_survivors)],
    }


def _find_no_elim_step(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    expected_survivors: Set[str],
) -> Optional[Tuple[Triple, str]]:
    """Find an evidence triple that does not eliminate any current survivor."""
    oracle_rule = rules[oracle]
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        desired_result = _sample_label_unseeded()
        if (desired_result == "YES") != oracle_rule.validate(t):
            continue
        result = desired_result
        survivors = _compute_survivors(rules, active_evidence + [(t, result)])
        if survivors == expected_survivors:
            return t, result
    return None


def _matching_rules_for_triple(rules: Dict[str, Any], triple: Triple) -> Set[str]:
    return {name for name, rule in rules.items() if rule.validate(triple)}


def _find_singleton_step_with_exact_profile(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    required_profile: Set[str],
    forbidden_triples: Optional[Set[Triple]] = None,
) -> Optional[Tuple[Triple, str]]:
    """Find a YES-labeled evidence that keeps singleton oracle and matches exact rule profile."""
    forbidden = forbidden_triples or set()
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        if t in forbidden:
            continue
        if not rules[oracle].validate(t):
            continue

        matched = _matching_rules_for_triple(rules, t)
        if matched != required_profile:
            continue

        survivors = _compute_survivors(rules, active_evidence + [(t, "YES")])
        if survivors == {oracle}:
            return t, "YES"
    return None


def _find_best_singleton_interference_profile(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    max_attempts: int,
    forbidden_triples: Optional[Set[Triple]] = None,
) -> Optional[Tuple[Set[str], Triple]]:
    """Find a singleton-preserving oracle-true triple with a targeted interference profile.

    Required by failed_stay interference:
    - must include oracle
    - should include at least one non-oracle rule if possible

    Strategy: prefer profile sizes 2-3 (oracle + 1-2 already-eliminated rules).
    This creates a cognitively challenging scenario where the model sees a small
    number of specific eliminated rules re-appearing as consistent with new evidence,
    inducing genuine belief failed_stay rather than trivial all-rules-agree noise.
    Falls back to the smallest available profile if the preferred range is not found.
    """
    forbidden = forbidden_triples or set()

    # Collect one representative candidate per profile size.
    candidates_by_size: Dict[int, Tuple[Set[str], Triple]] = {}

    for _ in range(max_attempts):
        t = _random_triple(rng)
        if t in forbidden:
            continue
        if not rules[oracle].validate(t):
            continue

        survivors = _compute_survivors(rules, active_evidence + [(t, "YES")])
        if survivors != {oracle}:
            continue

        profile = _matching_rules_for_triple(rules, t)
        if oracle not in profile:
            continue

        sz = len(profile)
        if sz not in candidates_by_size:
            candidates_by_size[sz] = (profile, t)

        # Early-exit once we have a sample in the preferred range.
        if sz in {2, 3}:
            break

    if not candidates_by_size:
        return None

    # Prefer profile size 2, then 3, then fall back to globally smallest.
    for preferred in (2, 3):
        if preferred in candidates_by_size:
            return candidates_by_size[preferred]

    min_sz = min(candidates_by_size)
    return candidates_by_size[min_sz]


def generate_failed_update_sequence_v2_host(
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str = "easy",
    evidence_per_round: int = 4,
) -> Optional[ChallengeSequence]:
    """Generate host-driven failed_update sequence with configurable evidence per round.

    Supported modes:
    - evidence_per_round=4: batched host mode (model updates once every 4 evidences)
    - evidence_per_round=1: single-evidence mode with coarse convergence
      over rounds (10->5->1), then correction recovers to 4 survivors.

    For evidence_per_round=4, batch rounds are:
    - Round A (4 evidences): converge 10 -> 5
    - Round B (4 evidences): converge 5 -> 1 (wrong singleton)
    - Final correction: correct the misrecorded evidence, recover 1 -> 4.

    Fine-grained survivor trace over all events:
    [5, 5, 5, 5, 5, 5, 5, 1, 4]
    """
    if evidence_per_round not in {1, 4}:
        raise ValueError("evidence_per_round must be 1 or 4")

    if evidence_per_round == 1:
        seq = generate_failed_update_sequence_v2(
            oracle=oracle,
            rng=rng,
            candidate_names=candidate_names,
            target_sizes=[5, 1, 4],
            heldout_set=heldout_set,
        )
        if seq is None:
            return None

        seq["challenge_type"] = "failed_update_v2_host_single_evidence_single_correction"
        seq["coarse_target_sizes"] = [5, 1, 4]
        seq["evidence_per_round"] = 1
        return seq

    all_rules = _all_rules(heldout_set)
    for name in candidate_names:
        get_rule(name)

    if oracle not in all_rules:
        return None

    rules = {n: all_rules[n] for n in candidate_names}

    events: List[Dict[str, Any]] = []
    active_evidence: List[Tuple[Triple, str]] = []
    evidence_batch_ranges: List[List[int]] = []

    # 10 -> 5
    step0 = _find_true_step(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        target_size=5,
    )
    if step0 is None:
        return None
    t0, r0, s5 = step0
    events.append({"type": "evidence", "triple": t0})
    active_evidence.append((t0, r0))

    # Round A extras: keep survivors at 5 for total 4 evidences.
    for _ in range(3):
        extra = _find_no_elim_step(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            expected_survivors=s5,
        )
        if extra is None:
            return None
        t_extra, r_extra = extra
        events.append({"type": "evidence", "triple": t_extra})
        active_evidence.append((t_extra, r_extra))
    evidence_batch_ranges.append([0, 3])

    # Round B starts with three no-elimination evidences, then the misrecord
    # collapses 5 -> 1. Putting the misrecord last avoids post-correction
    # evidence accidentally eliminating rules from the corrected 4-survivor set.
    for _ in range(3):
        extra = _find_no_elim_step(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            expected_survivors=s5,
        )
        if extra is None:
            return None
        t_extra, r_extra = extra
        events.append({"type": "evidence", "triple": t_extra})
        active_evidence.append((t_extra, r_extra))

    mis = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        desired_true = _sample_label_unseeded()
        if (desired_true == "YES") != rules[oracle].validate(t):
            continue
        true_result = desired_true
        wrong_result = "NO" if true_result == "YES" else "YES"
        wrong_survivors = _compute_survivors(rules, active_evidence + [(t, wrong_result)])
        if len(wrong_survivors) != 1:
            continue
        if oracle in wrong_survivors:
            continue
        # Must eliminate the current 5-survivor set down to a wrong singleton.
        removed = s5 - wrong_survivors
        if len(removed) != 4 or oracle not in removed:
            continue
        corrected_survivors = _compute_survivors(rules, active_evidence + [(t, true_result)])
        if len(corrected_survivors) != 4 or oracle not in corrected_survivors:
            continue
        mis = (t, wrong_result, true_result, wrong_survivors, corrected_survivors)
        break
    if mis is None:
        return None

    t_mis, wrong_mis, _true_mis, s1_wrong, s4_corrected = mis
    events.append({"type": "evidence", "triple": t_mis, "recorded_result": wrong_mis})
    active_evidence.append((t_mis, wrong_mis))
    evidence_batch_ranges.append([4, 7])

    # Final correction: retract the oracle-eliminating evidence at event index 7.
    events.append(
        {
            "type": "retraction",
            "retract_turn": 7,
            "new_triple": t_mis,
            "use_retracted_triple": True,
        }
    )

    ground_truth = _compute_retraction_ground_truth_v2(
        oracle=oracle,
        rules=rules,
        events=events,
    )
    if len(ground_truth) != 9:
        return None
    trace = [len(step["survivors"]) for step in ground_truth]
    if trace != [5, 5, 5, 5, 5, 5, 5, 1, 4]:
        return None
    if oracle not in set(ground_truth[-1]["survivors"]):
        return None

    return {
        "challenge_type": "failed_update_v2_host_ordered_single_correction",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [8],
        "convergence_turn": 7,
        "total_turns": len(events),
        "target_sizes": [5, 5, 5, 5, 5, 5, 5, 1, 4],
        "coarse_target_sizes": [5, 1, 4],
        "evidence_batch_ranges": evidence_batch_ranges,
        "ordered_collapse_segment": {
            "from_turn": 4,
            "to_turn": 7,
            "pattern": [
                "no_elimination",
                "no_elimination",
                "no_elimination",
                "misrecord_eliminate_to_wrong_singleton",
            ],
        },
        "wrong_survivors_before_correction": sorted(s1_wrong),
        "corrected_survivors": sorted(s4_corrected),
        "evidence_per_round": 4,
    }


def generate_failed_stay_sequence_v2(
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    target_sizes: Optional[List[int]] = None,
    heldout_set: str = "easy",
    evidence_per_round: int = 1,
    post_convergence_interference_rounds: int = 0,
) -> Optional[ChallengeSequence]:
    """Generate failed_stay sequence with configurable evidence-per-round and interference.

    Example with n heldout rules plus 5 benchmark rules: target_sizes = [5, 1],
    corresponding to trajectories n+5 -> 5 -> 1 from no-evidence prior.

    If post_convergence_interference_rounds > 0, additional interference rounds
    are appended immediately after singleton convergence. Every interference evidence must:
    - keep survivors at singleton {oracle},
    - satisfy oracle,
    - share an exact same matched-rule profile across all interference evidences,
      and this profile is chosen to maximize matched rules.

    For evidence_per_round=4, each coarse round is represented by 4 evidence
    events and encoded in evidence_batch_ranges.
    """
    if evidence_per_round not in {1, 4}:
        raise ValueError("evidence_per_round must be 1 or 4")

    all_rules = _all_rules(heldout_set)
    for name in candidate_names:
        get_rule(name)

    if oracle not in all_rules:
        return None

    rules = {n: all_rules[n] for n in candidate_names}
    oracle_rule = rules[oracle]
    coarse_sizes = list(target_sizes) if target_sizes is not None else [5, 1]

    if not coarse_sizes or coarse_sizes[-1] != 1:
        return None

    if any(s < 1 for s in coarse_sizes):
        return None

    prev_size = len(candidate_names)
    for s in coarse_sizes:
        if s > prev_size:
            return None
        prev_size = s

    if post_convergence_interference_rounds < 0:
        return None

    events: List[Dict[str, Any]] = []
    evidence: List[Tuple[Triple, str]] = []
    step_survivors: List[Set[str]] = []
    evidence_batch_ranges: List[List[int]] = []

    for target in coarse_sizes:
        round_start = len(events)

        chosen: Optional[Tuple[Triple, str, Set[str]]] = None
        for _ in range(MAX_SAMPLE_ATTEMPTS):
            t = _random_triple(rng)
            label = "YES" if oracle_rule.validate(t) else "NO"
            new_evidence = evidence + [(t, label)]
            survivors = _compute_survivors(rules, new_evidence)
            if oracle not in survivors:
                continue
            if len(survivors) != target:
                continue
            chosen = (t, label, survivors)
            break

        if chosen is None:
            return None

        triple, _label, survivors = chosen
        events.append({"type": "evidence", "triple": triple})
        evidence.append((triple, "YES" if oracle_rule.validate(triple) else "NO"))
        step_survivors.append(set(survivors))

        if evidence_per_round == 4:
            for _ in range(3):
                extra = _find_no_elim_step(
                    rng=rng,
                    rules=rules,
                    oracle=oracle,
                    active_evidence=evidence,
                    expected_survivors=survivors,
                )
                if extra is None:
                    return None
                t_extra, r_extra = extra
                events.append({"type": "evidence", "triple": t_extra})
                evidence.append((t_extra, r_extra))
                step_survivors.append(set(survivors))
            evidence_batch_ranges.append([round_start, round_start + 3])

    used_triples: Set[Triple] = {tuple(t) for t, _label in evidence}
    interference_profile: Optional[Set[str]] = None
    interference_round_start = len(events)
    for _ in range(post_convergence_interference_rounds):
        round_start = len(events)

        if interference_profile is None:
            best = _find_best_singleton_interference_profile(
                rng=rng,
                rules=rules,
                oracle=oracle,
                active_evidence=evidence,
                max_attempts=MAX_SAMPLE_ATTEMPTS,
                forbidden_triples=used_triples,
            )
            if best is None:
                return None
            interference_profile, anchor_t = best
            events.append({"type": "evidence", "triple": anchor_t})
            evidence.append((anchor_t, "YES"))
            used_triples.add(anchor_t)
            step_survivors.append({oracle})
        else:
            step = _find_singleton_step_with_exact_profile(
                rng=rng,
                rules=rules,
                oracle=oracle,
                active_evidence=evidence,
                required_profile=interference_profile,
                forbidden_triples=used_triples,
            )
            if step is None:
                return None
            t_step, r_step = step
            events.append({"type": "evidence", "triple": t_step})
            evidence.append((t_step, r_step))
            used_triples.add(t_step)
            step_survivors.append({oracle})

        if evidence_per_round == 4:
            if interference_profile is None:
                return None
            for _ in range(3):
                extra = _find_singleton_step_with_exact_profile(
                    rng=rng,
                    rules=rules,
                    oracle=oracle,
                    active_evidence=evidence,
                    required_profile=interference_profile,
                    forbidden_triples=used_triples,
                )
                if extra is None:
                    return None
                t_extra, r_extra = extra
                events.append({"type": "evidence", "triple": t_extra})
                evidence.append((t_extra, r_extra))
                used_triples.add(t_extra)
                step_survivors.append({oracle})
            evidence_batch_ranges.append([round_start, round_start + 3])

    convergence_turn = len(coarse_sizes) * evidence_per_round - 1
    challenge_turns = [max(0, convergence_turn - evidence_per_round)]

    ground_truth: List[Dict[str, Any]] = []
    for i, (triple, label) in enumerate(evidence):
        ground_truth.append(
            {
                "turn": i,
                "triple": triple,
                "result": label,
                "survivors": sorted(step_survivors[i]),
            }
        )

    return {
        "challenge_type": "failed_stay_v2",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": challenge_turns,
        "convergence_turn": convergence_turn,
        "total_turns": len(events),
        "target_sizes": [len(s) for s in step_survivors],
        "coarse_target_sizes": coarse_sizes + [1] * post_convergence_interference_rounds,
        "prefix_survivors": sorted(step_survivors[0]) if step_survivors else [],
        "evidence_per_round": evidence_per_round,
        "post_convergence_interference_rounds": post_convergence_interference_rounds,
        "interference_rule_profile": (
            sorted(interference_profile) if interference_profile is not None else []
        ),
        "interference_rule_profile_size": (
            len(interference_profile) if interference_profile is not None else 0
        ),
        "interference_round_start_turn": (
            interference_round_start if post_convergence_interference_rounds > 0 else None
        ),
        "evidence_batch_ranges": evidence_batch_ranges,
    }
