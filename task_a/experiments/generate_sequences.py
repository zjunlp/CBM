"""Generate failed_update and failed_stay challenge sequences for Scenario A.

Key functions:
1. generate_failed_update_sequence_v2 — 3-turn misrecord+correction sequence
2. generate_failed_stay_sequence — 3-turn convergence+ambiguous sequence
3. is_effective_failed_update_data — filter for trajectories exhibiting belief failed_update
3. main — end-to-end pipeline: generate, run model, filter, save
"""

import argparse
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from task_a.core.config import (
    CHALLENGE_MISRECORD_CORRECTION,
    ChallengeSequence,
    ExperimentConfig,
)
from task_a.core.environment import RetractionEnvironment
from task_a.core.evidence_sequences import (
    compute_retraction_ground_truth,
    compute_surviving_set,
)
from task_a.core.orchestrator import GameOrchestrator
from task_a.core.rules import BENCHMARK_RULES as CORE_BENCHMARK_RULES
from task_a.core.rules import RULES, Triple, get_rule
from task_a.experiments.challenge_metrics import (
    annotate_challenge_trajectory,
    summarize_trajectory,
)
from utils.llm_backend import VLLMBackend
from utils.io import save_json

TRIPLE_RANGE = (-20, 20)
MAX_SAMPLE_ATTEMPTS = 5000


def _random_triple(rng: random.Random) -> Triple:
    lo, hi = TRIPLE_RANGE
    return (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))


def generate_random_failed_update_sequence(
    oracle: str,
    rng: random.Random,
) -> Optional[ChallengeSequence]:
    """Generate a random 3-turn failed_update sequence for a given oracle rule.

    Returns None if a valid sequence cannot be found within MAX_SAMPLE_ATTEMPTS.

    Structure (3 turns total):
    - Turn 0: 1 prefix triple that converges survivors to exactly {oracle}
    - Turn 1: misrecorded evidence (wrong YES/NO) that contradicts oracle
    - Turn 2: retraction correcting turn 1 back to the true result

    After turn 1, formal survivors are empty (since prefix already eliminated
    all non-oracle rules, and wrong result eliminates oracle). This is expected:
    the purpose is to test whether the model exhibits belief failed_update — i.e.,
    whether it correctly recovers oracle after the correction at turn 2.
    We check model behavior, not formal survivor sets.
    """
    rule = RULES[oracle]

    # Phase 1: find 1 prefix triple that is YES for oracle and NO for all others
    prefix_triple = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        if not rule.validate(t):
            continue
        survivors = compute_surviving_set([(t, "YES")])
        if survivors == {oracle}:
            prefix_triple = t
            break

    if prefix_triple is None:
        return None

    # Phase 2: pick a misrecord triple — any triple works, just flip the result
    misrecord_triple = _random_triple(rng)
    true_result = "YES" if rule.validate(misrecord_triple) else "NO"
    wrong_recorded = "NO" if true_result == "YES" else "YES"

    # Build events: 1 prefix + 1 misrecord + 1 retraction = 3 turns
    events: List[Dict] = []
    events.append({"type": "evidence", "triple": prefix_triple})
    events.append({
        "type": "evidence",
        "triple": misrecord_triple,
        "recorded_result": wrong_recorded,
    })
    events.append({
        "type": "retraction",
        "retract_turn": 1,  # retract the misrecord (event index 1)
        "new_triple": misrecord_triple,
    })

    # Compute ground truth
    gt = compute_retraction_ground_truth(oracle, events)

    return {
        "challenge_type": CHALLENGE_MISRECORD_CORRECTION,
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": [dict(step, turn=i) for i, step in enumerate(gt)],
        "challenge_turns": [2],  # retraction is at turn 2
        "convergence_turn": 0,   # correct convergence at turn 0
        "total_turns": len(events),
    }


# ---------------------------------------------------------------------------
# V2 generators: partial convergence prefix (2-3 survivors including oracle)
# ---------------------------------------------------------------------------


def _generate_partial_convergence_prefix(
    oracle: str,
    rng: random.Random,
) -> Optional[Tuple[Triple, List[Tuple[Triple, str]], set]]:
    """Find a single prefix triple that leaves 2-3 survivors including oracle.

    Returns (prefix_triple, evidence_list, survivors) or None.
    """
    rule = RULES[oracle]
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        result = "YES" if rule.validate(t) else "NO"
        survivors = compute_surviving_set([(t, result)])
        if oracle in survivors and 2 <= len(survivors) <= 3:
            return t, [(t, result)], survivors
    return None


def generate_failed_update_sequence_v2(
    oracle: str,
    rng: random.Random,
) -> Optional[ChallengeSequence]:
    """Generate a 3-turn failed_update sequence with partial convergence prefix.

    Structure:
    - Turn 0: prefix triple → survivors = {oracle, wrong1} or {oracle, w1, w2}
    - Turn 1: misrecord triple → oracle eliminated, only wrong rules survive
    - Turn 2: retract turn 1 → oracle should be restored
    """
    rule = RULES[oracle]

    # Phase 1: partial convergence prefix
    prefix_result = _generate_partial_convergence_prefix(oracle, rng)
    if prefix_result is None:
        return None
    prefix_triple, prefix_evidence, prefix_survivors = prefix_result

    # Phase 2: find a misrecord triple that eliminates oracle but keeps some wrong rules
    misrecord_triple = None
    wrong_recorded = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        true_result = "YES" if rule.validate(t) else "NO"
        flipped = "NO" if true_result == "YES" else "YES"
        # Compute survivors after prefix + misrecorded triple
        new_evidence = prefix_evidence + [(t, flipped)]
        new_survivors = compute_surviving_set(new_evidence)
        # Oracle must be eliminated, at least one wrong rule must survive
        if oracle not in new_survivors and len(new_survivors) >= 1:
            misrecord_triple = t
            wrong_recorded = flipped
            break

    if misrecord_triple is None:
        return None

    # Build events
    prefix_result_str = "YES" if rule.validate(prefix_triple) else "NO"
    events: List[Dict] = [
        {"type": "evidence", "triple": prefix_triple},
        {
            "type": "evidence",
            "triple": misrecord_triple,
            "recorded_result": wrong_recorded,
        },
        {
            "type": "retraction",
            "retract_turn": 1,
            "new_triple": misrecord_triple,
        },
    ]

    gt = compute_retraction_ground_truth(oracle, events)

    return {
        "challenge_type": CHALLENGE_MISRECORD_CORRECTION,
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": [dict(step, turn=i) for i, step in enumerate(gt)],
        "challenge_turns": [2],
        "convergence_turn": 0,
        "total_turns": 3,
        "prefix_survivors": sorted(prefix_survivors),
    }


def generate_failed_stay_sequence(
    oracle: str,
    rng: random.Random,
) -> Optional[ChallengeSequence]:
    """Generate a 3-turn failed_stay sequence with partial convergence prefix.

    Structure:
    - Turn 0: prefix triple → survivors = {oracle, wrong1} or {oracle, w1, w2}
    - Turn 1: discriminating triple → survivors narrow to {oracle}
    - Turn 2: ambiguous triple (true result consistent with oracle + others) → survivors expand
    """
    rule = RULES[oracle]

    # Phase 1: partial convergence prefix
    prefix_result = _generate_partial_convergence_prefix(oracle, rng)
    if prefix_result is None:
        return None
    prefix_triple, prefix_evidence, prefix_survivors = prefix_result

    # Phase 2: find a discriminating triple that narrows survivors to {oracle}
    disc_triple = None
    disc_evidence = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        result = "YES" if rule.validate(t) else "NO"
        new_evidence = prefix_evidence + [(t, result)]
        new_survivors = compute_surviving_set(new_evidence)
        if new_survivors == {oracle}:
            disc_triple = t
            disc_evidence = new_evidence
            break

    if disc_triple is None:
        return None

    # Phase 3: find an ambiguous triple — on this triple alone, oracle and at least
    # one other rule give the same YES/NO prediction (model sees them agree).
    # Cumulative survivors still = {oracle}, but the single-turn signal is ambiguous.
    ambig_triple = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        oracle_result = rule.validate(t)
        # Count how many other rules agree with oracle on this triple
        agreeing = [
            r for r in RULES
            if r != oracle and RULES[r].validate(t) == oracle_result
        ]
        if len(agreeing) >= 1:
            ambig_triple = t
            break

    if ambig_triple is None:
        return None

    # Build events (all true results, no misrecording)
    events: List[Dict] = [
        {"type": "evidence", "triple": prefix_triple},
        {"type": "evidence", "triple": disc_triple},
        {"type": "evidence", "triple": ambig_triple},
    ]

    gt = compute_retraction_ground_truth(oracle, events)

    return {
        "challenge_type": "failed_stay",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": [dict(step, turn=i) for i, step in enumerate(gt)],
        "challenge_turns": [2],
        "convergence_turn": 1,  # model should converge to oracle at turn 1
        "total_turns": 3,
        "prefix_survivors": sorted(prefix_survivors),
    }


def generate_belief_stats_failed_update_sequence(
    *,
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str = "easy",
    evidence_per_round: int = 1,
    n_yes_per_round: Optional[int] = None,
) -> Optional[ChallengeSequence]:
    """Generate the local belief_stats failed_update sequence.

    Benchmark-only keeps the original Scenario A generator. Benchmark+heldout
    uses the host-driven heldout generator. Both emit one evidence item per
    model turn.
    """
    if set(candidate_names) == set(CORE_BENCHMARK_RULES) and len(candidate_names) == len(CORE_BENCHMARK_RULES):
        return generate_failed_update_sequence_v2(oracle, rng)

    if evidence_per_round == 2:
        return _generate_belief_stats_failed_update_sequence_two_evidence(
            oracle=oracle,
            rng=rng,
            candidate_names=candidate_names,
            heldout_set=heldout_set,
        )
    if evidence_per_round >= 4:
        return _generate_belief_stats_failed_update_sequence_four_evidence(
            oracle=oracle,
            rng=rng,
            candidate_names=candidate_names,
            heldout_set=heldout_set,
            n_yes_per_round=n_yes_per_round,
        )

    return _generate_belief_stats_failed_update_sequence_one_evidence(
        oracle=oracle,
        rng=rng,
        candidate_names=candidate_names,
        heldout_set=heldout_set,
    )


def generate_belief_stats_failed_stay_sequence(
    *,
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str = "easy",
    post_convergence_interference_rounds: int = 1,
    evidence_per_round: int = 1,
    n_yes_per_round: Optional[int] = None,
) -> Optional[ChallengeSequence]:
    """Generate the local belief_stats failed_stay sequence.

    Benchmark-only keeps the original Scenario A generator. Benchmark+heldout
    uses the host-driven n+5 -> 5 -> 1 structure plus post-interference rounds.
    Both emit one evidence item per model turn.
    """
    if set(candidate_names) == set(CORE_BENCHMARK_RULES) and len(candidate_names) == len(CORE_BENCHMARK_RULES):
        return generate_failed_stay_sequence(oracle, rng)

    if evidence_per_round == 2:
        return _generate_belief_stats_failed_stay_sequence_two_evidence(
            oracle=oracle,
            rng=rng,
            candidate_names=candidate_names,
            heldout_set=heldout_set,
            post_convergence_interference_rounds=post_convergence_interference_rounds,
        )
    if evidence_per_round == 3:
        return _generate_belief_stats_failed_stay_sequence_three_evidence(
            oracle=oracle,
            rng=rng,
            candidate_names=candidate_names,
            heldout_set=heldout_set,
            post_convergence_interference_rounds=post_convergence_interference_rounds,
        )
    if evidence_per_round >= 4:
        return _generate_belief_stats_failed_stay_sequence_four_evidence(
            oracle=oracle,
            rng=rng,
            candidate_names=candidate_names,
            heldout_set=heldout_set,
            post_convergence_interference_rounds=post_convergence_interference_rounds,
            n_yes_per_round=n_yes_per_round,
        )

    return _generate_belief_stats_failed_stay_sequence_one_evidence(
        oracle=oracle,
        rng=rng,
        candidate_names=candidate_names,
        heldout_set=heldout_set,
        post_convergence_interference_rounds=post_convergence_interference_rounds,
    )


def _generate_belief_stats_failed_update_sequence_one_evidence(
    *,
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
) -> Optional[ChallengeSequence]:
    from task_a.experiments.host_driven_sequences import (
        _all_rules,
        _compute_retraction_ground_truth_v2,
        _compute_survivors,
        _random_triple,
    )

    all_rules = _all_rules(heldout_set)
    if oracle not in all_rules:
        return None

    rules = {name: all_rules[name] for name in candidate_names}
    events: List[Dict[str, Any]] = []
    active_evidence: List[Tuple[Triple, str]] = []
    used_triples: Set[Triple] = set()

    for target in (8, 5):
        step = _find_true_step_exact_for_belief_stats(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            target_size=target,
            forbidden_triples=used_triples,
        )
        if step is None:
            return None
        t, result, _survivors = step
        events.append({"type": "evidence", "triple": t})
        active_evidence.append((t, result))
        used_triples.add(t)

    mis = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        if t in used_triples:
            continue
        true_result = "YES" if rng.random() < 0.5 else "NO"
        if (true_result == "YES") != rules[oracle].validate(t):
            continue
        wrong_result = "NO" if true_result == "YES" else "YES"
        wrong_survivors = _compute_survivors(rules, active_evidence + [(t, wrong_result)])
        if len(wrong_survivors) != 1 or oracle in wrong_survivors:
            continue
        corrected_survivors = _compute_survivors(rules, active_evidence + [(t, true_result)])
        if len(corrected_survivors) != 4 or oracle not in corrected_survivors:
            continue
        mis = (t, wrong_result, wrong_survivors, corrected_survivors)
        break
    if mis is None:
        return None

    t_mis, wrong_mis, s1_wrong, s4_corrected = mis
    events.append({"type": "evidence", "triple": t_mis, "recorded_result": wrong_mis})
    events.append(
        {
            "type": "retraction",
            "retract_turn": 2,
            "new_triple": t_mis,
            "use_retracted_triple": True,
        }
    )

    ground_truth = _compute_retraction_ground_truth_v2(
        oracle=oracle,
        rules=rules,
        events=events,
    )
    trace = [len(step["survivors"]) for step in ground_truth]
    if trace != [8, 5, 1, 4]:
        return None

    return {
        "challenge_type": "belief_stats_failed_update_one_evidence",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [3],
        "convergence_turn": 2,
        "total_turns": len(events),
        "target_sizes": trace,
        "coarse_target_sizes": [8, 5, 1, 4],
        "wrong_survivors_before_correction": sorted(s1_wrong),
        "corrected_survivors": sorted(s4_corrected),
        "evidence_per_round": 1,
    }


def _generate_belief_stats_failed_update_sequence_two_evidence(
    *,
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
) -> Optional[ChallengeSequence]:
    from task_a.experiments.host_driven_sequences import (
        _all_rules,
        _compute_retraction_ground_truth_v2,
        _compute_survivors,
        _random_triple,
    )

    all_rules = _all_rules(heldout_set)
    if oracle not in all_rules:
        return None

    rules = {name: all_rules[name] for name in candidate_names}
    events: List[Dict[str, Any]] = []
    active_evidence: List[Tuple[Triple, str]] = []
    evidence_batch_ranges: List[List[int]] = []
    used_triples: Set[Triple] = set()

    step0 = _find_true_step_exact_for_belief_stats(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        target_size=8,
        forbidden_triples=used_triples,
    )
    if step0 is None:
        return None
    t0, r0, s8 = step0
    events.append({"type": "evidence", "triple": t0})
    active_evidence.append((t0, r0))
    used_triples.add(t0)

    no_elim0 = _find_no_elim_step_for_belief_stats(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        expected_survivors=s8,
        forbidden_triples=used_triples,
    )
    if no_elim0 is None:
        return None
    t1, r1 = no_elim0
    events.append({"type": "evidence", "triple": t1})
    active_evidence.append((t1, r1))
    used_triples.add(t1)
    evidence_batch_ranges.append([0, 1])

    no_elim1 = _find_no_elim_step_for_belief_stats(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        expected_survivors=s8,
        forbidden_triples=used_triples,
    )
    if no_elim1 is None:
        return None
    t2, r2 = no_elim1
    events.append({"type": "evidence", "triple": t2})
    active_evidence.append((t2, r2))
    used_triples.add(t2)

    step1 = _find_true_step_exact_for_belief_stats(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        target_size=5,
        forbidden_triples=used_triples,
    )
    if step1 is None:
        return None
    t3, r3, _s5 = step1
    events.append({"type": "evidence", "triple": t3})
    active_evidence.append((t3, r3))
    used_triples.add(t3)
    evidence_batch_ranges.append([2, 3])

    step2 = _find_true_step_exact_for_belief_stats(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        target_size=3,
        forbidden_triples=used_triples,
    )
    if step2 is None:
        return None
    t4, r4, _s3 = step2
    events.append({"type": "evidence", "triple": t4})
    active_evidence.append((t4, r4))
    used_triples.add(t4)

    mis = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        if t in used_triples:
            continue
        true_result = "YES" if rng.random() < 0.5 else "NO"
        if (true_result == "YES") != rules[oracle].validate(t):
            continue
        wrong_result = "NO" if true_result == "YES" else "YES"
        wrong_survivors = _compute_survivors(rules, active_evidence + [(t, wrong_result)])
        if len(wrong_survivors) != 1 or oracle in wrong_survivors:
            continue
        corrected_survivors = _compute_survivors(rules, active_evidence + [(t, true_result)])
        if len(corrected_survivors) != 2 or oracle not in corrected_survivors:
            continue
        mis = (t, wrong_result, wrong_survivors, corrected_survivors)
        break
    if mis is None:
        return None

    t_mis, wrong_mis, s1_wrong, s2_corrected = mis
    events.append({"type": "evidence", "triple": t_mis, "recorded_result": wrong_mis})
    active_evidence.append((t_mis, wrong_mis))
    used_triples.add(t_mis)
    evidence_batch_ranges.append([4, 5])
    events.append(
        {
            "type": "retraction",
            "retract_turn": 5,
            "new_triple": t_mis,
            "use_retracted_triple": True,
        }
    )

    ground_truth = _compute_retraction_ground_truth_v2(
        oracle=oracle,
        rules=rules,
        events=events,
    )
    trace = [len(step["survivors"]) for step in ground_truth]
    if trace != [8, 8, 8, 5, 3, 1, 2]:
        return None

    return {
        "challenge_type": "belief_stats_failed_update_two_evidence",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [6],
        "convergence_turn": 5,
        "total_turns": len(events),
        "target_sizes": trace,
        "coarse_target_sizes": [8, 5, 1, 2],
        "evidence_batch_ranges": evidence_batch_ranges,
        "wrong_survivors_before_correction": sorted(s1_wrong),
        "corrected_survivors": sorted(s2_corrected),
        "evidence_per_round": 2,
    }


def _generate_belief_stats_failed_update_sequence_four_evidence(
    *,
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
    n_yes_per_round: Optional[int] = None,
) -> Optional[ChallengeSequence]:
    from task_a.experiments.host_driven_sequences import (
        _all_rules,
        _compute_retraction_ground_truth_v2,
        _compute_survivors,
        _random_triple,
    )

    all_rules = _all_rules(heldout_set)
    if oracle not in all_rules:
        return None

    rules = {name: all_rules[name] for name in candidate_names}
    events: List[Dict[str, Any]] = []
    active_evidence: List[Tuple[Triple, str]] = []
    evidence_batch_ranges: List[List[int]] = []
    used_triples: Set[Triple] = set()

    def _round_results(n: int = 4) -> List[Optional[str]]:
        """Return a shuffled list of required_result values for one round."""
        if n_yes_per_round is None:
            return [None] * n
        n_yes = min(n_yes_per_round, n)
        results: List[Optional[str]] = ["YES"] * n_yes + ["NO"] * (n - n_yes)
        rng.shuffle(results)
        return results

    # Round 1: exact_8, no_elim x3  -> trace [8, 8, 8, 8]
    _rr1 = _round_results(4)
    step0 = _find_true_step_exact_for_belief_stats(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        target_size=8,
        forbidden_triples=used_triples,
        required_result=_rr1[0],
    )
    if step0 is None:
        return None
    t0, r0, s8 = step0
    events.append({"type": "evidence", "triple": t0})
    active_evidence.append((t0, r0))
    used_triples.add(t0)

    for _rr1_i, _rr1_val in enumerate(_rr1[1:], 1):
        no_elim = _find_no_elim_step_for_belief_stats(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            expected_survivors=s8,
            forbidden_triples=used_triples,
            required_result=_rr1_val,
        )
        if no_elim is None:
            return None
        t_ne, r_ne = no_elim
        events.append({"type": "evidence", "triple": t_ne})
        active_evidence.append((t_ne, r_ne))
        used_triples.add(t_ne)
    evidence_batch_ranges.append([0, 3])

    # Round 2: exact_5, no_elim x2, exact_3  -> trace [5, 5, 5, 3]
    _rr2 = _round_results(4)
    step1 = _find_true_step_exact_for_belief_stats(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        target_size=5,
        forbidden_triples=used_triples,
        required_result=_rr2[0],
    )
    if step1 is None:
        return None
    t1, r1, s5 = step1
    events.append({"type": "evidence", "triple": t1})
    active_evidence.append((t1, r1))
    used_triples.add(t1)

    for _rr2_val in _rr2[1:3]:
        no_elim = _find_no_elim_step_for_belief_stats(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence,
            expected_survivors=s5,
            forbidden_triples=used_triples,
            required_result=_rr2_val,
        )
        if no_elim is None:
            return None
        t_ne, r_ne = no_elim
        events.append({"type": "evidence", "triple": t_ne})
        active_evidence.append((t_ne, r_ne))
        used_triples.add(t_ne)

    step2 = _find_true_step_exact_for_belief_stats(
        rng=rng,
        rules=rules,
        oracle=oracle,
        active_evidence=active_evidence,
        target_size=3,
        forbidden_triples=used_triples,
        required_result=_rr2[3],
    )
    if step2 is None:
        return None
    t2, r2, _s3 = step2
    events.append({"type": "evidence", "triple": t2})
    active_evidence.append((t2, r2))
    used_triples.add(t2)
    evidence_batch_ranges.append([4, 7])

    # Round 3 (interference): all 4 evidences are misrecorded (recorded result
    # is the flip of oracle's true answer). The model-visible labels still obey
    # the n_yes_per_round constraint. Each individual misrecord must keep the
    # cumulative survivors non-empty while excluding oracle. The single
    # correction turn that follows retracts all 4 at once, restoring the
    # post-round-2 state (survivors = 3 including oracle).
    round3_start = len(events)  # expected to be 8
    post_round2_survivors = _compute_survivors(rules, active_evidence)

    mis_block = None
    for _outer in range(200):
        _rr3 = _round_results(4)
        candidate_mis: List[Tuple[Triple, str]] = []
        rolling_evidence = list(active_evidence)
        rolling_used = set(used_triples)
        failed = False
        for rr_val in _rr3:
            found = None
            for _ in range(MAX_SAMPLE_ATTEMPTS):
                t = _random_triple(rng)
                if t in rolling_used:
                    continue
                true_result = "YES" if rules[oracle].validate(t) else "NO"
                wrong_result = "NO" if true_result == "YES" else "YES"
                if rr_val is not None and wrong_result != rr_val:
                    continue
                new_survivors = _compute_survivors(
                    rules, rolling_evidence + [(t, wrong_result)]
                )
                if oracle in new_survivors:
                    continue
                if len(new_survivors) < 1:
                    continue
                found = (t, wrong_result)
                break
            if found is None:
                failed = True
                break
            candidate_mis.append(found)
            rolling_evidence.append(found)
            rolling_used.add(found[0])
        if failed:
            continue
        final_wrong_survivors = _compute_survivors(rules, rolling_evidence)
        if oracle in final_wrong_survivors or not final_wrong_survivors:
            continue
        mis_block = (candidate_mis, final_wrong_survivors)
        break

    if mis_block is None:
        return None

    mis_list, s_wrong_final = mis_block
    for (t_m, wr_m) in mis_list:
        events.append({"type": "evidence", "triple": t_m, "recorded_result": wr_m})
        active_evidence.append((t_m, wr_m))
        used_triples.add(t_m)
    evidence_batch_ranges.append([round3_start, round3_start + 3])

    # Single correction turn that retracts all 4 misrecorded turns.
    retract_turn_idxs = list(range(round3_start, round3_start + 4))
    events.append(
        {
            "type": "retraction",
            "retract_turns": retract_turn_idxs,
        }
    )

    ground_truth = _compute_retraction_ground_truth_v2(
        oracle=oracle,
        rules=rules,
        events=events,
    )
    trace = [len(step["survivors"]) for step in ground_truth]
    # Sanity checks: oracle should be eliminated at the end of round 3 and
    # restored to the post-round-2 survivor set after the multi-retraction.
    end_of_round3 = round3_start + 3
    if oracle in set(ground_truth[end_of_round3]["survivors"]):
        return None
    final_survivors = set(ground_truth[-1]["survivors"])
    if final_survivors != set(post_round2_survivors):
        return None

    coarse_target_sizes = [8, 5, len(s_wrong_final), len(final_survivors)]

    return {
        "challenge_type": "belief_stats_failed_update_four_evidence",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [round3_start + 4],
        "convergence_turn": round3_start + 3,
        "total_turns": len(events),
        "target_sizes": trace,
        "coarse_target_sizes": coarse_target_sizes,
        "evidence_batch_ranges": evidence_batch_ranges,
        "wrong_survivors_before_correction": sorted(s_wrong_final),
        "corrected_survivors": sorted(final_survivors),
        "evidence_per_round": 4,
    }


def _find_true_step_between_sizes_for_belief_stats(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    min_size: int,
    max_size: int,
    forbidden_triples: Set[Triple],
    required_result: Optional[str] = None,
) -> Optional[Tuple[Triple, str, Set[str]]]:
    from task_a.experiments.host_driven_sequences import _random_triple

    if min_size > max_size:
        return None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        if t in forbidden_triples:
            continue
        result = required_result if required_result is not None else ("YES" if rng.random() < 0.5 else "NO")
        if (result == "YES") != rules[oracle].validate(t):
            continue
        survivors = {
            name
            for name, rule in rules.items()
            if all(rule.validate(triple) == (label == "YES") for triple, label in active_evidence + [(t, result)])
        }
        if oracle in survivors and min_size <= len(survivors) <= max_size:
            return t, result, survivors
    return None


def _find_true_step_exact_for_belief_stats(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    target_size: int,
    forbidden_triples: Set[Triple],
    required_result: Optional[str] = None,
) -> Optional[Tuple[Triple, str, Set[str]]]:
    from task_a.experiments.host_driven_sequences import _random_triple

    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        if t in forbidden_triples:
            continue
        result = required_result if required_result is not None else ("YES" if rng.random() < 0.5 else "NO")
        if (result == "YES") != rules[oracle].validate(t):
            continue
        survivors = {
            name
            for name, rule in rules.items()
            if all(
                rule.validate(triple) == (label == "YES")
                for triple, label in active_evidence + [(t, result)]
            )
        }
        if oracle in survivors and len(survivors) == target_size:
            return t, result, survivors
    return None


def _find_no_elim_step_for_belief_stats(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    expected_survivors: Set[str],
    forbidden_triples: Set[Triple],
    required_result: Optional[str] = None,
) -> Optional[Tuple[Triple, str]]:
    from task_a.experiments.host_driven_sequences import _random_triple

    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        if t in forbidden_triples:
            continue
        result = required_result if required_result is not None else ("YES" if rng.random() < 0.5 else "NO")
        if (result == "YES") != rules[oracle].validate(t):
            continue
        survivors = {
            name
            for name, rule in rules.items()
            if all(
                rule.validate(triple) == (label == "YES")
                for triple, label in active_evidence + [(t, result)]
            )
        }
        if survivors == expected_survivors:
            return t, result
    return None


def _find_singleton_post_step_for_belief_stats(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    forbidden_triples: Set[Triple],
    oracle_value: Optional[bool] = None,
) -> Optional[Tuple[Triple, str]]:
    from task_a.experiments.host_driven_sequences import _random_triple

    for _ in range(MAX_SAMPLE_ATTEMPTS):
        t = _random_triple(rng)
        if t in forbidden_triples:
            continue
        oracle_matches = rules[oracle].validate(t)
        if oracle_value is not None and oracle_matches != oracle_value:
            continue
        result = "YES" if oracle_matches else "NO"
        survivors = {
            name
            for name, rule in rules.items()
            if all(
                rule.validate(triple) == (label == "YES")
                for triple, label in active_evidence + [(t, result)]
            )
        }
        if survivors == {oracle}:
            return t, result
    return None


def _sample_belief_stats_middle_target(
    *,
    rng: random.Random,
    n_candidates: int,
) -> int:
    if n_candidates <= 2:
        raise ValueError("Need at least 3 candidates to sample middle target")
    return rng.randint(2, n_candidates - 1)


def _generate_belief_stats_failed_stay_sequence_one_evidence(
    *,
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
    post_convergence_interference_rounds: int,
) -> Optional[ChallengeSequence]:
    from task_a.experiments.host_driven_sequences import _all_rules

    all_rules = _all_rules(heldout_set)
    if oracle not in all_rules:
        return None

    rules = {name: all_rules[name] for name in candidate_names}
    n_candidates = len(candidate_names)
    middle_target = _sample_belief_stats_middle_target(rng=rng, n_candidates=n_candidates)

    events: List[Dict[str, Any]] = []
    evidence: List[Tuple[Triple, str]] = []
    step_survivors: List[Set[str]] = []
    used_triples: Set[Triple] = set()

    for target in [middle_target, 1]:
        step = _find_true_step_exact_for_belief_stats(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=evidence,
            target_size=target,
            forbidden_triples=used_triples,
        )
        if step is None:
            return None
        t, result, survivors = step
        events.append({"type": "evidence", "triple": t})
        evidence.append((t, result))
        used_triples.add(t)
        step_survivors.append(set(survivors))

    interference_round_start = len(events)
    for _ in range(post_convergence_interference_rounds):
        step = _find_singleton_post_step_for_belief_stats(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=evidence,
            forbidden_triples=used_triples,
        )
        if step is None:
            return None
        t_post, r_post = step
        events.append({"type": "evidence", "triple": t_post})
        evidence.append((t_post, r_post))
        used_triples.add(t_post)
        step_survivors.append({oracle})

    ground_truth = [
        {
            "turn": i,
            "triple": triple,
            "result": label,
            "survivors": sorted(step_survivors[i]),
        }
        for i, (triple, label) in enumerate(evidence)
    ]

    return {
        "challenge_type": "belief_stats_failed_stay_one_evidence",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [0],
        "convergence_turn": 1,
        "total_turns": len(events),
        "target_sizes": [len(s) for s in step_survivors],
        "coarse_target_sizes": [middle_target, 1] + [1] * post_convergence_interference_rounds,
        "middle_target": middle_target,
        "prefix_survivors": sorted(step_survivors[0]) if step_survivors else [],
        "evidence_per_round": 1,
        "post_convergence_interference_rounds": post_convergence_interference_rounds,
        "interference_round_start_turn": (
            interference_round_start if post_convergence_interference_rounds > 0 else None
        ),
        "evidence_batch_ranges": [],
    }


def _generate_belief_stats_failed_stay_sequence_two_evidence(
    *,
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
    post_convergence_interference_rounds: int,
) -> Optional[ChallengeSequence]:
    from task_a.experiments.host_driven_sequences import (
        _all_rules,
        _compute_survivors,
    )

    all_rules = _all_rules(heldout_set)
    if oracle not in all_rules:
        return None

    rules = {name: all_rules[name] for name in candidate_names}
    events: List[Dict[str, Any]] = []
    evidence: List[Tuple[Triple, str]] = []
    step_survivors: List[Set[str]] = []
    evidence_batch_ranges: List[List[int]] = []
    used_triples: Set[Triple] = set()

    n_candidates = len(candidate_names)
    middle_target = _sample_belief_stats_middle_target(
        rng=rng,
        n_candidates=n_candidates,
    )

    for target in [middle_target, 1]:
        current_survivors = _compute_survivors(rules, evidence)
        round_start = len(events)
        intermediate = _find_true_step_between_sizes_for_belief_stats(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=evidence,
            min_size=target + 1,
            max_size=len(current_survivors) - 1,
            forbidden_triples=used_triples,
        )
        if intermediate is None:
            no_elim = _find_no_elim_step_for_belief_stats(
                rng=rng,
                rules=rules,
                oracle=oracle,
                active_evidence=evidence,
                expected_survivors=current_survivors,
                forbidden_triples=used_triples,
            )
            if no_elim is None:
                return None
            t_mid, r_mid = no_elim
            s_mid = set(current_survivors)
        else:
            t_mid, r_mid, s_mid = intermediate
        events.append({"type": "evidence", "triple": t_mid})
        evidence.append((t_mid, r_mid))
        used_triples.add(t_mid)
        step_survivors.append(set(s_mid))

        final_step = _find_true_step_exact_for_belief_stats(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=evidence,
            target_size=target,
            forbidden_triples=used_triples,
        )
        if final_step is None:
            return None
        t_final, r_final, s_final = final_step
        if t_final in used_triples:
            return None
        events.append({"type": "evidence", "triple": t_final})
        evidence.append((t_final, r_final))
        used_triples.add(t_final)
        step_survivors.append(set(s_final))
        evidence_batch_ranges.append([round_start, round_start + 1])

    interference_round_start = len(events)
    for _ in range(post_convergence_interference_rounds):
        round_start = len(events)
        for oracle_value in (True, False):
            step = _find_singleton_post_step_for_belief_stats(
                rng=rng,
                rules=rules,
                oracle=oracle,
                active_evidence=evidence,
                forbidden_triples=used_triples,
                oracle_value=oracle_value,
            )
            if step is None:
                return None
            t_post, r_post = step
            events.append({"type": "evidence", "triple": t_post})
            evidence.append((t_post, r_post))
            used_triples.add(t_post)
            step_survivors.append({oracle})
        evidence_batch_ranges.append([round_start, round_start + 1])

    ground_truth = [
        {
            "turn": i,
            "triple": triple,
            "result": label,
            "survivors": sorted(step_survivors[i]),
        }
        for i, (triple, label) in enumerate(evidence)
    ]

    return {
        "challenge_type": "belief_stats_failed_stay_two_evidence",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [2],
        "convergence_turn": 3,
        "total_turns": len(events),
        "target_sizes": [len(s) for s in step_survivors],
        "coarse_target_sizes": [middle_target, 1] + [1] * post_convergence_interference_rounds,
        "middle_target": middle_target,
        "prefix_survivors": sorted(step_survivors[0]) if step_survivors else [],
        "evidence_per_round": 2,
        "post_convergence_interference_rounds": post_convergence_interference_rounds,
        "interference_rule_profile": [],
        "interference_rule_profile_size": 0,
        "interference_round_start_turn": (
            interference_round_start if post_convergence_interference_rounds > 0 else None
        ),
        "evidence_batch_ranges": evidence_batch_ranges,
    }


def _generate_belief_stats_failed_stay_sequence_three_evidence(
    *,
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
    post_convergence_interference_rounds: int,
) -> Optional[ChallengeSequence]:
    from task_a.experiments.host_driven_sequences import (
        _all_rules,
        _compute_survivors,
    )

    all_rules = _all_rules(heldout_set)
    if oracle not in all_rules:
        return None

    rules = {name: all_rules[name] for name in candidate_names}
    events: List[Dict[str, Any]] = []
    evidence: List[Tuple[Triple, str]] = []
    step_survivors: List[Set[str]] = []
    evidence_batch_ranges: List[List[int]] = []
    used_triples: Set[Triple] = set()

    n_candidates = len(candidate_names)
    middle_target = _sample_belief_stats_middle_target(
        rng=rng,
        n_candidates=n_candidates,
    )

    for target in [middle_target, 1]:
        round_start = len(events)
        for _ in range(2):
            current_survivors = _compute_survivors(rules, evidence)
            intermediate = _find_true_step_between_sizes_for_belief_stats(
                rng=rng,
                rules=rules,
                oracle=oracle,
                active_evidence=evidence,
                min_size=target + 1,
                max_size=len(current_survivors) - 1,
                forbidden_triples=used_triples,
            )
            if intermediate is None:
                no_elim = _find_no_elim_step_for_belief_stats(
                    rng=rng,
                    rules=rules,
                    oracle=oracle,
                    active_evidence=evidence,
                    expected_survivors=current_survivors,
                    forbidden_triples=used_triples,
                )
                if no_elim is None:
                    return None
                t_mid, r_mid = no_elim
                s_mid = set(current_survivors)
            else:
                t_mid, r_mid, s_mid = intermediate
            events.append({"type": "evidence", "triple": t_mid})
            evidence.append((t_mid, r_mid))
            used_triples.add(t_mid)
            step_survivors.append(set(s_mid))

        final_step = _find_true_step_exact_for_belief_stats(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=evidence,
            target_size=target,
            forbidden_triples=used_triples,
        )
        if final_step is None:
            return None
        t_final, r_final, s_final = final_step
        if t_final in used_triples:
            return None
        events.append({"type": "evidence", "triple": t_final})
        evidence.append((t_final, r_final))
        used_triples.add(t_final)
        step_survivors.append(set(s_final))
        evidence_batch_ranges.append([round_start, round_start + 2])

    interference_round_start = len(events)
    for _ in range(post_convergence_interference_rounds):
        round_start = len(events)
        for oracle_value in (True, False, None):
            step = _find_singleton_post_step_for_belief_stats(
                rng=rng,
                rules=rules,
                oracle=oracle,
                active_evidence=evidence,
                forbidden_triples=used_triples,
                oracle_value=oracle_value,
            )
            if step is None:
                return None
            t_post, r_post = step
            events.append({"type": "evidence", "triple": t_post})
            evidence.append((t_post, r_post))
            used_triples.add(t_post)
            step_survivors.append({oracle})
        evidence_batch_ranges.append([round_start, round_start + 2])

    ground_truth = [
        {
            "turn": i,
            "triple": triple,
            "result": label,
            "survivors": sorted(step_survivors[i]),
        }
        for i, (triple, label) in enumerate(evidence)
    ]

    return {
        "challenge_type": "belief_stats_failed_stay_three_evidence",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [2],
        "convergence_turn": 5,
        "total_turns": len(events),
        "target_sizes": [len(s) for s in step_survivors],
        "coarse_target_sizes": [middle_target, 1] + [1] * post_convergence_interference_rounds,
        "middle_target": middle_target,
        "prefix_survivors": sorted(step_survivors[0]) if step_survivors else [],
        "evidence_per_round": 3,
        "post_convergence_interference_rounds": post_convergence_interference_rounds,
        "interference_rule_profile": [],
        "interference_rule_profile_size": 0,
        "interference_round_start_turn": (
            interference_round_start if post_convergence_interference_rounds > 0 else None
        ),
        "evidence_batch_ranges": evidence_batch_ranges,
    }



def _generate_belief_stats_failed_stay_sequence_four_evidence(
    *,
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
    post_convergence_interference_rounds: int,
    n_yes_per_round: Optional[int] = None,
) -> Optional[ChallengeSequence]:
    from task_a.experiments.host_driven_sequences import (
        _all_rules,
        _compute_survivors,
    )

    all_rules = _all_rules(heldout_set)
    if oracle not in all_rules:
        return None

    rules = {name: all_rules[name] for name in candidate_names}
    events: List[Dict[str, Any]] = []
    evidence: List[Tuple[Triple, str]] = []
    step_survivors: List[Set[str]] = []
    evidence_batch_ranges: List[List[int]] = []
    used_triples: Set[Triple] = set()

    def _round_results(n: int = 4) -> List[Optional[str]]:
        """Return a shuffled list of required_result values for one round."""
        if n_yes_per_round is None:
            return [None] * n
        n_yes = min(n_yes_per_round, n)
        results: List[Optional[str]] = ["YES"] * n_yes + ["NO"] * (n - n_yes)
        rng.shuffle(results)
        return results

    n_candidates = len(candidate_names)
    middle_target = _sample_belief_stats_middle_target(
        rng=rng,
        n_candidates=n_candidates,
    )

    for target in [middle_target, 1]:
        round_start = len(events)
        _rr = _round_results(4)
        for _rr_i in range(3):
            current_survivors = _compute_survivors(rules, evidence)
            intermediate = _find_true_step_between_sizes_for_belief_stats(
                rng=rng,
                rules=rules,
                oracle=oracle,
                active_evidence=evidence,
                min_size=target + 1,
                max_size=len(current_survivors) - 1,
                forbidden_triples=used_triples,
                required_result=_rr[_rr_i],
            )
            if intermediate is None:
                no_elim = _find_no_elim_step_for_belief_stats(
                    rng=rng,
                    rules=rules,
                    oracle=oracle,
                    active_evidence=evidence,
                    expected_survivors=current_survivors,
                    forbidden_triples=used_triples,
                    required_result=_rr[_rr_i],
                )
                if no_elim is None:
                    return None
                t_mid, r_mid = no_elim
                s_mid = set(current_survivors)
            else:
                t_mid, r_mid, s_mid = intermediate
            events.append({"type": "evidence", "triple": t_mid})
            evidence.append((t_mid, r_mid))
            used_triples.add(t_mid)
            step_survivors.append(set(s_mid))

        final_step = _find_true_step_exact_for_belief_stats(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=evidence,
            target_size=target,
            forbidden_triples=used_triples,
            required_result=_rr[3],
        )
        if final_step is None:
            return None
        t_final, r_final, s_final = final_step
        if t_final in used_triples:
            return None
        events.append({"type": "evidence", "triple": t_final})
        evidence.append((t_final, r_final))
        used_triples.add(t_final)
        step_survivors.append(set(s_final))
        evidence_batch_ranges.append([round_start, round_start + 3])

    interference_round_start = len(events)
    for _ in range(post_convergence_interference_rounds):
        round_start = len(events)
        # For post-convergence rounds: if n_yes_per_round controls YES/NO,
        # use it; otherwise fall back to original (True, False, None, None).
        if n_yes_per_round is not None:
            _post_rr = _round_results(4)
            post_oracle_values = [
                (True if r == "YES" else False) if r is not None else None
                for r in _post_rr
            ]
        else:
            post_oracle_values = [True, False, None, None]
        for oracle_value in post_oracle_values:
            step = _find_singleton_post_step_for_belief_stats(
                rng=rng,
                rules=rules,
                oracle=oracle,
                active_evidence=evidence,
                forbidden_triples=used_triples,
                oracle_value=oracle_value,
            )
            if step is None:
                return None
            t_post, r_post = step
            events.append({"type": "evidence", "triple": t_post})
            evidence.append((t_post, r_post))
            used_triples.add(t_post)
            step_survivors.append({oracle})
        evidence_batch_ranges.append([round_start, round_start + 3])

    ground_truth = [
        {
            "turn": i,
            "triple": triple,
            "result": label,
            "survivors": sorted(step_survivors[i]),
        }
        for i, (triple, label) in enumerate(evidence)
    ]

    return {
        "challenge_type": "belief_stats_failed_stay_four_evidence",
        "oracle": oracle,
        "events": events,
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [2],
        "convergence_turn": 7,
        "total_turns": len(events),
        "target_sizes": [len(s) for s in step_survivors],
        "coarse_target_sizes": [middle_target, 1] + [1] * post_convergence_interference_rounds,
        "middle_target": middle_target,
        "prefix_survivors": sorted(step_survivors[0]) if step_survivors else [],
        "evidence_per_round": 4,
        "post_convergence_interference_rounds": post_convergence_interference_rounds,
        "interference_rule_profile": [],
        "interference_rule_profile_size": 0,
        "interference_round_start_turn": (
            interference_round_start if post_convergence_interference_rounds > 0 else None
        ),
        "evidence_batch_ranges": evidence_batch_ranges,
    }


def is_effective_failed_update_data(
    trajectory: Dict[str, Any],
    challenge_sequence: ChallengeSequence,
) -> bool:
    """Check if a trajectory exhibits belief failed_update.

    Effective = model converged to oracle at turn 2 (correct convergence)
                BUT after turn 4 correction, model did NOT return to oracle.

    We check model behavior directly since formal survivor sets are empty
    at the misrecord turn (all rules formally eliminated).
    """
    oracle = challenge_sequence["oracle"]
    turns = trajectory.get("turns", [])
    convergence_turn = challenge_sequence["convergence_turn"]  # turn 2
    challenge_turn = challenge_sequence["challenge_turns"][0]   # turn 4

    # Condition 1: model converged to {oracle} at convergence_turn
    if len(turns) <= convergence_turn:
        return False
    hyps_at_conv = set(turns[convergence_turn].get("hypotheses") or [])
    if hyps_at_conv != {oracle}:
        return False

    # Condition 2: after correction (challenge_turn), model did NOT have {oracle}
    if len(turns) <= challenge_turn:
        return False
    hyps_at_challenge = set(turns[challenge_turn].get("hypotheses") or [])
    if hyps_at_challenge == {oracle}:
        return False  # model recovered correctly — not an failed_update case

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate failed_update training data"
    )
    parser.add_argument(
        "--agent-model-path", type=str,
        default=os.environ.get("AGENT_MODEL_PATH", "models/Qwen3-30B-A3B-Instruct-2507"),
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=2)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-attempts-multiplier", type=int, default=20,
                        help="Max attempts = num_effective * this multiplier")
    parser.add_argument("--num-effective", type=int, default=200,
                        help="Target effective data count per rule")
    parser.add_argument("--rules", nargs="+", default=CORE_BENCHMARK_RULES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-temperature", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "task_a/outputs",
        f"failed_update_data_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading vLLM: {args.agent_model_path} (tp={args.vllm_tensor_parallel_size})")
    backend = VLLMBackend(
        model_path=args.agent_model_path,
        dtype="bf16",
        max_model_len=4096,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    master_rng = random.Random(args.seed)
    all_summaries: List[Dict[str, Any]] = []

    for rule_name in args.rules:
        rule_obj = get_rule(rule_name)
        print(f"\n{'='*60}")
        print(f"Rule: {rule_name}  (target: {args.num_effective} effective)")
        print(f"{'='*60}")

        effective_trajectories: List[Dict[str, Any]] = []
        idx = 0
        max_attempts = args.num_effective * args.max_attempts_multiplier

        while len(effective_trajectories) < args.num_effective and idx < max_attempts:
            seq = generate_random_failed_update_sequence(rule_name, master_rng)
            if seq is None:
                idx += 1
                continue

            exp_id = f"failed_update_gen_{rule_name}_{idx}"
            cfg = ExperimentConfig(
                experiment_id=exp_id,
                rule_name=rule_name,
                max_turns=seq["total_turns"],
                seed=args.seed + idx,
                agent_model=os.path.basename(args.agent_model_path),
                agent_temperature=args.agent_temperature,
                agent_max_tokens=512,
                output_dir=output_dir,
            )

            env = RetractionEnvironment(rule_obj, events=seq["events"])
            orchestrator = GameOrchestrator(
                cfg, backend, env,
                label_mode="elimination",
                include_evidence_table=False,
                include_rule_predictions=True,
            )
            trajectory = orchestrator.run()

            annotation = annotate_challenge_trajectory(trajectory, seq)
            trajectory["challenge_annotation"] = annotation
            trajectory["challenge_sequence"] = seq

            is_effective = is_effective_failed_update_data(trajectory, seq)

            n_eff = len(effective_trajectories)
            eff = "EFFECTIVE" if is_effective else ""
            print(f"  [{n_eff}/{args.num_effective}] {exp_id}: {eff}")

            if is_effective:
                effective_trajectories.append(trajectory)
                save_json(
                    os.path.join(output_dir, f"{exp_id}.json"),
                    trajectory,
                )

            summary = summarize_trajectory(
                trajectory, annotation, CHALLENGE_MISRECORD_CORRECTION, args.seed + idx,
            )
            summary["is_effective"] = is_effective
            all_summaries.append(summary)
            idx += 1

        print(f"  Collected {len(effective_trajectories)}/{args.num_effective} "
              f"effective trajectories ({idx} total runs)")

    # Save overall report
    save_json(os.path.join(output_dir, "all_summaries.json"), all_summaries)

    effective_count = sum(1 for s in all_summaries if s.get("is_effective"))
    total_count = len(all_summaries)
    print(f"\n{'='*60}")
    print(f"TOTAL: {effective_count}/{total_count} effective trajectories")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
