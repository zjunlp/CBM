"""Generate an external auto-sampled failed_isolation template bank for Scenario B.

FailedIsolation templates are failed_stay-compatible payloads with this structure:
- t0/t1/t2: measurements that converge to a singleton oracle by turn 2
- optional t3..: post-convergence truthful rounds that remain consistent with oracle
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from task_b.domain.rule_engine import get_topology
from task_b.experiments.belief_stats import FailedStayChallengeSequence, FailedStayTemplateCase
from task_b.templates.sampler import MeasurementCandidate, enumerate_measurement_candidates
from utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate verified failed_isolation templates for Scenario B"
    )
    parser.add_argument("--circuit-type", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--faults",
        nargs="+",
        default=None,
        help="Fault IDs to generate. Defaults to the selected topology fault space.",
    )
    parser.add_argument(
        "--max-cases-per-fault",
        type=int,
        default=0,
        help="0 means no truncation.",
    )
    parser.add_argument("--max-powered-conditions", type=int, default=3)
    parser.add_argument(
        "--include-isolation-healthy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--post-truthful-rounds",
        type=int,
        default=2,
        help="Number of truthful rounds appended after t2 convergence. Use 0 for a three-turn bank.",
    )
    parser.add_argument(
        "--min-prefix-survivors",
        type=int,
        default=2,
        help="Minimum survivor count for t0/t1 local measurements.",
    )
    parser.add_argument(
        "--max-prefix-survivors",
        type=int,
        default=6,
        help="Maximum survivor count for t0/t1/t2 local measurements.",
    )
    parser.add_argument(
        "--max-pair-survivors",
        type=int,
        default=4,
        help="Maximum survivor count after intersecting t0 and t1.",
    )
    return parser.parse_args()


def _case_score(
    t0: MeasurementCandidate,
    t1: MeasurementCandidate,
    t2: MeasurementCandidate,
    pair_survivors: Set[str],
) -> Tuple[Any, ...]:
    """Prefer compact prefixes that converge exactly at t2."""
    return (
        abs(len(pair_survivors) - 2),
        t0.condition_count + t1.condition_count + t2.condition_count,
        len(t0.survivors),
        len(t1.survivors),
        len(t2.survivors),
        t0.signature,
        t1.signature,
        t2.signature,
    )


def _pick_truthful_post_rounds(
    *,
    candidates: Sequence[MeasurementCandidate],
    oracle: str,
    used_signatures: Set[Tuple[Tuple[str, str], ...]],
    rounds: int,
) -> List[MeasurementCandidate]:
    """Pick post rounds that are consistent with oracle and diversify keys."""
    if rounds <= 0:
        return []

    pool = [
        candidate
        for candidate in candidates
        if oracle in candidate.survivors and candidate.signature not in used_signatures
    ]
    if len(pool) < rounds:
        return []

    used_keys: Set[str] = {
        key
        for signature in used_signatures
        for key, _ in signature
    }

    selected: List[MeasurementCandidate] = []
    selected_signatures = set(used_signatures)
    while len(selected) < rounds:
        remaining = [c for c in pool if c.signature not in selected_signatures]
        if not remaining:
            break

        remaining.sort(
            key=lambda c: (
                len({key for key, _ in c.measurements} & used_keys),
                c.condition_count,
                len(c.survivors),
                c.signature,
            )
        )
        chosen = remaining[0]
        selected.append(chosen)
        selected_signatures.add(chosen.signature)
        used_keys |= {key for key, _ in chosen.measurements}

    return selected if len(selected) == rounds else []


def enumerate_failed_isolation_template_cases(
    circuit_type: str,
    oracle: str,
    *,
    max_powered_conditions: int,
    include_isolation_healthy: bool,
    post_truthful_rounds: int,
    min_prefix_survivors: int,
    max_prefix_survivors: int,
    max_pair_survivors: int,
) -> List[FailedStayTemplateCase]:
    candidates = enumerate_measurement_candidates(
        circuit_type,
        max_powered_conditions=max_powered_conditions,
        include_isolation_healthy=include_isolation_healthy,
    )
    powered = [candidate for candidate in candidates if candidate.mode == "powered"]
    oracle_powered = [candidate for candidate in powered if oracle in candidate.survivors]

    cases: List[FailedStayTemplateCase] = []
    seen_signatures = set()

    for t0 in oracle_powered:
        s0 = set(t0.survivors)
        if not (min_prefix_survivors <= len(s0) <= max_prefix_survivors):
            continue

        for t1 in oracle_powered:
            if t1.signature == t0.signature:
                continue
            s1 = set(t1.survivors)
            if not (min_prefix_survivors <= len(s1) <= max_prefix_survivors):
                continue

            s01 = s0 & s1
            if oracle not in s01:
                continue
            # Keep some ambiguity before t2, then force convergence at t2.
            if len(s01) <= 1 or len(s01) > max_pair_survivors:
                continue

            for t2 in oracle_powered:
                if t2.signature in {t0.signature, t1.signature}:
                    continue
                s2 = set(t2.survivors)
                if len(s2) > max_prefix_survivors:
                    continue

                s012 = s01 & s2
                if s012 != {oracle}:
                    continue

                used_signatures = {t0.signature, t1.signature, t2.signature}
                post_rounds = _pick_truthful_post_rounds(
                    candidates=oracle_powered,
                    oracle=oracle,
                    used_signatures=used_signatures,
                    rounds=post_truthful_rounds,
                )
                if len(post_rounds) != post_truthful_rounds:
                    continue

                case = FailedStayTemplateCase(
                    oracle=oracle,
                    t0=t0,
                    t1=t1,
                    t2=t2,
                    score=_case_score(t0, t1, t2, s01),
                    post_interference=tuple(post_rounds),
                    post_interference_batches=tuple((candidate,) for candidate in post_rounds),
                    direct_converge=False,
                    failed_stay_eval_start_turn=3,
                )
                if case.signature in seen_signatures:
                    continue
                seen_signatures.add(case.signature)
                cases.append(case)

    cases.sort(key=lambda case: case.score)
    return cases


def _verify_failed_isolation_cases(
    circuit_type: str,
    fault: str,
    cases: Sequence[FailedStayTemplateCase],
    post_truthful_rounds: int,
) -> None:
    for index, case in enumerate(cases):
        challenge = FailedStayChallengeSequence(
            circuit_type,
            fault,
            index,
            case,
            prefix_rule_prediction_turns=0,
        )
        gt = challenge.ground_truth
        if len(gt) < 3 + post_truthful_rounds:
            raise ValueError(
                f"{circuit_type}:{fault}:tpl{index} has insufficient turns ({len(gt)})"
            )

        t2_survivors = set(gt[2]["survivors"])
        if t2_survivors != {fault}:
            raise ValueError(
                f"{circuit_type}:{fault}:tpl{index} not converged at t2: {sorted(t2_survivors)}"
            )

        for post_turn in range(3, 3 + post_truthful_rounds):
            survivors = set(gt[post_turn]["survivors"])
            if survivors != {fault}:
                raise ValueError(
                    f"{circuit_type}:{fault}:tpl{index} post turn t{post_turn} not truthful: "
                    f"{sorted(survivors)}"
                )


def _case_payloads(cases: Sequence[FailedStayTemplateCase]) -> List[Dict[str, Any]]:
    return [case.as_template_case() for case in cases]


def main() -> None:
    args = parse_args()
    if args.max_powered_conditions <= 0:
        raise ValueError("--max-powered-conditions must be >= 1")
    if args.max_cases_per_fault < 0:
        raise ValueError("--max-cases-per-fault must be >= 0")
    if args.post_truthful_rounds < 0:
        raise ValueError("--post-truthful-rounds must be >= 0")
    if args.min_prefix_survivors < 2:
        raise ValueError("--min-prefix-survivors must be >= 2")
    if args.max_prefix_survivors < args.min_prefix_survivors:
        raise ValueError("--max-prefix-survivors must be >= --min-prefix-survivors")
    if args.max_pair_survivors < 2:
        raise ValueError("--max-pair-survivors must be >= 2")

    topology = get_topology(args.circuit_type)
    faults = list(args.faults) if args.faults else list(topology.fault_ids)
    invalid_faults = [fault for fault in faults if fault not in topology.fault_ids]
    if invalid_faults:
        raise ValueError(f"Unknown faults for {args.circuit_type}: {invalid_faults}")

    selected_faults = set(faults)
    templates: Dict[str, List[Mapping[str, Any]]] = {
        fault: [] for fault in topology.fault_ids
    }
    counts: Dict[str, int] = {fault: 0 for fault in topology.fault_ids}

    for fault in topology.fault_ids:
        if fault not in selected_faults:
            continue

        cases = enumerate_failed_isolation_template_cases(
            args.circuit_type,
            fault,
            max_powered_conditions=args.max_powered_conditions,
            include_isolation_healthy=args.include_isolation_healthy,
            post_truthful_rounds=args.post_truthful_rounds,
            min_prefix_survivors=args.min_prefix_survivors,
            max_prefix_survivors=args.max_prefix_survivors,
            max_pair_survivors=args.max_pair_survivors,
        )
        if args.max_cases_per_fault:
            cases = cases[: args.max_cases_per_fault]

        _verify_failed_isolation_cases(
            args.circuit_type,
            fault,
            cases,
            post_truthful_rounds=args.post_truthful_rounds,
        )

        templates[fault] = _case_payloads(cases)
        counts[fault] = len(cases)
        print(f"{fault}: {len(cases)}")

    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generator": "task_b.experiments.generate_failed_isolation_template_bank",
            "circuit_type": args.circuit_type,
            "challenge_family": "belief_failed_isolation_measurement",
            "verified": True,
            "search_config": {
                "faults": faults,
                "max_cases_per_fault": args.max_cases_per_fault,
                "max_powered_conditions": args.max_powered_conditions,
                "include_isolation_healthy": args.include_isolation_healthy,
                "post_truthful_rounds": args.post_truthful_rounds,
                "min_prefix_survivors": args.min_prefix_survivors,
                "max_prefix_survivors": args.max_prefix_survivors,
                "max_pair_survivors": args.max_pair_survivors,
            },
            "case_counts": counts,
            "total_cases": sum(counts.values()),
        },
        "templates": templates,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_json(args.output, payload)

    print(f"Total cases: {payload['metadata']['total_cases']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
