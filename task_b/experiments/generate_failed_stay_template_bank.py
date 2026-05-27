"""Generate an external auto-sampled failed_stay template bank for Scenario B."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence

from task_b.domain.rule_engine import get_topology
from task_b.experiments.belief_stats import (
    FailedStayChallengeSequence,
    _add_post_convergence_interference,
    enumerate_failed_stay_template_cases,
)
from task_b.templates.sampler import enumerate_measurement_candidates
from utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate verified failed_stay templates for Scenario B"
    )
    parser.add_argument("--circuit-type", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--faults",
        nargs="+",
        default=None,
        help="Fault IDs to generate. Defaults to the selected topology fault space.",
    )
    parser.add_argument("--max-cases-per-fault", type=int, default=0)
    parser.add_argument("--max-powered-conditions", type=int, default=3)
    parser.add_argument(
        "--include-isolation-healthy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--failed_stay-case-strategy",
        type=str,
        default="multi_candidate_set_failed_stay_v9",
        choices=["multi_candidate_set_failed_stay_v9", "singleton_converge_broad_v1"],
    )
    parser.add_argument("--failed_stay-interference-rounds", type=int, default=1)
    parser.add_argument("--failed_stay-blocks-per-post-turn", type=int, default=1)
    return parser.parse_args()


def _verify_failed_stay_cases(
    circuit_type: str,
    fault: str,
    cases: Sequence[Any],
) -> None:
    for index, case in enumerate(cases):
        FailedStayChallengeSequence(
            circuit_type,
            fault,
            index,
            case,
            prefix_rule_prediction_turns=0,
        )


def _case_payloads(cases: Sequence[Any]) -> List[Dict[str, Any]]:
    return [case.as_template_case() for case in cases]


def main() -> None:
    args = parse_args()
    if args.max_powered_conditions <= 0:
        raise ValueError("--max-powered-conditions must be >= 1")
    if args.max_cases_per_fault < 0:
        raise ValueError("--max-cases-per-fault must be >= 0")
    if args.failed_stay_interference_rounds < 0:
        raise ValueError("--failed_stay-interference-rounds must be >= 0")
    if args.failed_stay_blocks_per_post_turn <= 0:
        raise ValueError("--failed_stay-blocks-per-post-turn must be >= 1")

    topology = get_topology(args.circuit_type)
    faults = list(args.faults) if args.faults else list(topology.fault_ids)
    invalid_faults = [fault for fault in faults if fault not in topology.fault_ids]
    if invalid_faults:
        raise ValueError(f"Unknown faults for {args.circuit_type}: {invalid_faults}")

    extra_candidates = [
        candidate
        for candidate in enumerate_measurement_candidates(
            args.circuit_type,
            max_powered_conditions=args.max_powered_conditions,
            include_isolation_healthy=args.include_isolation_healthy,
        )
        if candidate.mode == "powered"
    ]

    selected_faults = set(faults)
    templates: Dict[str, List[Mapping[str, Any]]] = {
        fault: [] for fault in topology.fault_ids
    }
    counts: Dict[str, int] = {fault: 0 for fault in topology.fault_ids}
    for fault in topology.fault_ids:
        if fault not in selected_faults:
            continue
        cases = enumerate_failed_stay_template_cases(
            args.circuit_type,
            fault,
            max_powered_conditions=args.max_powered_conditions,
            include_isolation_healthy=args.include_isolation_healthy,
            strategy=args.failed_stay_case_strategy,
        )
        if args.max_cases_per_fault:
            cases = cases[: args.max_cases_per_fault]
        cases = _add_post_convergence_interference(
            cases,
            cases,
            rounds=args.failed_stay_interference_rounds,
            blocks_per_turn=args.failed_stay_blocks_per_post_turn,
            coherent_distractor=True,
            prefer_broad_distractor=True,
            extra_candidates=extra_candidates,
        )
        _verify_failed_stay_cases(args.circuit_type, fault, cases)
        templates[fault] = _case_payloads(cases)
        counts[fault] = len(cases)
        print(f"{fault}: {len(cases)}")

    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generator": "task_b.experiments.generate_failed_stay_template_bank",
            "circuit_type": args.circuit_type,
            "challenge_family": "belief_failed_stay_measurement",
            "verified": True,
            "search_config": {
                "faults": faults,
                "max_cases_per_fault": args.max_cases_per_fault,
                "max_powered_conditions": args.max_powered_conditions,
                "include_isolation_healthy": args.include_isolation_healthy,
                "failed_stay_case_strategy": args.failed_stay_case_strategy,
                "failed_stay_interference_rounds": args.failed_stay_interference_rounds,
                "failed_stay_blocks_per_post_turn": args.failed_stay_blocks_per_post_turn,
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
