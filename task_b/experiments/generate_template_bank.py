"""Generate an external auto-sampled template bank for Scenario B."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from task_b.templates.sampler import (
    build_generated_templates_payload,
    enumerate_failed_update_template_cases,
    generate_templates_for_circuit,
)
from task_b.templates.verify import verify_templates
from utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate verified external templates for Scenario B")
    parser.add_argument("--circuit-type", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-cases-per-fault", type=int, default=12)
    parser.add_argument(
        "--faults",
        nargs="+",
        default=None,
        help="Optional fault IDs to generate. Omitted faults are written as empty lists.",
    )
    parser.add_argument("--max-powered-conditions", type=int, default=2)
    parser.add_argument("--t0-min-survivors", type=int, default=2)
    parser.add_argument(
        "--t0-max-survivors",
        type=int,
        default=0,
        help="Optional maximum survivor count after T0; 0 means no upper bound",
    )
    parser.add_argument(
        "--exclude-isolation-healthy",
        action="store_true",
        help="Do not use isolation_<component>=healthy in the candidate pool",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=("default", "belief_prone_correction_v2"),
        default="default",
        help="Template ordering/selection strategy before max-cases truncation.",
    )
    return parser.parse_args()


def _rank_case_templates(
    *,
    circuit_type: str,
    fault_id: str,
    templates: Sequence[Dict[str, Any]],
    selection_strategy: str,
) -> List[Dict[str, Any]]:
    if selection_strategy == "default":
        return list(templates)

    from task_b.domain.rule_engine import compute_ground_truth_history, enrich_event
    from task_b.experiments.belief_stats import _failed_update_template_score_v2

    scored: List[Tuple[Tuple[Any, ...], int]] = []
    for template_idx, template in enumerate(templates):
        t0_meas = dict(template["t0"]["meas"][0])
        t1_meas = dict(template["t1"]["meas"][0])
        t2_meas = dict(template["t2"]["meas"][0])
        events = [
            enrich_event({"turn": 0, "type": "initial_measurement", "measurements": t0_meas}),
            enrich_event({"turn": 1, "type": "misleading_measurement", "measurements": t1_meas}),
            enrich_event(
                {
                    "turn": 2,
                    "type": "retraction_measurement",
                    "retract_turn": 1,
                    "measurements": t2_meas,
                }
            ),
        ]
        challenge_dict = {
            "circuit_type": circuit_type,
            "oracle": fault_id,
            "misleading_target": template["misleading_target"],
            "events": events,
            "ground_truth": compute_ground_truth_history(events, circuit_type),
        }
        score = _failed_update_template_score_v2(challenge_dict)
        scored.append((score, template_idx))

    scored.sort(key=lambda item: item[0])
    return [templates[template_idx] for _, template_idx in scored]


def _generate_ranked_templates_for_circuit(
    *,
    circuit_type: str,
    max_cases_per_fault: int,
    max_powered_conditions: int,
    include_isolation_healthy: bool,
    t0_min_survivors: int,
    t0_max_survivors: Optional[int],
    selection_strategy: str,
    faults: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    from task_b.domain.rule_engine import get_topology

    topology = get_topology(circuit_type)
    selected_faults = set(faults or topology.fault_ids)
    templates: Dict[str, List[Dict[str, Any]]] = {fault_id: [] for fault_id in topology.fault_ids}
    for fault_id in topology.fault_ids:
        if fault_id not in selected_faults:
            continue
        cases = enumerate_failed_update_template_cases(
            circuit_type,
            fault_id,
            max_powered_conditions=max_powered_conditions,
            include_isolation_healthy=include_isolation_healthy,
            t0_min_survivors=t0_min_survivors,
            t0_max_survivors=t0_max_survivors,
        )
        ranked = _rank_case_templates(
            circuit_type=circuit_type,
            fault_id=fault_id,
            templates=[case.as_template_case() for case in cases],
            selection_strategy=selection_strategy,
        )
        templates[fault_id] = ranked[:max_cases_per_fault]
    return templates


def main() -> None:
    args = parse_args()
    t0_max_survivors: Optional[int] = args.t0_max_survivors or None

    if args.selection_strategy == "default":
        templates = generate_templates_for_circuit(
            args.circuit_type,
            max_cases_per_fault=args.max_cases_per_fault,
            max_powered_conditions=args.max_powered_conditions,
            include_isolation_healthy=not args.exclude_isolation_healthy,
            t0_min_survivors=args.t0_min_survivors,
            t0_max_survivors=t0_max_survivors,
        )
        if args.faults:
            selected_faults = set(args.faults)
            templates = {
                fault_id: (cases if fault_id in selected_faults else [])
                for fault_id, cases in templates.items()
            }
    else:
        templates = _generate_ranked_templates_for_circuit(
            circuit_type=args.circuit_type,
            max_cases_per_fault=args.max_cases_per_fault,
            max_powered_conditions=args.max_powered_conditions,
            include_isolation_healthy=not args.exclude_isolation_healthy,
            t0_min_survivors=args.t0_min_survivors,
            t0_max_survivors=t0_max_survivors,
            selection_strategy=args.selection_strategy,
            faults=args.faults,
        )

    ok, errors = verify_templates(
        templates,
        circuit_type=args.circuit_type,
        require_single_condition=False,
        require_all_faults=True,
    )
    if not ok:
        print("Generated templates failed verification:")
        for error in errors[:50]:
            print(f"  - {error}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more")
        raise SystemExit(1)

    payload = build_generated_templates_payload(
        args.circuit_type,
        templates,
        max_cases_per_fault=args.max_cases_per_fault,
        max_powered_conditions=args.max_powered_conditions,
        include_isolation_healthy=not args.exclude_isolation_healthy,
        t0_min_survivors=args.t0_min_survivors,
        t0_max_survivors=t0_max_survivors,
        verified=True,
    )
    payload["metadata"]["search_config"]["selection_strategy"] = args.selection_strategy
    payload["metadata"]["search_config"]["faults"] = args.faults

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_json(args.output, payload)

    counts = payload["metadata"]["case_counts"]
    print(f"Generated verified template bank for {args.circuit_type}")
    for fault_id, count in counts.items():
        print(f"  {fault_id}: {count}")
    print(f"Total cases: {payload['metadata']['total_cases']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
