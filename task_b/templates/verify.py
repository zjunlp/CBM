"""Validation helpers for Scenario B templates and generated challenges."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

from task_b.domain.measurement_protocol import identify_component_name, parse_instrument_key
from task_b.domain.rule_engine import (
    compute_ground_truth_history,
    enrich_event,
    get_topology,
    supported_faults_for_condition,
    validate_challenge_against_rules,
)


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_valid_measurement_mapping(measurements: Any, allow_empty: bool = False) -> bool:
    if not isinstance(measurements, Mapping):
        return False
    return allow_empty or bool(measurements)


def _normalize_template_cases(template: Any) -> List[Dict[str, Any]]:
    if isinstance(template, list):
        cases = template
    else:
        cases = [template]

    normalized: List[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            normalized.append(case)
            continue

        if {"t0", "t1", "t2"}.issubset(case.keys()):
            t1 = case.get("t1")
            if isinstance(t1, Mapping) and "misleading_target" not in t1 and "misleading_target" in case:
                merged_case = dict(case)
                merged_t1 = dict(t1)
                merged_t1["misleading_target"] = case["misleading_target"]
                merged_case["t1"] = merged_t1
                normalized.append(merged_case)
            else:
                normalized.append(dict(case))
            continue

        normalized.append(case)
    return normalized


def verify_templates(
    templates: Dict[str, Any],
    circuit_type: Optional[str] = None,
    require_single_condition: bool = False,
    require_all_faults: bool = True,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not isinstance(templates, dict):
        return False, ["templates must be a dict"]

    topology = get_topology(circuit_type)
    fault_ids = topology.fault_ids
    fault_options = topology.fault_options

    if require_all_faults:
        missing = [fault_id for fault_id in fault_ids if fault_id not in templates]
        extra = [key for key in templates.keys() if key not in fault_ids]
        if missing:
            errors.append(f"missing faults: {missing}")
        if extra:
            errors.append(f"unknown faults: {extra}")

    for fault_id in fault_ids:
        template_value = templates.get(fault_id)
        if template_value is None:
            continue
        cases = _normalize_template_cases(template_value)
        for case_idx, template in enumerate(cases):
            fault_errors: List[str] = []
            case_label = f"{fault_id}:case {case_idx}"
            if not isinstance(template, dict):
                errors.append(f"{case_label}: template must be a dict")
                continue

            t0 = template.get("t0")
            t1 = template.get("t1")
            t2 = template.get("t2")
            if not isinstance(t0, dict) or not isinstance(t1, dict) or not isinstance(t2, dict):
                errors.append(f"{case_label}: t0/t1/t2 must be dicts")
                continue

            misleading_target = t1.get("misleading_target")
            if misleading_target not in fault_options:
                fault_errors.append(f"{case_label}: invalid misleading_target={misleading_target}")
            elif misleading_target == fault_id:
                fault_errors.append(f"{case_label}: misleading_target cannot equal oracle")

            for turn_name, turn_data in (("t0", t0), ("t1", t1), ("t2", t2)):
                meas_list = turn_data.get("meas")
                if not isinstance(meas_list, list) or not meas_list:
                    fault_errors.append(f"{case_label}: {turn_name}.meas must be a non-empty list")
                    continue
                for idx, measurements in enumerate(meas_list):
                    if not _is_valid_measurement_mapping(measurements):
                        fault_errors.append(f"{case_label}: {turn_name}.meas[{idx}] must be a non-empty mapping")
                        continue
                    if require_single_condition and len(measurements) != 1:
                        fault_errors.append(
                            f"{case_label}: {turn_name}.meas[{idx}] must contain exactly one condition"
                        )
                    for key, value in measurements.items():
                        if not _is_non_empty_str(key) or not _is_non_empty_str(value):
                            fault_errors.append(
                                f"{case_label}: {turn_name}.meas[{idx}] key/value must be non-empty strings"
                            )
                        elif parse_instrument_key(key) is None and identify_component_name(key) is None:
                            fault_errors.append(
                                f"{case_label}: {turn_name}.meas[{idx}] unknown instrument/component key '{key}'"
                            )
                    try:
                        enrich_event({
                            "turn": 0,
                            "type": f"{turn_name}_measurement",
                            "measurements": dict(measurements),
                        })
                    except Exception as exc:
                        fault_errors.append(f"{case_label}: {turn_name}.meas[{idx}] {exc}")

            if fault_errors:
                errors.extend(fault_errors)
                continue

            variant_count = max(
                len(t0["meas"]),
                len(t1["meas"]),
                len(t2["meas"]),
            )
            for idx in range(variant_count):
                t0_meas = dict(t0["meas"][idx % len(t0["meas"])])
                t1_meas = dict(t1["meas"][idx % len(t1["meas"])])
                t2_meas = dict(t2["meas"][idx % len(t2["meas"])])
                events = [
                    enrich_event({"turn": 0, "type": "initial_measurement", "measurements": t0_meas}),
                    enrich_event({"turn": 1, "type": "misleading_measurement", "measurements": t1_meas}),
                    enrich_event({
                        "turn": 2,
                        "type": "retraction_measurement",
                        "retract_turn": 1,
                        "measurements": t2_meas,
                    }),
                ]
                challenge_dict = {
                    "circuit_type": circuit_type,
                    "oracle": fault_id,
                    "misleading_target": misleading_target,
                    "t0_survivors": sorted(
                        supported_faults_for_condition(t0_meas, circuit_type)
                    ),
                    "events": events,
                    "ground_truth": [
                        {"turn": item["turn"], "survivors": sorted(item["survivors"])}
                        for item in compute_ground_truth_history(events, circuit_type)
                    ],
                }
                ok, challenge_errors = validate_challenge_against_rules(challenge_dict)
                if not ok:
                    errors.extend(
                        f"{case_label}: variant {idx}: {message}"
                        for message in challenge_errors
                    )

    return len(errors) == 0, errors


def verify_challenge_dict(
    challenge_dict: Dict[str, Any],
    require_single_condition: bool = False,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    events = challenge_dict.get("events")
    if not isinstance(events, list) or len(events) != 3:
        errors.append("events must be a list of length 3")
    else:
        for idx, event in enumerate(events):
            measurements = event.get("measurements") if isinstance(event, dict) else None
            allow_empty = (event.get("type") == "pure_retraction")
            if not _is_valid_measurement_mapping(measurements, allow_empty=allow_empty):
                errors.append(f"events[{idx}].measurements must be a non-empty mapping")
                continue
            if require_single_condition and len(measurements) != 1:
                errors.append(f"events[{idx}].measurements must contain exactly one condition")
            for key, value in measurements.items():
                if not _is_non_empty_str(key) or not _is_non_empty_str(value):
                    errors.append(f"events[{idx}] contains empty key/value")
                elif parse_instrument_key(key) is None and identify_component_name(key) is None:
                    errors.append(f"events[{idx}] unknown instrument/component key '{key}'")

    rules_ok, rule_errors = validate_challenge_against_rules(challenge_dict)
    if not rules_ok:
        errors.extend(rule_errors)

    return len(errors) == 0, errors


def build_verified_output_payload(
    templates: Dict[str, Any],
    source_model: str,
    source_base_url: str,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "verified": True,
            "verified_at": datetime.now().isoformat(),
            "verify_rules": {
                "require_all_faults": True,
                "require_single_condition": False,
                "rules_validated": True,
                "rules_mode": "topology_consistency",
            },
            "source": {
                "model": source_model,
                "base_url": source_base_url,
            },
        },
        "templates": templates,
    }


def extract_templates_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and "templates" in payload and isinstance(payload["templates"], dict):
        return payload["templates"]
    return payload
