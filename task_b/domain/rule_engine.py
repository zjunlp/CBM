"""Rule engine and challenge validation for Scenario B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from task_b.domain.faults import (
    FAULT_IDS,
    FAULT_OPTIONS,
)
from task_b.domain.measurement_protocol import (
    build_event_setup,
    build_update_policy,
    identify_component_name,
    infer_measurement_mode_for_mapping,
    infer_measurement_targets,
    is_retest_key,
    normalize_token,
    parse_instrument_key,
)
from task_b.domain.topologies import TOPOLOGY_DEFINITIONS


@dataclass(frozen=True)
class MeasurementCondition:
    key: str
    value: str
    component: str
    is_retest: bool


def _normalize_powered_value(value: str) -> str:
    normalized_value = normalize_token(value)
    if normalized_value in ("0a", "0v", "0", "absence", "none", "off"):
        return "absence"
    if normalized_value in ("presence", "normal", "ok", "battery_voltage", "high_current", "on"):
        return "presence"
    return normalized_value


def _supported_faults_for_isolation(
    component: str,
    normalized_value: str,
    *,
    fault_ids: Sequence[str],
    component_to_fault: Mapping[str, Sequence[str]],
    open_circuit_faults: Set[str],
    short_circuit_faults: Set[str],
) -> Set[str]:
    all_faults = set(fault_ids)
    component_faults = set(component_to_fault.get(component, ()))

    if component == "battery":
        if normalized_value in ("normal", "ok", "good", "healthy"):
            return {fault_id for fault_id in all_faults if fault_id != "A"}
        if normalized_value in (
            "infinite",
            "open",
            "open_circuit",
            "open_path",
            "disconnected",
            "0ohm",
            "short",
            "short_circuit",
            "short_path",
            "closed",
            "no_output",
        ):
            return {"A"}

    if normalized_value in ("normal", "ok", "good", "healthy"):
        return {fault_id for fault_id in all_faults if fault_id not in component_faults}
    if normalized_value in ("infinite", "open", "open_circuit", "open_path", "disconnected"):
        return component_faults & open_circuit_faults
    if normalized_value in ("0ohm", "short", "short_circuit", "short_path", "closed"):
        return component_faults & short_circuit_faults
    return all_faults


@dataclass(frozen=True)
class CircuitTopology:
    circuit_type: str
    name: str
    description: str
    components: Tuple[str, ...]
    abnormal_support: Dict[str, Tuple[str, ...]]
    fault_options: Dict[str, str]
    fault_to_component: Dict[str, str]
    open_circuit_faults: frozenset[str]
    short_circuit_faults: frozenset[str]

    @property
    def fault_ids(self) -> Tuple[str, ...]:
        return tuple(self.fault_options.keys())

    @property
    def component_to_fault(self) -> Dict[str, Tuple[str, ...]]:
        return {
            component: tuple(
                fault_id
                for fault_id, fault_component in self.fault_to_component.items()
                if fault_component == component
            )
            for component in dict.fromkeys(self.fault_to_component.values())
        }

    def supported_faults(self, condition: MeasurementCondition) -> Set[str]:
        normalized_value = normalize_token(condition.value)
        if condition.is_retest:
            return _supported_faults_for_isolation(
                condition.component,
                normalized_value,
                fault_ids=self.fault_ids,
                component_to_fault=self.component_to_fault,
                open_circuit_faults=set(self.open_circuit_faults),
                short_circuit_faults=set(self.short_circuit_faults),
            )

        specific_key = f"{condition.component}_{_normalize_powered_value(normalized_value)}"
        if specific_key in self.abnormal_support:
            return set(self.abnormal_support[specific_key])
        return set(self.fault_ids)

    def is_fault_consistent(self, fault_id: str, condition: MeasurementCondition) -> bool:
        return fault_id in self.supported_faults(condition)

    @property
    def architecture(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "components": list(self.components),
            "topology": self.circuit_type,
        }


def _build_rule_guide_text(
    circuit_type: str,
    abnormal_support: Mapping[str, Sequence[str]],
) -> str:
    lines = [f"{circuit_type} consistency rules:"]
    for measurement_key, survivors in abnormal_support.items():
        lines.append(f"- {measurement_key} abnormal -> {list(survivors)}")
    lines.append("- isolation_<component>=open -> open-path fault on that component")
    lines.append("- isolation_<component>=short -> short-path fault on that component")
    lines.append("- isolation_<component>=healthy -> exclude faults on that component")
    return "\n".join(lines)


def _build_topology_registry() -> Dict[str, CircuitTopology]:
    registry: Dict[str, CircuitTopology] = {}
    for circuit_type, definition in TOPOLOGY_DEFINITIONS.items():
        architecture = dict(definition["architecture"])
        abnormal_support = dict(definition["abnormal_support"])
        fault_options = dict(definition.get("fault_options", FAULT_OPTIONS))
        fault_to_component = dict(definition.get("fault_to_component", {}))
        open_circuit_faults = frozenset(
            fault_id
            for fault_id, description in fault_options.items()
            if fault_id == "A" or "open" in description.lower() or "no output" in description.lower()
        )
        short_circuit_faults = frozenset(
            fault_id
            for fault_id, description in fault_options.items()
            if "short" in description.lower() or "stuck closed" in description.lower()
        )
        registry[circuit_type] = CircuitTopology(
            circuit_type=circuit_type,
            name=str(architecture["name"]),
            description=str(architecture["description"]),
            components=tuple(architecture.get("components", tuple())),
            abnormal_support={
                measurement_key: tuple(survivors)
                for measurement_key, survivors in abnormal_support.items()
            },
            fault_options=fault_options,
            fault_to_component=fault_to_component,
            open_circuit_faults=open_circuit_faults,
            short_circuit_faults=short_circuit_faults,
        )
    return registry


_TOPOLOGY_REGISTRY: Dict[str, CircuitTopology] = _build_topology_registry()

CIRCUIT_TYPES: List[str] = list(_TOPOLOGY_REGISTRY.keys())
CIRCUIT_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    circuit_type: topology.architecture
    for circuit_type, topology in _TOPOLOGY_REGISTRY.items()
}
RULE_GUIDE_TEXT: str = "\n\n".join(
    _build_rule_guide_text(circuit_type, topology.abnormal_support)
    for circuit_type, topology in _TOPOLOGY_REGISTRY.items()
)


def get_default_circuit_type() -> str:
    if not CIRCUIT_TYPES:
        raise ValueError("No circuit types registered")
    return CIRCUIT_TYPES[0]


def get_topology(circuit_type: Optional[str] = None) -> CircuitTopology:
    selected = circuit_type or get_default_circuit_type()
    if selected not in _TOPOLOGY_REGISTRY:
        raise ValueError(f"Unknown circuit type: {selected}")
    return _TOPOLOGY_REGISTRY[selected]


def get_rule_guide_text(circuit_type: Optional[str] = None) -> str:
    if circuit_type is None:
        return RULE_GUIDE_TEXT
    topology = get_topology(circuit_type)
    return _build_rule_guide_text(circuit_type, topology.abnormal_support)


def _parse_measurement_condition(key: str, value: str) -> MeasurementCondition:
    if is_retest_key(key):
        component = identify_component_name(key)
        if component is None:
            component = parse_instrument_key(key) or key
    else:
        component = parse_instrument_key(key)
        if component is None:
            raise ValueError(f"Unknown instrument for key '{key}'")

    return MeasurementCondition(
        key=key,
        value=value,
        component=component,
        is_retest=is_retest_key(key),
    )


def _parse_measurements(measurements: Mapping[str, str]) -> List[MeasurementCondition]:
    if not isinstance(measurements, Mapping) or not measurements:
        raise ValueError("measurements must be a non-empty mapping")

    conditions: List[MeasurementCondition] = []
    for key, value in measurements.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("measurement keys and values must be strings")
        conditions.append(_parse_measurement_condition(key, value))
    return conditions


def supported_faults_for_condition(
    condition: Mapping[str, str],
    circuit_type: Optional[str] = None,
) -> Set[str]:
    topology = get_topology(circuit_type)
    survivors = set(topology.fault_ids)
    for parsed_condition in _parse_measurements(condition):
        survivors &= topology.supported_faults(parsed_condition)
    return survivors


def _compute_surviving_faults(
    evidence: Sequence[Mapping[str, str]],
    circuit_type: Optional[str] = None,
) -> Set[str]:
    topology = get_topology(circuit_type)
    survivors = set(topology.fault_ids)
    for condition in evidence:
        survivors &= supported_faults_for_condition(condition, circuit_type)
    return survivors


def enrich_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    measurements = event.get("measurements")
    if not isinstance(measurements, Mapping):
        raise ValueError("event.measurements must be a mapping")

    event_type = str(event.get("type", "measurement"))
    retract_turn = event.get("retract_turn")
    measurement_mode = str(
        event.get("measurement_mode")
        or infer_measurement_mode_for_mapping(measurements)
    )

    setup = event.get("setup")
    if not isinstance(setup, Mapping):
        setup = build_event_setup(measurements, measurement_mode)

    update_policy = event.get("update_policy")
    if not isinstance(update_policy, Mapping):
        update_policy = build_update_policy(
            event_type,
            retract_turn=int(retract_turn) if retract_turn is not None else None,
            has_measurements=bool(measurements),
        )

    enriched = dict(event)
    enriched["measurement_mode"] = measurement_mode
    enriched["setup"] = dict(setup)
    enriched["update_policy"] = dict(update_policy)
    return enriched


def compute_ground_truth_history(
    events: Sequence[Mapping[str, Any]],
    circuit_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute survivors after each event, honoring evidence retraction."""
    active_events: Dict[int, Dict[str, Any]] = {}
    steps: List[Dict[str, Any]] = []
    topology = get_topology(circuit_type)
    previous_survivors = set(topology.fault_ids)

    for turn, event in enumerate(events):
        enriched_event = enrich_event(event)
        measurements = enriched_event["measurements"]
        retract_turn = enriched_event.get("retract_turn")
        if retract_turn is not None:
            active_events.pop(int(retract_turn), None)
        if measurements:
            active_events[turn] = enriched_event

        ordered_conditions = [
            active_events[idx]["measurements"]
            for idx in sorted(active_events.keys())
        ]
        survivors = _compute_surviving_faults(ordered_conditions, circuit_type)
        eliminated_faults = previous_survivors - survivors
        reintroduced_faults = survivors - previous_survivors
        steps.append({
            "turn": turn,
            "event_type": enriched_event.get("type", "measurement"),
            "measurements": dict(measurements),
            "measurement_mode": enriched_event["measurement_mode"],
            "setup": dict(enriched_event["setup"]),
            "update_policy": dict(enriched_event["update_policy"]),
            "survivors": survivors,
            "active_measurements": [
                {"source_turn": idx, "measurements": dict(active_events[idx]["measurements"])}
                for idx in sorted(active_events.keys())
            ],
            "active_evidence": [
                {
                    "source_turn": idx,
                    "event_type": active_events[idx].get("type", "measurement"),
                    "measurement_mode": active_events[idx]["measurement_mode"],
                    "setup": dict(active_events[idx]["setup"]),
                    "measurements": dict(active_events[idx]["measurements"]),
                }
                for idx in sorted(active_events.keys())
            ],
            "active_turns": sorted(active_events.keys()),
            "discarded_turns": list(enriched_event["update_policy"].get("discarded_turns", [])),
            "eliminated_faults": sorted(eliminated_faults),
            "reintroduced_faults": sorted(reintroduced_faults),
        })
        previous_survivors = set(survivors)

    return steps


def _validate_event_against_protocol(
    event: Mapping[str, Any],
    event_index: int,
) -> List[str]:
    errors: List[str] = []

    measurements = event.get("measurements")
    if not isinstance(measurements, Mapping):
        return [f"events[{event_index}].measurements must be a mapping"]

    try:
        inferred_mode = infer_measurement_mode_for_mapping(measurements)
    except Exception as exc:
        return [f"events[{event_index}] {exc}"]

    declared_mode = event.get("measurement_mode")
    if declared_mode is not None and declared_mode != inferred_mode:
        errors.append(
            f"events[{event_index}].measurement_mode mismatch: "
            f"declared={declared_mode} inferred={inferred_mode}"
        )

    if inferred_mode == "isolation" and measurements:
        targets = infer_measurement_targets(measurements)
        if len(targets) != 1:
            errors.append(
                f"events[{event_index}] isolation measurements must target exactly one component: "
                f"targets={targets}"
            )

    setup = event.get("setup")
    if isinstance(setup, Mapping):
        setup_power_state = setup.get("power_state")
        if inferred_mode == "powered" and setup_power_state == "removed":
            errors.append(f"events[{event_index}] powered measurements cannot use power_state=removed")
        if inferred_mode == "isolation":
            if setup_power_state == "connected":
                errors.append(f"events[{event_index}] isolation measurements cannot use power_state=connected")
            targets = infer_measurement_targets(measurements)
            isolated_component = setup.get("isolated_component")
            if targets and isolated_component is not None and isolated_component != targets[0]:
                errors.append(
                    f"events[{event_index}] isolated_component mismatch: "
                    f"declared={isolated_component} inferred={targets[0]}"
                )

    update_policy = event.get("update_policy")
    if isinstance(update_policy, Mapping) and event.get("retract_turn") is not None:
        expected = int(event["retract_turn"])
        discarded_turns = list(update_policy.get("discarded_turns", []))
        if expected not in discarded_turns:
            errors.append(
                f"events[{event_index}] update_policy.discarded_turns must include retract_turn={expected}"
            )

    return errors


def _validate_declared_history(
    *,
    computed: Sequence[Mapping[str, Any]],
    oracle: Any,
    misleading_target: Any,
    t0_survivors: Any,
    ground_truth: Any,
    events: Sequence[Mapping[str, Any]],
    require_support_compatibility: bool,
    fault_ids: Sequence[str],
) -> List[str]:
    errors: List[str] = []
    valid_faults = set(fault_ids)

    if isinstance(t0_survivors, list):
        expected_t0 = sorted(computed[0]["survivors"])
        if sorted(t0_survivors) != expected_t0:
            errors.append(
                f"t0_survivors mismatch: declared={sorted(t0_survivors)} computed={expected_t0}"
            )
        if oracle in valid_faults and oracle not in computed[0]["survivors"]:
            errors.append("oracle missing in T0 survivors")
        if misleading_target in valid_faults and misleading_target not in computed[0]["survivors"]:
            errors.append("misleading_target must be in T0 survivors")

    if isinstance(ground_truth, list):
        if len(ground_truth) != len(computed):
            errors.append(
                f"ground_truth length mismatch: declared={len(ground_truth)} computed={len(computed)}"
            )
        else:
            for turn, (declared, expected) in enumerate(zip(ground_truth, computed)):
                declared_survivors = declared.get("survivors") if isinstance(declared, Mapping) else None
                if sorted(declared_survivors or []) != sorted(expected["survivors"]):
                    errors.append(
                        f"turn {turn} survivors mismatch: declared={sorted(declared_survivors or [])} "
                        f"computed={sorted(expected['survivors'])}"
                    )

    if require_support_compatibility and len(computed) >= 2 and oracle in valid_faults:
        if oracle in computed[1]["survivors"]:
            errors.append("turn 1 should exclude the oracle for the failed_update challenge")
        if misleading_target in valid_faults and misleading_target not in computed[1]["survivors"]:
            errors.append("turn 1 should retain the misleading target")

    if require_support_compatibility and computed and oracle in valid_faults:
        final_survivors = computed[-1]["survivors"]
        is_pure_retraction = any(event.get("type") == "pure_retraction" for event in events)
        if not is_pure_retraction and final_survivors != {oracle}:
            errors.append(
                f"final survivors must converge to oracle: final={sorted(final_survivors)} oracle={oracle}"
            )

    return errors


def validate_challenge_against_rules(
    challenge_dict: Dict[str, Any],
    require_support_compatibility: bool = True,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    circuit_type = challenge_dict.get("circuit_type") or get_default_circuit_type()
    oracle = challenge_dict.get("oracle")
    misleading_target = challenge_dict.get("misleading_target")
    events = challenge_dict.get("events")
    ground_truth = challenge_dict.get("ground_truth")
    t0_survivors = challenge_dict.get("t0_survivors")

    topology: Optional[CircuitTopology] = None
    if circuit_type not in _TOPOLOGY_REGISTRY:
        errors.append(f"unknown circuit_type: {circuit_type}")
    else:
        topology = get_topology(str(circuit_type))
    valid_faults = set(topology.fault_ids if topology is not None else FAULT_IDS)
    if oracle not in valid_faults:
        errors.append(f"unknown oracle: {oracle}")
    if misleading_target not in valid_faults:
        errors.append(f"unknown misleading_target: {misleading_target}")
    elif misleading_target == oracle:
        errors.append("misleading_target cannot equal oracle")

    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        return False, errors

    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            errors.append(f"events[{event_index}] must be a mapping")
            continue
        errors.extend(_validate_event_against_protocol(event, event_index))

    try:
        computed = compute_ground_truth_history(events, circuit_type)
    except Exception as exc:
        errors.append(str(exc))
        return False, errors

    errors.extend(
        _validate_declared_history(
            computed=computed,
            oracle=oracle,
            misleading_target=misleading_target,
            t0_survivors=t0_survivors,
            ground_truth=ground_truth,
            events=events,
            require_support_compatibility=require_support_compatibility,
            fault_ids=tuple(valid_faults),
        )
    )

    return len(errors) == 0, errors
