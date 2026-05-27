"""Measurement parsing and protocol helpers for Scenario B."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from task_b.domain.faults import COMPONENT_NAMES


_INSTRUMENT_MARKERS: Dict[str, Tuple[str, ...]] = {
    "ammeter_main": ("ammeter_main", "current_main"),
    "ammeter_r1": ("ammeter_r1", "current_r1"),
    "ammeter_r3": ("ammeter_r3", "current_r3"),
    "ammeter_r4": ("ammeter_r4", "current_r4"),
    "ammeter_r2": ("ammeter_r2", "current_r2"),
    "ammeter": ("ammeter", "current"),
    "voltmeter_r1": ("voltmeter_r1", "v_r1"),
    "voltmeter_r3": ("voltmeter_r3", "v_r3"),
    "voltmeter_r4": ("voltmeter_r4", "v_r4"),
    "voltmeter_r2": ("voltmeter_r2", "v_r2"),
    "voltmeter_battery": ("voltmeter_battery", "v_bat", "battery", "v_battery"),
    "switch": ("switch", "sw"),
    "r1": ("r1",),
    "r3": ("r3",),
    "r4": ("r4",),
    "r2": ("r2",),
}

_RETEST_KEY_MARKERS: Tuple[str, ...] = (
    "retest",
    "ohmmeter",
    "resistance",
    "isolation",
    "continuity",
)

_BENCHMARK_PROTOCOL: Dict[str, Any] = {
    "protocol_version": "task_b_v4_6_fault9",
    "single_fault_assumption": True,
    "fault_count": 9,
    "measurement_assumptions": {
        "current_readout": (
            "Branch-current readings are treated as direct, low-disturbance observations "
            "in this benchmark model. This is an idealized approximation and does not imply "
            "strictly zero-intrusion measurement in real hardware."
        ),
    },
    "measurement_modes": {
        "powered": {
            "setup_type": "canonical_powered_diagnostic",
            "power_state": "connected",
            "switch_command": "closed",
            "description": (
                "Powered measurements are taken under a canonical diagnostic setup "
                "with power connected and the switch commanded closed."
            ),
        },
        "isolation": {
            "setup_type": "canonical_isolation_check",
            "power_state": "removed",
            "description": (
                "Isolation checks are taken with power removed and the named component isolated."
            ),
        },
        "none": {
            "setup_type": "no_new_measurement",
            "description": "No new measurements are added on this turn.",
        },
    },
    "update_policy": {
        "recompute_from_scratch": True,
        "active_evidence_only": True,
        "description": (
            "When evidence is retracted, discard the old event and recompute survivors "
            "from scratch using only active evidence."
        ),
    },
}


def normalize_token(text: str) -> str:
    return text.strip().lower().replace("-", "_")


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def get_benchmark_protocol() -> Dict[str, Any]:
    return copy.deepcopy(_BENCHMARK_PROTOCOL)


def parse_instrument_key(key: str) -> Optional[str]:
    normalized_key = normalize_token(key)
    for instrument, markers in _INSTRUMENT_MARKERS.items():
        if _contains_any(normalized_key, markers):
            return instrument
    return None


def identify_component_name(key: str) -> Optional[str]:
    normalized_key = normalize_token(key)
    for component in COMPONENT_NAMES:
        if component in normalized_key:
            return component
    for component in ("r4",):
        if component in normalized_key:
            return component
    return None


def is_retest_key(key: str) -> bool:
    return _contains_any(normalize_token(key), _RETEST_KEY_MARKERS)


def infer_measurement_mode(key: str) -> str:
    if is_retest_key(key):
        return "isolation"
    if parse_instrument_key(key) is not None:
        return "powered"
    raise ValueError(f"Unknown measurement key '{key}'")


def infer_measurement_target(key: str) -> str:
    if is_retest_key(key):
        component = identify_component_name(key)
        if component is None:
            raise ValueError(f"Unknown isolation target for key '{key}'")
        return component

    instrument = parse_instrument_key(key)
    if instrument is None:
        raise ValueError(f"Unknown powered measurement key '{key}'")

    target_map = {
        "ammeter": "main_loop",
        "ammeter_main": "main_branch",
        "ammeter_r1": "r1_branch",
        "ammeter_r3": "r3_branch",
        "ammeter_r4": "r4_branch",
        "ammeter_r2": "r2_branch",
        "voltmeter_battery": "battery_terminals",
        "voltmeter_r1": "r1",
        "voltmeter_r3": "r3",
        "voltmeter_r4": "r4",
        "voltmeter_r2": "r2",
        "switch": "switch",
    }
    return target_map.get(instrument, instrument)


def infer_measurement_mode_for_mapping(measurements: Mapping[str, str]) -> str:
    if not measurements:
        return "none"

    modes = {infer_measurement_mode(key) for key in measurements}
    if len(modes) != 1:
        raise ValueError(
            "measurements in one event must use a single mode; "
            f"got {sorted(modes)} for keys {sorted(measurements.keys())}"
        )
    return next(iter(modes))


def infer_measurement_targets(measurements: Mapping[str, str]) -> List[str]:
    return sorted({infer_measurement_target(key) for key in measurements})


def build_event_setup(
    measurements: Mapping[str, str],
    measurement_mode: Optional[str] = None,
) -> Dict[str, Any]:
    mode = measurement_mode or infer_measurement_mode_for_mapping(measurements)
    if mode == "none":
        return {"power_state": "unchanged", "setup_type": "no_new_measurement"}
    if mode == "powered":
        return {
            "setup_type": "canonical_powered_diagnostic",
            "power_state": "connected",
            "switch_command": "closed",
            "targets": infer_measurement_targets(measurements),
        }
    if mode == "isolation":
        targets = infer_measurement_targets(measurements)
        return {
            "setup_type": "canonical_isolation_check",
            "power_state": "removed",
            "isolated_component": targets[0] if targets else None,
            "targets": targets,
        }
    raise ValueError(f"Unknown measurement mode '{mode}'")


def build_update_policy(
    event_type: str,
    retract_turn: Optional[int] = None,
    has_measurements: bool = True,
) -> Dict[str, Any]:
    discarded_turns = [int(retract_turn)] if retract_turn is not None else []
    return {
        "discarded_turns": discarded_turns,
        "adds_measurements": bool(has_measurements),
        "recompute_from_scratch": True,
        "active_evidence_only": True,
        "operation": (
            "discard_only"
            if event_type == "pure_retraction"
            else "discard_then_add"
            if retract_turn is not None
            else "append_measurement"
        ),
    }
