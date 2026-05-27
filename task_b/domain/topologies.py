"""Topology definitions for Scenario B."""

from __future__ import annotations

from typing import Any, Dict

from task_b.domain.faults import FAULT_OPTIONS, FAULT_TO_COMPONENT


PARALLEL_SERIES_PAIRS_FAULT_OPTIONS: Dict[str, str] = {
    "A": FAULT_OPTIONS["A"],
    "B": FAULT_OPTIONS["B"],
    "C": FAULT_OPTIONS["C"],
    "D": "R1 open path (series-pair branch element broken or burned out)",
    "E": "R1 short path (series-pair branch element failed short, 0 Ω)",
    "F": "R2 open path (series-pair branch element broken or burned out)",
    "G": "R2 short path (series-pair branch element failed short, 0 Ω)",
    "H": "R3 open path (series-pair branch element broken or burned out)",
    "I": "R3 short path (series-pair branch element failed short, 0 Ω)",
    "J": "R4 open path (series-pair branch element broken or burned out)",
    "K": "R4 short path (series-pair branch element failed short, 0 Ω)",
}

PARALLEL_SERIES_PAIRS_FAULT_TO_COMPONENT: Dict[str, str] = {
    "A": "battery",
    "B": "switch",
    "C": "switch",
    "D": "r1",
    "E": "r1",
    "F": "r2",
    "G": "r2",
    "H": "r3",
    "I": "r3",
    "J": "r4",
    "K": "r4",
}

PARALLEL_R12_FAULT_OPTIONS: Dict[str, str] = {
    fault_id: FAULT_OPTIONS[fault_id]
    for fault_id in ("A", "B", "C", "D", "E", "F", "G")
}

PARALLEL_R12_FAULT_TO_COMPONENT: Dict[str, str] = {
    fault_id: FAULT_TO_COMPONENT[fault_id]
    for fault_id in PARALLEL_R12_FAULT_OPTIONS
}


TOPOLOGY_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "series_all": {
        "architecture": {
            "name": "Series Circuit",
            "description": (
                "An ideal basic series circuit with a Battery, a Switch, Resistor 1 (R1), "
                "and Resistor 2 (R2). The battery has one no-output fault; the switch and "
                "resistors each have one open-path fault and one short-path fault."
            ),
            "components": ["battery", "switch", "r1", "r2"],
            "topology": "series",
        },
        "fault_options": PARALLEL_R12_FAULT_OPTIONS,
        "fault_to_component": PARALLEL_R12_FAULT_TO_COMPONENT,
        "abnormal_support": {
            "ammeter_absence": ("A", "B", "D", "F"),
            "ammeter_presence": ("C", "E", "G"),
            "voltmeter_r1_absence": ("A", "B", "E", "F"),
            "voltmeter_r1_presence": ("C", "D", "G"),
            "voltmeter_r2_absence": ("A", "B", "D", "G"),
            "voltmeter_r2_presence": ("C", "E", "F"),
            "voltmeter_battery_absence": ("A",),
            "voltmeter_battery_presence": ("B", "C", "D", "E", "F", "G"),
        },
    },
    "parallel_r12_series": {
        "architecture": {
            "name": "Parallel Circuit",
            "description": (
                "An ideal parallel circuit. The Battery and Switch are in the main branch. "
                "Each parallel branch has a resistor (R1, R2). The battery has one no-output "
                "fault; the switch and resistors each have one open-path fault and one short-path fault."
            ),
            "components": ["battery", "switch", "r1", "r2"],
            "topology": "parallel_r12_series",
        },
        "fault_options": PARALLEL_R12_FAULT_OPTIONS,
        "fault_to_component": PARALLEL_R12_FAULT_TO_COMPONENT,
        "abnormal_support": {
            "ammeter_main_absence": ("A", "B"),
            "ammeter_main_presence": ("C", "D", "E", "F", "G"),
            "ammeter_r1_absence": ("A", "B", "D", "G"),
            "ammeter_r1_presence": ("C", "E", "F"),
            "ammeter_r2_absence": ("A", "B", "E", "F"),
            "ammeter_r2_presence": ("C", "D", "G"),
            "voltmeter_r1_absence": ("A", "B", "E", "G"),
            "voltmeter_r1_presence": ("C", "D", "F"),
            "voltmeter_r2_absence": ("A", "B", "E", "G"),
            "voltmeter_r2_presence": ("C", "D", "F"),
            "voltmeter_battery_absence": ("A",),
            "voltmeter_battery_presence": ("B", "C", "D", "E", "F", "G"),
        },
    },
    "parallel_r123_series": {
        "architecture": {
            "name": "Parallel Circuit (Three Resistor Branches)",
            "description": (
                "An ideal parallel circuit. The Battery and Switch are in the main branch. "
                "R1, R2, and R3 form three parallel branches. "
                "The battery has one no-output fault; the switch and all resistors have "
                "open/short faults."
            ),
            "components": ["battery", "switch", "r1", "r2", "r3"],
            "topology": "parallel_r123_series",
        },
        "abnormal_support": {
            "ammeter_main_absence": ("A", "B"),
            "ammeter_main_presence": ("C", "D", "E", "F", "G", "H", "I"),
            "ammeter_r1_absence": ("A", "B", "D", "G", "I"),
            "ammeter_r1_presence": ("C", "E", "F", "H"),
            "ammeter_r2_absence": ("A", "B", "E", "F", "I"),
            "ammeter_r2_presence": ("C", "D", "G", "H"),
            "ammeter_r3_absence": ("A", "B", "E", "G", "H"),
            "ammeter_r3_presence": ("C", "D", "F", "I"),
            "voltmeter_r1_absence": ("A", "B", "E", "G", "I"),
            "voltmeter_r1_presence": ("C", "D", "F", "H"),
            "voltmeter_r2_absence": ("A", "B", "E", "G", "I"),
            "voltmeter_r2_presence": ("C", "D", "F", "H"),
            "voltmeter_r3_absence": ("A", "B", "E", "G", "I"),
            "voltmeter_r3_presence": ("C", "D", "F", "H"),
            "voltmeter_battery_absence": ("A",),
            "voltmeter_battery_presence": ("B", "C", "D", "E", "F", "G", "H", "I"),
        },
    },
    "parallel_series_pairs": {
        "architecture": {
            "name": "Parallel Circuit (Two Series Resistor Branches)",
            "description": (
                "An ideal parallel circuit. The Battery and Switch are in the main branch. "
                "One parallel branch contains R1 and R2 in series; the other parallel branch "
                "contains R3 and R4 in series. The battery has one no-output fault; the switch "
                "and all resistors have open/short faults."
            ),
            "components": ["battery", "switch", "r1", "r2", "r3", "r4"],
            "topology": "parallel_series_pairs",
        },
        "fault_options": PARALLEL_SERIES_PAIRS_FAULT_OPTIONS,
        "fault_to_component": PARALLEL_SERIES_PAIRS_FAULT_TO_COMPONENT,
        "abnormal_support": {
            "ammeter_main_absence": ("A", "B"),
            "ammeter_main_presence": ("C", "D", "E", "F", "G", "H", "I", "J", "K"),
            "ammeter_r1_absence": ("A", "B", "D", "F"),
            "ammeter_r1_presence": ("C", "E", "G", "H", "I", "J", "K"),
            "ammeter_r2_absence": ("A", "B", "D", "F"),
            "ammeter_r2_presence": ("C", "E", "G", "H", "I", "J", "K"),
            "ammeter_r3_absence": ("A", "B", "H", "J"),
            "ammeter_r3_presence": ("C", "D", "E", "F", "G", "I", "K"),
            "ammeter_r4_absence": ("A", "B", "H", "J"),
            "ammeter_r4_presence": ("C", "D", "E", "F", "G", "I", "K"),
            "voltmeter_r1_absence": ("A", "B", "E", "F"),
            "voltmeter_r1_presence": ("C", "D", "G", "H", "I", "J", "K"),
            "voltmeter_r2_absence": ("A", "B", "D", "G"),
            "voltmeter_r2_presence": ("C", "E", "F", "H", "I", "J", "K"),
            "voltmeter_r3_absence": ("A", "B", "I", "J"),
            "voltmeter_r3_presence": ("C", "D", "E", "F", "G", "H", "K"),
            "voltmeter_r4_absence": ("A", "B", "H", "K"),
            "voltmeter_r4_presence": ("C", "D", "E", "F", "G", "I", "J"),
            "voltmeter_battery_absence": ("A",),
            "voltmeter_battery_presence": ("B", "C", "D", "E", "F", "G", "H", "I", "J", "K"),
        },
    },
}

for definition in TOPOLOGY_DEFINITIONS.values():
    definition.setdefault("fault_options", FAULT_OPTIONS)
    definition.setdefault("fault_to_component", FAULT_TO_COMPONENT)


def get_topology_definition(circuit_type: str) -> Dict[str, Any]:
    try:
        return TOPOLOGY_DEFINITIONS[circuit_type]
    except KeyError as exc:
        available = sorted(TOPOLOGY_DEFINITIONS)
        raise ValueError(
            f"Unknown circuit type '{circuit_type}'. Available: {available}"
        ) from exc
