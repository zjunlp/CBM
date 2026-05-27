"""Fault-domain constants for Scenario B."""

from __future__ import annotations

from typing import Dict, List, Tuple


FAULT_OPTIONS: Dict[str, str] = {
    "A": (
        "Battery no output "
        "(the battery cannot provide usable voltage at its external terminals in this benchmark)"
    ),
    "B": "Switch stuck open (not conducting)",
    "C": "Switch stuck closed (conducting)",
    "D": "R1 open path (resistor branch broken or burned out)",
    "E": "R1 short path (resistor branch failed short, 0 Ω)",
    "F": "R2 open path (resistor branch broken or burned out)",
    "G": "R2 short path (resistor branch failed short, 0 Ω)",
    "H": "R3 open path (resistor branch broken or burned out)",
    "I": "R3 short path (resistor branch failed short, 0 Ω)",
}

FAULT_IDS: List[str] = list(FAULT_OPTIONS.keys())

FAULT_TO_COMPONENT: Dict[str, str] = {
    "A": "battery",
    "B": "switch",
    "C": "switch",
    "D": "r1",
    "E": "r1",
    "F": "r2",
    "G": "r2",
    "H": "r3",
    "I": "r3",
}

COMPONENT_TO_FAULT: Dict[str, Tuple[str, ...]] = {
    component: tuple(
        fault_id
        for fault_id, fault_component in FAULT_TO_COMPONENT.items()
        if fault_component == component
    )
    for component in dict.fromkeys(FAULT_TO_COMPONENT.values())
}

OPEN_CIRCUIT_FAULTS: frozenset[str] = frozenset({"A", "B", "D", "F", "H"})
SHORT_CIRCUIT_FAULTS: frozenset[str] = frozenset({"C", "E", "G", "I"})
COMPONENT_NAMES: Tuple[str, ...] = tuple(dict.fromkeys(FAULT_TO_COMPONENT.values()))
