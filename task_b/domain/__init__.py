"""Domain model for Scenario B."""

from task_b.domain.faults import (
    COMPONENT_TO_FAULT,
    FAULT_IDS,
    FAULT_OPTIONS,
    FAULT_TO_COMPONENT,
    OPEN_CIRCUIT_FAULTS,
    SHORT_CIRCUIT_FAULTS,
)
from task_b.domain.measurement_protocol import (
    get_benchmark_protocol,
    identify_component_name,
    parse_instrument_key,
)
from task_b.domain.rule_engine import (
    CIRCUIT_ARCHITECTURES,
    CIRCUIT_TYPES,
    RULE_GUIDE_TEXT,
    CircuitTopology,
    MeasurementCondition,
    compute_ground_truth_history,
    enrich_event,
    get_default_circuit_type,
    get_rule_guide_text,
    get_topology,
    supported_faults_for_condition,
    validate_challenge_against_rules,
)

__all__ = [
    "COMPONENT_TO_FAULT",
    "CIRCUIT_ARCHITECTURES",
    "CIRCUIT_TYPES",
    "CircuitTopology",
    "FAULT_IDS",
    "FAULT_OPTIONS",
    "FAULT_TO_COMPONENT",
    "MeasurementCondition",
    "OPEN_CIRCUIT_FAULTS",
    "RULE_GUIDE_TEXT",
    "SHORT_CIRCUIT_FAULTS",
    "compute_ground_truth_history",
    "enrich_event",
    "get_benchmark_protocol",
    "get_default_circuit_type",
    "get_rule_guide_text",
    "get_topology",
    "identify_component_name",
    "parse_instrument_key",
    "supported_faults_for_condition",
    "validate_challenge_against_rules",
]
