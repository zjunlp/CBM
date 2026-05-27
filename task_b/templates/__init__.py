"""Template management for Scenario B."""

from task_b.templates.bank import (
    get_default_template_bank_path,
    load_default_templates,
    resolve_template_db,
)
from task_b.templates.sampler import (
    build_generated_templates_payload,
    enumerate_failed_update_template_cases,
    enumerate_measurement_candidates,
    generate_templates_for_circuit,
)
from task_b.templates.verify import (
    build_verified_output_payload,
    extract_templates_payload,
    verify_challenge_dict,
    verify_templates,
)

__all__ = [
    "build_generated_templates_payload",
    "build_verified_output_payload",
    "enumerate_failed_update_template_cases",
    "enumerate_measurement_candidates",
    "extract_templates_payload",
    "generate_templates_for_circuit",
    "get_default_template_bank_path",
    "load_default_templates",
    "resolve_template_db",
    "verify_challenge_dict",
    "verify_templates",
]
