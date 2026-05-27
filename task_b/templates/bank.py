"""Default template-bank loader for Scenario B."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


_DEFAULT_TEMPLATE_FILES = {
    "series_all": "series_all_default_verified.json",
    "parallel_r12_series": "parallel_r12_series_default_verified.json",
}


def _generated_templates_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "generated_templates"


def get_default_template_bank_path(circuit_type: str) -> Path:
    try:
        filename = _DEFAULT_TEMPLATE_FILES[circuit_type]
    except KeyError as exc:
        available = sorted(_DEFAULT_TEMPLATE_FILES)
        raise ValueError(
            f"Unknown circuit type '{circuit_type}'. Available: {available}"
        ) from exc
    return _generated_templates_dir() / filename


def _extract_templates(payload: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    templates = payload.get("templates", payload)
    if not isinstance(templates, dict):
        raise ValueError("template payload must contain a dict-valued 'templates' key")
    return dict(templates)


def _select_template_payload(
    circuit_type: str,
    template_db: Mapping[str, Any],
) -> Mapping[str, Any]:
    templates_by_circuit = template_db.get("templates_by_circuit")
    if isinstance(templates_by_circuit, Mapping):
        payload = templates_by_circuit.get(circuit_type)
        if payload is None:
            raise ValueError(
                f"No template bank configured for circuit type '{circuit_type}'"
            )
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Template bank for circuit type '{circuit_type}' must be a mapping"
            )
        return payload

    payload = template_db.get(circuit_type)
    if isinstance(payload, Mapping):
        return payload

    return template_db


@lru_cache(maxsize=None)
def _load_template_bank_file(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Template bank not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Template bank payload must be a dict: {path}")
    return dict(payload)


def load_default_templates(circuit_type: str) -> Dict[str, List[Dict[str, Any]]]:
    path = get_default_template_bank_path(circuit_type)
    payload = _load_template_bank_file(str(path))
    return copy.deepcopy(_extract_templates(payload))


def resolve_template_db(
    circuit_type: str,
    template_db: Optional[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    if template_db is not None:
        if not isinstance(template_db, dict):
            raise ValueError("template_db must be a dict when provided")
        selected = _select_template_payload(circuit_type, template_db)
        templates = _extract_templates(selected)
        return {fault_id: list(cases) for fault_id, cases in templates.items()}
    return load_default_templates(circuit_type)
