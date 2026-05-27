"""Validate failed_isolation analysis source cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def normalize_failed_isolation_source_case(path: Path, payload: Dict[str, Any], task: str) -> Dict[str, Any]:
    if str(payload.get("cbm_challenge_type", "")).lower() != "failed_isolation":
        raise ValueError(f"{path} must use cbm_challenge_type=failed_isolation")
    if not payload.get("turns"):
        raise ValueError(f"{path} must contain turns")
    out = dict(payload)
    out["task"] = task
    return out
