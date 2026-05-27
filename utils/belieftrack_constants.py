"""Canonical BeliefTrack / CBM vocabulary (paper-aligned naming)."""

from __future__ import annotations

from typing import Dict

# Challenge types (datasets D_stay, D_update, D_iso)
FAILED_STAY = "failed_stay"  # FailedStay (FSR)
FAILED_UPDATE = "failed_update"  # FailedUpdate / correction (FUR)
FAILED_ISOLATION = "failed_isolation"  # FailedIsolation (FIR)
CBM_CHALLENGE_TYPES = (FAILED_STAY, FAILED_UPDATE, FAILED_ISOLATION)

# Eval / export sample categories
ORACLE_MATCH = "oracle_match"  # Hypothesis set equals oracle S*_t on scored turns
BELIEF_FAILURE = "belief_failure"  # Target failure manifested (counts toward FSR/FUR/FIR)
INSUFFICIENT_CAPABILITY = "insufficient_capability"  # Prefix / capability filter (unchanged)
PARSE_ERROR = "parse_error"
CBM_EVAL_CATEGORIES = (
    INSUFFICIENT_CAPABILITY,
    BELIEF_FAILURE,
    ORACLE_MATCH,
    PARSE_ERROR,
)

FAILURE_METRIC_BY_CHALLENGE: Dict[str, str] = {
    FAILED_STAY: "FSR",
    FAILED_UPDATE: "FUR",
    FAILED_ISOLATION: "FIR",
}

def normalize_cbm_challenge_type(value: str) -> str:
    key = (value or "").strip().lower()
    if key not in CBM_CHALLENGE_TYPES:
        raise ValueError(f"Unknown CBM challenge type: {value!r}")
    return key


def normalize_cbm_eval_category(value: str) -> str:
    key = (value or "").strip().lower()
    if key not in CBM_EVAL_CATEGORIES:
        raise ValueError(f"Unknown CBM eval category: {value!r}")
    return key


def cbm_failure_metric(challenge_type: str) -> str | None:
    return FAILURE_METRIC_BY_CHALLENGE.get(normalize_cbm_challenge_type(challenge_type))


def canonicalize_category_counts(counts: Dict[str, int]) -> Dict[str, int]:
    out = {category: 0 for category in CBM_EVAL_CATEGORIES}
    for category, count in counts.items():
        normalized = normalize_cbm_eval_category(str(category))
        out[normalized] = out.get(normalized, 0) + int(count)
    return out


def percentages_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(int(value) for value in counts.values())
    return {
        category: round(int(value) / max(total, 1) * 100, 2)
        for category, value in counts.items()
    }
