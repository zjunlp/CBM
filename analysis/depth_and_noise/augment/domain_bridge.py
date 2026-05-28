"""Thin wrappers around task_a / task_b domain helpers."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from task_a.experiments.belief_stats import (
    MAX_SAMPLE_ATTEMPTS,
    TRIPLE_RANGE,
    Triple,
    compute_golden_hypotheses,
    triple_text,
)
from task_a.core.rules import get_rule as get_extended_rule

_RULE_ID_RE = re.compile(r'- "([^"]+)":')


def parse_candidate_rules_from_system_prompt(system_prompt: str) -> List[str]:
    return _RULE_ID_RE.findall(system_prompt)


def build_rules_map(candidate_names: List[str]) -> Dict[str, Any]:
    return {name: get_extended_rule(name) for name in candidate_names}


def sample_redundant_triple(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    target_survivors: Set[str],
    compatible_survivors: Optional[Set[str]] = None,
) -> Optional[Triple]:
    lo, hi = TRIPLE_RANGE
    compatible_survivors = compatible_survivors or set()
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        triple = (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))
        result = "YES" if rules[oracle].validate(triple) else "NO"
        result_bool = result == "YES"
        if any(rules[name].validate(triple) != result_bool for name in compatible_survivors if name in rules):
            continue
        survivors = compute_golden_hypotheses(rules, active_evidence + [(triple, result)])
        if survivors == target_survivors:
            return triple
    return None


def oracle_label(rules: Dict[str, Any], oracle: str, triple: Triple) -> str:
    return "YES" if rules[oracle].validate(triple) else "NO"


def format_triple(triple: Triple) -> str:
    return triple_text(triple)
