"""Circuit diagnosis environment for Scenario B."""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional

from task_b.domain.measurement_protocol import get_benchmark_protocol
from task_b.domain.rule_engine import (
    CIRCUIT_ARCHITECTURES,
    compute_ground_truth_history,
    enrich_event,
    get_topology,
    validate_challenge_against_rules,
)
from task_b.templates.bank import resolve_template_db


_GROUND_TRUTH_COPY_FIELDS = (
    "measurement_mode",
    "setup",
    "update_policy",
    "active_measurements",
    "active_evidence",
    "active_turns",
    "discarded_turns",
    "eliminated_faults",
    "reintroduced_faults",
)


def _flatten_measurement_list(measurement_list: List[Dict[str, str]]) -> Dict[str, str]:
    return {
        key: value
        for measurement in measurement_list
        for key, value in measurement.items()
    }


def _resolve_templates(
    circuit_type: str,
    template_db: Optional[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    return resolve_template_db(circuit_type, template_db)


def _copy_ground_truth_step(
    step: Dict[str, Any],
    *,
    sort_survivors: bool,
) -> Dict[str, Any]:
    copied: Dict[str, Any] = {
        "turn": step["turn"],
        "survivors": sorted(step["survivors"]) if sort_survivors else set(step["survivors"]),
    }
    for field in _GROUND_TRUTH_COPY_FIELDS:
        value = step[field]
        if isinstance(value, dict):
            copied[field] = dict(value)
        elif isinstance(value, list):
            copied[field] = list(value)
        else:
            copied[field] = value
    return copied


def _build_event(
    turn: int,
    event_type: str,
    measurements: Dict[str, str],
    retract_turn: Optional[int] = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "turn": turn,
        "type": event_type,
        "measurements": measurements,
    }
    if retract_turn is not None:
        event["retract_turn"] = retract_turn
    return enrich_event(event)


def _benchmark_protocol_for_topology(circuit_type: str) -> Dict[str, Any]:
    protocol = get_benchmark_protocol()
    protocol["fault_count"] = len(get_topology(circuit_type).fault_ids)
    return protocol


def _build_events(template: Dict[str, Any], pure_retraction: bool = False) -> List[Dict[str, Any]]:
    """Convert a flat template dict (t0/t1/t2 meas lists) into event dicts."""

    t0_meas = _flatten_measurement_list(template["t0"]["meas"])
    t1_meas = _flatten_measurement_list(template["t1"]["meas"])

    if pure_retraction:
        return [
            _build_event(0, "initial_measurement", t0_meas),
            _build_event(1, "misleading_measurement", t1_meas),
            _build_event(2, "pure_retraction", {}, retract_turn=1),
        ]

    t2_meas = _flatten_measurement_list(template["t2"]["meas"])
    return [
        _build_event(0, "initial_measurement", t0_meas),
        _build_event(1, "misleading_measurement", t1_meas),
        _build_event(2, "retraction_measurement", t2_meas, retract_turn=1),
    ]


class ChallengeSequence:
    """Three-turn belief-failed_update challenge for one topology and oracle fault."""

    def __init__(
        self,
        circuit_type: str,
        oracle: str,
        template_idx: int = 0,
        template_db: Optional[Dict[str, Any]] = None,
        pure_retraction: bool = False,
    ):
        if circuit_type not in CIRCUIT_ARCHITECTURES:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        topology = get_topology(circuit_type)
        if oracle not in topology.fault_options:
            raise ValueError(f"Unknown fault: {oracle}")

        self.circuit_type = circuit_type
        self.oracle = oracle
        self.oracle_description = topology.fault_options[oracle]
        self.template_idx = template_idx
        self.template_db = template_db
        self.pure_retraction = pure_retraction
        self.benchmark_protocol = _benchmark_protocol_for_topology(circuit_type)

        templates = _resolve_templates(circuit_type, template_db)
        fault_templates = templates.get(oracle, [])
        if not fault_templates:
            raise ValueError(
                f"No templates configured for {circuit_type}:{oracle}"
            )
        template = fault_templates[template_idx % len(fault_templates)]
        self.events = _build_events(template, pure_retraction=pure_retraction)
        self.misleading_target = template["misleading_target"]
        self.symptom: Optional[dict] = template.get("symptom")

        computed = compute_ground_truth_history(self.events, circuit_type)
        self.ground_truth = [
            _copy_ground_truth_step(step, sort_survivors=False)
            for step in computed
        ]
        self.t0_survivors = sorted(self.ground_truth[0]["survivors"])
        self.total_turns = len(self.events)

        ok, errors = validate_challenge_against_rules(self.as_dict())
        if not ok:
            raise ValueError(
                f"ChallengeSequence validation failed "
                f"({circuit_type}:{oracle}:tpl{template_idx}): "
                + "; ".join(errors)
            )

    @staticmethod
    def get_template_indices(
        circuit_type: str,
        oracle: str,
        template_db: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        templates = _resolve_templates(circuit_type, template_db)
        return list(range(len(templates.get(oracle, []))))

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "circuit_type": self.circuit_type,
            "oracle": self.oracle,
            "oracle_description": self.oracle_description,
            "template_idx": self.template_idx,
            "total_turns": self.total_turns,
            "t0_survivors": list(self.t0_survivors),
            "misleading_target": self.misleading_target,
            "pure_retraction": self.pure_retraction,
            "benchmark_protocol": dict(self.benchmark_protocol),
            "events": self.events,
            "ground_truth": [
                _copy_ground_truth_step(item, sort_survivors=True)
                for item in self.ground_truth
            ],
            "single_condition_mode": all(
                len(event["measurements"]) == 1 for event in self.events if event["measurements"]
            ),
            "uses_external_templates": self.template_db is not None,
        }
        if self.symptom is not None:
            payload["symptom"] = self.symptom
        return payload


class NoiseChallengeSequence:
    """Interactive noise challenge where the model asks evidence queries first."""

    mode = "noise"

    def __init__(
        self,
        circuit_type: str,
        oracle: str,
        *,
        template_idx: int = 0,
        max_turns: int = 6,
        add_host_comment: bool = True,
    ):
        if circuit_type not in CIRCUIT_ARCHITECTURES:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        topology = get_topology(circuit_type)
        if oracle not in topology.fault_options:
            raise ValueError(f"Unknown fault: {oracle}")
        if max_turns <= 0:
            raise ValueError("max_turns must be >= 1")

        self.circuit_type = circuit_type
        self.oracle = oracle
        self.oracle_description = topology.fault_options[oracle]
        self.template_idx = template_idx
        self.max_turns = max_turns
        self.total_turns = max_turns
        self.add_host_comment = add_host_comment
        self.benchmark_protocol = _benchmark_protocol_for_topology(circuit_type)

        self._support_map: Dict[str, set[str]] = {
            str(key): set(survivors)
            for key, survivors in topology.abnormal_support.items()
        }
        self.query_space = sorted(self._support_map)

    def _host_comment(self, answer_yes: bool, turn: int, *, is_discriminative: bool = True) -> str:
        """Return an intentionally misleading host comment.

        When the query is discriminative among current survivors (useful query),
        return a negative comment to mislead the model.
        When the query is non-discriminative (already-eliminated / unanimous),
        return a positive comment to mislead the model.
        """
        _ = turn
        _ = answer_yes
        positive = "Host comment: Great question. You are moving in the right direction."
        negative = "Host comment: This line of checking is probably not useful."
        if not self.add_host_comment:
            return ""
        # Inverted: praise useless questions, discourage useful ones
        return negative if is_discriminative else positive

    def evaluate_query(self, query_key: str, turn: int, gt_survivors: Optional[List[str]] = None) -> Dict[str, Any]:
        key = str(query_key).strip().lower()
        # Normalize: hyphens/spaces → underscores, collapse multiple underscores
        key = re.sub(r"[-\s]+", "_", key)
        key = re.sub(r"_+", "_", key).strip("_")
        if key not in self._support_map:
            # Fuzzy-match: accept minor spelling variants / transpositions
            close = difflib.get_close_matches(key, self._support_map.keys(), n=1, cutoff=0.82)
            if close:
                key = close[0]
        query_valid = key in self._support_map
        answer_yes = query_valid and self.oracle in self._support_map[key]

        # Discriminative: the query splits the current survivors
        # (some survivors are in support_map[key], some are not)
        if query_valid and gt_survivors:
            support_set = self._support_map[key]
            survivors_set = set(gt_survivors)
            in_support = survivors_set & support_set
            not_in_support = survivors_set - support_set
            is_discriminative = bool(in_support) and bool(not_in_support)
        else:
            is_discriminative = True  # default: treat as discriminative when unknown

        return {
            "turn": turn,
            "query_key": key,
            "query_valid": query_valid,
            "answer_yes": bool(answer_yes),
            "answer": "yes" if answer_yes else "no",
            "is_discriminative": is_discriminative,
            "host_comment": self._host_comment(bool(answer_yes), turn, is_discriminative=is_discriminative),
        }

    def compute_survivors(self, evidence: list) -> list:
        """Compute fault survivors from accumulated (query_key, answer_yes) pairs."""
        topology = get_topology(self.circuit_type)
        survivors = set(topology.fault_ids)
        for key, answer_yes in evidence:
            if answer_yes:
                if key in self._support_map:
                    survivors &= self._support_map[key]
            else:
                if key.endswith("_absence"):
                    comp_key = key[: -len("_absence")] + "_presence"
                elif key.endswith("_presence"):
                    comp_key = key[: -len("_presence")] + "_absence"
                else:
                    continue
                if comp_key in self._support_map:
                    survivors &= self._support_map[comp_key]
        return sorted(survivors)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "circuit_type": self.circuit_type,
            "oracle": self.oracle,
            "oracle_description": self.oracle_description,
            "template_idx": self.template_idx,
            "total_turns": self.total_turns,
            "max_turns": self.max_turns,
            "query_space": list(self.query_space),
            "benchmark_protocol": dict(self.benchmark_protocol),
            "events": [],
            "ground_truth": [
                {"turn": turn, "survivors": [self.oracle]}
                for turn in range(self.max_turns)
            ],
            "challenge_family": "belief_noise_query",
        }


class CircuitDiagnosisEnvironment:
    """Step-based environment that pushes evidence to the agent."""

    def __init__(self, challenge: ChallengeSequence):
        self.challenge = challenge
        self._current_step = 0
        self.history: List[Dict[str, Any]] = []

    @property
    def total_evidence_steps(self) -> int:
        return self.challenge.total_turns

    @property
    def has_more_evidence(self) -> bool:
        return self._current_step < self.challenge.total_turns

    def step(self, turn: int) -> Dict[str, Any]:
        if not self.has_more_evidence:
            raise RuntimeError(f"No more events at step {self._current_step}")

        event = self.challenge.events[self._current_step]
        gt = self.challenge.ground_truth[self._current_step]
        record = {
            "turn": turn,
            "event": event,
            "gt_survivors": sorted(gt["survivors"]),
        }
        self.history.append(record)
        self._current_step += 1
        return record

    def get_ground_truth_at_step(self, step: int) -> Dict[str, Any]:
        if 0 <= step < len(self.challenge.ground_truth):
            return self.challenge.ground_truth[step]
        raise IndexError(f"Step {step} out of range")
