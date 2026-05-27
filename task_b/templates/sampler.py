"""Automatic template sampler for Scenario B."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from task_b.domain.rule_engine import get_topology, supported_faults_for_condition


@dataclass(frozen=True)
class MeasurementCandidate:
    circuit_type: str
    mode: str
    measurements: Tuple[Tuple[str, str], ...]
    survivors: Tuple[str, ...]

    @property
    def condition_count(self) -> int:
        return len(self.measurements)

    def as_mapping(self) -> Dict[str, str]:
        return dict(self.measurements)

    @property
    def signature(self) -> Tuple[Tuple[str, str], ...]:
        return self.measurements


@dataclass(frozen=True)
class FailedUpdateTemplateCase:
    oracle: str
    misleading_target: str
    t0: MeasurementCandidate
    t1: MeasurementCandidate
    t2: MeasurementCandidate
    t0_survivors: Tuple[str, ...]
    t1_survivors: Tuple[str, ...]
    score: Tuple[Any, ...]

    @property
    def signature(self) -> Tuple[Any, ...]:
        return (
            self.oracle,
            self.misleading_target,
            self.t0.signature,
            self.t1.signature,
            self.t2.signature,
        )

    def as_template_case(self) -> Dict[str, Any]:
        return {
            "t0": {"meas": [self.t0.as_mapping()]},
            "misleading_target": self.misleading_target,
            "t1": {"meas": [self.t1.as_mapping()]},
            "t2": {"meas": [self.t2.as_mapping()]},
            "auto_generated": {
                "t0_survivors": list(self.t0_survivors),
                "t1_survivors": list(self.t1_survivors),
                "t2_survivors": [self.oracle],
                "measurement_modes": {
                    "t0": self.t0.mode,
                    "t1": self.t1.mode,
                    "t2": self.t2.mode,
                },
            },
        }


def _mode_rank(mode: str) -> int:
    return 0 if mode == "powered" else 1


def _build_powered_atomic_candidates(circuit_type: str) -> List[MeasurementCandidate]:
    topology = get_topology(circuit_type)
    candidates: List[MeasurementCandidate] = []
    for specific_key, survivors in sorted(topology.abnormal_support.items()):
        instrument, value = specific_key.rsplit("_", 1)
        candidates.append(
            MeasurementCandidate(
                circuit_type=circuit_type,
                mode="powered",
                measurements=((instrument, value),),
                survivors=tuple(sorted(survivors)),
            )
        )
    return candidates


def _build_isolation_atomic_candidates(
    circuit_type: str,
    include_healthy: bool = True,
) -> List[MeasurementCandidate]:
    topology = get_topology(circuit_type)
    values = ["open", "short"]
    if include_healthy:
        values.append("healthy")

    candidates: List[MeasurementCandidate] = []
    for component in topology.components:
        for value in values:
            measurements = ((f"isolation_{component}", value),)
            survivors = supported_faults_for_condition(dict(measurements), circuit_type)
            candidates.append(
                MeasurementCandidate(
                    circuit_type=circuit_type,
                    mode="isolation",
                    measurements=measurements,
                    survivors=tuple(sorted(survivors)),
                )
            )
    return candidates


def enumerate_measurement_candidates(
    circuit_type: str,
    *,
    max_powered_conditions: int = 2,
    include_isolation_healthy: bool = True,
) -> List[MeasurementCandidate]:
    powered_atomic = _build_powered_atomic_candidates(circuit_type)
    topology = get_topology(circuit_type)

    combined_powered: List[MeasurementCandidate] = []
    seen_signatures = set()
    for size in range(1, max_powered_conditions + 1):
        for combo in combinations(powered_atomic, size):
            merged: Dict[str, str] = {}
            survivors = set(topology.fault_ids)
            valid = True
            for candidate in combo:
                for key, value in candidate.measurements:
                    if key in merged:
                        valid = False
                        break
                    merged[key] = value
                if not valid:
                    break
                survivors &= set(candidate.survivors)

            if not valid:
                continue

            signature = tuple(sorted(merged.items()))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            combined_powered.append(
                MeasurementCandidate(
                    circuit_type=circuit_type,
                    mode="powered",
                    measurements=signature,
                    survivors=tuple(sorted(survivors)),
                )
            )

    combined_powered.sort(
        key=lambda candidate: (
            _mode_rank(candidate.mode),
            candidate.condition_count,
            len(candidate.survivors),
            candidate.signature,
        )
    )
    return combined_powered


def _case_score(
    t0: MeasurementCandidate,
    t1: MeasurementCandidate,
    t2: MeasurementCandidate,
    t0_survivors: Sequence[str],
    t1_survivors: Sequence[str],
) -> Tuple[Any, ...]:
    return (
        _mode_rank(t1.mode),
        _mode_rank(t2.mode),
        t0.condition_count + t1.condition_count + t2.condition_count,
        abs(len(t0_survivors) - 2),
        len(t1_survivors),
        len(t0_survivors),
        t0.signature,
        t1.signature,
        t2.signature,
    )


def enumerate_failed_update_template_cases(
    circuit_type: str,
    oracle: str,
    *,
    max_powered_conditions: int = 2,
    include_isolation_healthy: bool = True,
    t0_min_survivors: int = 2,
    t0_max_survivors: Optional[int] = None,
) -> List[FailedUpdateTemplateCase]:
    candidates = enumerate_measurement_candidates(
        circuit_type,
        max_powered_conditions=max_powered_conditions,
        include_isolation_healthy=include_isolation_healthy,
    )
    
    cases: List[FailedUpdateTemplateCase] = []
    seen_signatures = set()
    t0_candidates = [
        candidate
        for candidate in candidates
        if candidate.mode == "powered"
        and oracle in candidate.survivors
        and len(candidate.survivors) >= t0_min_survivors
        and (t0_max_survivors is None or len(candidate.survivors) <= t0_max_survivors)
    ]

    for t0 in t0_candidates:
        t0_survivors = set(t0.survivors)
        misleading_targets = sorted(t0_survivors - {oracle})
        if not misleading_targets:
            continue

        for misleading_target in misleading_targets:
            for t1 in candidates:
                turn1_survivors = sorted(t0_survivors & set(t1.survivors))
                if not turn1_survivors or oracle in turn1_survivors:
                    continue
                if misleading_target not in turn1_survivors:
                    continue

                for t2 in candidates:
                    final_survivors = t0_survivors & set(t2.survivors)
                    if final_survivors != {oracle}:
                        continue

                    case = FailedUpdateTemplateCase(
                        oracle=oracle,
                        misleading_target=misleading_target,
                        t0=t0,
                        t1=t1,
                        t2=t2,
                        t0_survivors=tuple(sorted(t0_survivors)),
                        t1_survivors=tuple(turn1_survivors),
                        score=_case_score(
                            t0,
                            t1,
                            t2,
                            sorted(t0_survivors),
                            turn1_survivors,
                        ),
                    )
                    if case.signature in seen_signatures:
                        continue
                    seen_signatures.add(case.signature)
                    cases.append(case)

    cases.sort(key=lambda case: case.score)
    return cases


def _select_diverse_cases(
    cases: Sequence[FailedUpdateTemplateCase],
    max_cases: int,
) -> List[FailedUpdateTemplateCase]:
    if max_cases <= 0 or not cases:
        return []
    if len(cases) <= max_cases:
        return list(cases)

    remaining = list(cases)
    selected: List[FailedUpdateTemplateCase] = []
    used_t0 = set()
    used_t1 = set()
    used_t2 = set()
    used_targets = set()

    while remaining and len(selected) < max_cases:
        best_index = 0
        best_key: Optional[Tuple[Any, ...]] = None
        for index, case in enumerate(remaining):
            novelty_key = (
                1 if case.misleading_target in used_targets else 0,
                1 if case.t0.signature in used_t0 else 0,
                1 if case.t1.signature in used_t1 else 0,
                1 if case.t2.signature in used_t2 else 0,
                case.score,
            )
            if best_key is None or novelty_key < best_key:
                best_index = index
                best_key = novelty_key

        chosen = remaining.pop(best_index)
        selected.append(chosen)
        used_targets.add(chosen.misleading_target)
        used_t0.add(chosen.t0.signature)
        used_t1.add(chosen.t1.signature)
        used_t2.add(chosen.t2.signature)

    return selected


def generate_templates_for_circuit(
    circuit_type: str,
    *,
    max_cases_per_fault: int = 12,
    max_powered_conditions: int = 2,
    include_isolation_healthy: bool = True,
    t0_min_survivors: int = 2,
    t0_max_survivors: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    topology = get_topology(circuit_type)
    templates: Dict[str, List[Dict[str, Any]]] = {fault_id: [] for fault_id in topology.fault_ids}
    for fault_id in topology.fault_ids:
        cases = enumerate_failed_update_template_cases(
            circuit_type,
            fault_id,
            max_powered_conditions=max_powered_conditions,
            include_isolation_healthy=include_isolation_healthy,
            t0_min_survivors=t0_min_survivors,
            t0_max_survivors=t0_max_survivors,
        )
        selected = _select_diverse_cases(cases, max_cases=max_cases_per_fault)
        templates[fault_id] = [case.as_template_case() for case in selected]
    return templates


def build_generated_templates_payload(
    circuit_type: str,
    templates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_cases_per_fault: int,
    max_powered_conditions: int,
    include_isolation_healthy: bool,
    t0_min_survivors: int,
    t0_max_survivors: Optional[int],
    verified: bool,
) -> Dict[str, Any]:
    counts = {fault_id: len(list(cases)) for fault_id, cases in templates.items()}
    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generator": "task_b.templates.sampler",
            "circuit_type": circuit_type,
            "challenge_family": "belief_failed_update_retraction",
            "verified": verified,
            "search_config": {
                "max_cases_per_fault": max_cases_per_fault,
                "max_powered_conditions": max_powered_conditions,
                "include_isolation_healthy": include_isolation_healthy,
                "t0_min_survivors": t0_min_survivors,
                "t0_max_survivors": t0_max_survivors,
            },
            "case_counts": counts,
            "total_cases": sum(counts.values()),
        },
        "templates": dict(templates),
    }
