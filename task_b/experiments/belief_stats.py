"""Scenario B belief statistics for failed_update and failed_stay.

Execution model:
  - one vLLM instance across all requested GPUs via tensor parallelism
  - prepare all task/template/run/repeat sessions first
  - run one global vLLM batch per conversation turn
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from tqdm.auto import tqdm

from task_b.domain.faults import FAULT_IDS
from task_b.domain.measurement_protocol import get_benchmark_protocol
from task_b.domain.rule_engine import (
    CIRCUIT_TYPES,
    compute_ground_truth_history,
    enrich_event,
    get_topology,
    supported_faults_for_condition,
    validate_challenge_against_rules,
)
from task_b.experiments.challenge_metrics import (
    CATEGORIES,
    build_category_dirs,
    build_stats_report,
    classify_trajectory,
    compact_case_export,
    sanitize_trajectory_for_export,
)
from task_b.experiments.batched_runner import run_sessions_batched
from task_b.templates.sampler import (
    MeasurementCandidate,
    enumerate_measurement_candidates,
)
from utils.io import save_json
from utils.llm_backend import APIBackend, VLLMBackend


MAX_TOKENS = 2048
MODES = ("failed_update", "failed_stay", "noise", "failed_isolation")
PAIRED_BRANCH_FAULTS = {
    "D": "F",
    "F": "D",
    "E": "G",
    "G": "E",
    "H": "D",
    "I": "G",
}
PAIRED_BRANCH_FAULTS_BY_CIRCUIT = {
    "parallel_series_pairs": {
        "D": "F",
        "F": "D",
        "E": "G",
        "G": "E",
        "H": "J",
        "J": "H",
        "I": "K",
        "K": "I",
    },
}


def _paired_branch_fault(circuit_type: str, oracle: str) -> Optional[str]:
    return PAIRED_BRANCH_FAULTS_BY_CIRCUIT.get(circuit_type, PAIRED_BRANCH_FAULTS).get(oracle)


def _fault_component(circuit_type: str, fault_id: Optional[str]) -> str:
    if not fault_id:
        return ""
    return get_topology(circuit_type).fault_to_component.get(fault_id, "")


def _multi_candidate_target_set(circuit_type: str, oracle: str) -> Set[str]:
    if circuit_type == "parallel_series_pairs":
        if oracle in {"D", "F"}:
            return {"D", "F"}
        if oracle in {"H", "J"}:
            return {"H", "J"}
        if oracle == "E":
            return {"E", "F"}
        if oracle == "G":
            return {"D", "G"}
        if oracle == "I":
            return {"I", "J"}
        if oracle == "K":
            return {"H", "K"}
    if circuit_type == "parallel_r12_series":
        target_sets = {
            "C": {"C", "E"},
            "D": {"D", "G"},
            "E": {"E", "F"},
            "F": {"E", "F"},
            "G": {"D", "G"},
        }
        if oracle in target_sets:
            return target_sets[oracle]
    paired_fault = _paired_branch_fault(circuit_type, oracle)
    return {oracle, paired_fault} if paired_fault else {oracle}
INERTIA_CASE_STRATEGIES = (
    "default",
    "belief_prone_correction_v2",
)

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


def aggregate_repeat_categories(
    per_run_categories: Sequence[str],
    threshold: float = 0.5,
) -> str:
    if not per_run_categories:
        return "insufficient_capability"

    category, count = Counter(per_run_categories).most_common(1)[0]
    return category if count / len(per_run_categories) > threshold else "unstable"


def _parse_gpu_ids(raw_gpus: str) -> List[int]:
    gpu_ids = [int(gpu.strip()) for gpu in raw_gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpu_ids


def _load_template_db(templates_json: str) -> Optional[Dict[str, Any]]:
    if not templates_json:
        return None
    with open(templates_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("templates", payload)


def _load_template_index_filter(path: str) -> Dict[str, Set[int]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError("--only-template-indices-json must contain a JSON object")

    result: Dict[str, Set[int]] = {}
    for key, values in payload.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Template index filter for {key!r} must be a list")
        result[str(key)] = {int(value) for value in values}
    return result


def _allowed_template_indices(
    *,
    circuit_type: str,
    fault: str,
    args: argparse.Namespace,
) -> Optional[Set[int]]:
    filters = getattr(args, "only_template_indices", None) or {}
    task_label = f"{circuit_type}:{fault}"
    if task_label in filters:
        return set(filters[task_label])
    if fault in filters:
        return set(filters[fault])
    return None


def _default_prepare_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(16, cpu_count))


def _thread_map_ordered(
    fn: Any,
    items: Sequence[Any],
    *,
    desc: str,
    workers: int,
) -> List[Any]:
    if workers <= 1 or len(items) <= 1:
        return [
            fn(item)
            for item in tqdm(
                items,
                desc=desc,
                unit="point",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(fn, items),
                total=len(items),
                desc=f"{desc} ({workers} threads)",
                unit="point",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        )


def _validate_failed_update_challenge(challenge_dict: Dict[str, Any]) -> None:
    from task_b.templates.verify import verify_challenge_dict

    ok, errors = verify_challenge_dict(
        challenge_dict,
        require_single_condition=False,
    )
    if not ok:
        raise ValueError(f"Invalid challenge: {errors}")


def _measurement_signature(measurements: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in measurements.items()))


def _is_powered_measurement_key(key: str) -> bool:
    return not (key.startswith("isolation_") or key.startswith("ohmmeter_"))


def _challenge_uses_only_powered_measurements(challenge_dict: Mapping[str, Any]) -> bool:
    for event in challenge_dict.get("events", []):
        if not isinstance(event, Mapping):
            return False
        measurements = event.get("measurements", {})
        if not isinstance(measurements, Mapping):
            return False
        if any(not _is_powered_measurement_key(str(key)) for key in measurements):
            return False
    return True


def _failed_update_prefix_complexity(
    t0_sig: Tuple[Tuple[str, str], ...],
    t1_sig: Tuple[Tuple[str, str], ...],
) -> int:
    difficult_keys = {"ammeter_r1", "ammeter_r3", "ammeter_r2", "voltmeter_r1", "voltmeter_r3", "voltmeter_r2"}
    keys = {key for key, _ in t0_sig + t1_sig}
    branch_read_count = sum(1 for key in keys if key in difficult_keys)
    return (
        max(0, len(t0_sig) - 1)
        + max(0, len(t1_sig) - 1)
        + max(0, branch_read_count - 2)
    )


def _failed_update_template_score_v2(challenge_dict: Mapping[str, Any]) -> Tuple[Any, ...]:
    oracle = str(challenge_dict.get("oracle", ""))
    target = str(challenge_dict.get("misleading_target", ""))
    circuit_type = str(challenge_dict.get("circuit_type") or "parallel_r123_series")
    events = list(challenge_dict.get("events", []))
    ground_truth = list(challenge_dict.get("ground_truth", []))
    if len(events) < 3 or len(ground_truth) < 3:
        return (999,)

    gt0 = set(ground_truth[0].get("survivors", []))
    gt1 = set(ground_truth[1].get("survivors", []))
    gt2 = set(ground_truth[2].get("survivors", []))
    t0_sig = _measurement_signature(events[0].get("measurements", {}))
    t1_sig = _measurement_signature(events[1].get("measurements", {}))
    t2_sig = _measurement_signature(events[2].get("measurements", {}))

    t2_active_sets = [
        set(item.get("survivors", []))
        for item in ground_truth[2].get("active_evidence", [])
        if isinstance(item, Mapping)
    ]
    t2_single_survivors = t2_active_sets[-1] if t2_active_sets else set()

    pair = (oracle, target)
    if circuit_type == "parallel_series_pairs":
        core_pairs = {
            ("D", "G"),
            ("E", "F"),
            ("F", "E"),
            ("G", "D"),
            ("G", "E"),
            ("H", "K"),
            ("I", "J"),
            ("J", "H"),
            ("J", "I"),
            ("K", "H"),
            ("K", "I"),
        }
        medium_pairs = {
            ("D", "F"),
            ("E", "G"),
            ("F", "D"),
            ("H", "J"),
            ("I", "K"),
        }
    else:
        core_pairs = {
            ("C", "E"),
            ("C", "G"),
            ("D", "G"),
            ("E", "F"),
            ("F", "E"),
            ("G", "E"),
            ("E", "G"),
            ("B", "A"),
        }
        medium_pairs = {
            ("B", "D"),
            ("D", "B"),
            ("G", "D"),
            ("A", "E"),
            ("A", "G"),
        }
    sibling = _paired_branch_fault(circuit_type, oracle)
    target_sibling = _paired_branch_fault(circuit_type, target)
    sibling_pair = bool(
        sibling == target
        or target_sibling == oracle
        or {oracle, target} in ({"C", "E"}, {"C", "G"})
    )
    t2_keeps_target = bool(target and target in t2_single_survivors)
    t2_keeps_sibling = bool(sibling and sibling in t2_single_survivors)
    t2_broad_or_confusing = len(t2_single_survivors) >= 2 or t2_keeps_target or t2_keeps_sibling
    prefix_complexity = _failed_update_prefix_complexity(t0_sig, t1_sig)

    prefix_single_measurements = len(t0_sig) == 1 and len(t1_sig) == 1
    target_safe = bool(
        gt1 == {target}
        or (
            oracle not in gt1
            and target in gt1
            and len(gt1) <= 3
            and prefix_single_measurements
        )
    )
    t0_reasonable = 2 <= len(gt0) <= 4
    t2_not_too_complex = len(t2_sig) <= 2

    if circuit_type == "parallel_series_pairs":
        oracle_penalty = {
            "D": 0,
            "F": 0,
            "H": 0,
            "J": 0,
            "E": 1,
            "G": 1,
            "I": 1,
            "K": 1,
            "A": 4,
            "B": 4,
            "C": 4,
        }.get(oracle, 2)
    else:
        oracle_penalty = {
            "C": 0,
            "D": 0,
            "F": 0,
            "G": 1,
            "E": 1,
            "B": 1,
            "A": 4,
        }.get(oracle, 2)
    pair_rank = 0 if pair in core_pairs else 1 if pair in medium_pairs else 2
    hard_shunt_prefix = int(
        oracle in {"E", "G"}
        and target in {"E", "G"}
        and prefix_complexity > 0
    )
    battery_only_oracle = int(
        oracle == "A" and t2_sig == (("voltmeter_battery", "absence"),)
    )

    return (
        int(gt2 != {oracle}),
        oracle_penalty,
        pair_rank,
        int(not target_safe),
        int(not t0_reasonable),
        prefix_complexity,
        hard_shunt_prefix,
        int(not sibling_pair),
        int(not t2_broad_or_confusing),
        int(not t2_not_too_complex),
        battery_only_oracle,
        abs(len(gt0) - 3),
        len(gt1),
        len(t2_single_survivors),
        t0_sig,
        t1_sig,
        t2_sig,
    )


def _rank_failed_update_template_indices(
    *,
    circuit_type: str,
    fault: str,
    template_indices: Sequence[int],
    args: argparse.Namespace,
    template_db: Optional[Dict[str, Any]],
) -> List[int]:
    if args.failed_update_case_strategy == "default":
        return list(template_indices)

    from task_b.runtime.environment import ChallengeSequence

    scored: List[Tuple[Tuple[Any, ...], int]] = []
    for template_idx in template_indices:
        challenge = ChallengeSequence(
            circuit_type,
            fault,
            template_idx,
            template_db=template_db,
            pure_retraction=args.pure_retraction,
        )
        challenge_dict = challenge.as_dict()
        if args.failed_update_powered_only and not _challenge_uses_only_powered_measurements(challenge_dict):
            continue
        score = _failed_update_template_score_v2(challenge_dict)
        scored.append((score, template_idx))
    scored.sort(key=lambda item: item[0])
    return [template_idx for _, template_idx in scored]


def _prepare_failed_update_task(
    *,
    circuit_type: str,
    fault: str,
    args: argparse.Namespace,
    template_db: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    from task_b.runtime.environment import ChallengeSequence
    from task_b.runtime.orchestrator import CircuitOrchestrator

    template_indices = ChallengeSequence.get_template_indices(
        circuit_type,
        fault,
        template_db=template_db,
    )
    allowed_indices = _allowed_template_indices(
        circuit_type=circuit_type,
        fault=fault,
        args=args,
    )
    if allowed_indices is not None:
        template_indices = [
            template_idx
            for template_idx in template_indices
            if template_idx in allowed_indices
        ]
    template_indices = _rank_failed_update_template_indices(
        circuit_type=circuit_type,
        fault=fault,
        template_indices=template_indices,
        args=args,
        template_db=template_db,
    )
    if args.max_templates_per_fault > 0:
        template_indices = template_indices[: args.max_templates_per_fault]
    template_count = len(template_indices)
    task_label = f"{circuit_type}:{fault}"
    point_specs = [
        (point_index, run_idx, template_idx)
        for point_index, (run_idx, template_idx) in enumerate(
            (run_idx, template_idx)
            for run_idx in range(args.num_runs)
            for template_idx in template_indices
        )
    ]

    def build_point(item: Tuple[int, int, int]) -> Tuple[Dict[str, Any], Any]:
        point_index, run_idx, template_idx = item
        exp_id = f"stats_{circuit_type}_{fault}_tpl{template_idx}_run{run_idx}"
        challenge = ChallengeSequence(
            circuit_type,
            fault,
            template_idx,
            template_db=template_db,
            pure_retraction=args.pure_retraction,
        )
        challenge_dict = challenge.as_dict()

        point = {
            "point_index": point_index,
            "experiment_id": exp_id,
            "task_label": task_label,
            "circuit_type": circuit_type,
            "fault": fault,
            "template_idx": template_idx,
            "run_idx": run_idx,
            "template_count": template_count,
            "challenge": challenge_dict,
        }
        return point, challenge

    built_points = _thread_map_ordered(
        build_point,
        point_specs,
        desc=f"prepare {task_label}",
        workers=args.prepare_workers,
    )

    points: List[Dict[str, Any]] = []
    sessions: List[Dict[str, Any]] = []
    for point, challenge in built_points:
        points.append(point)
        for repeat_index in range(args.repeats):
            orchestrator = CircuitOrchestrator(
                challenge,
                None,
                prompt_style=args.prompt_style,
                temperature=args.agent_temperature,
                max_tokens=args.agent_max_tokens,
                context_turns=args.context_turns,
                include_fault_predictions=args.include_fault_predictions,
                model_type=args.model_type,
            )
            sessions.append({
                "task_label": task_label,
                "point_index": point["point_index"],
                "repeat_index": repeat_index,
                "orchestrator": orchestrator,
            })

    return points, sessions, template_count


@dataclass(frozen=True)
class FailedStayTemplateCase:
    oracle: str
    t0: MeasurementCandidate
    t1: MeasurementCandidate
    t2: MeasurementCandidate
    score: Tuple[Any, ...]
    post_interference: Tuple[MeasurementCandidate, ...] = ()
    post_interference_batches: Tuple[Tuple[MeasurementCandidate, ...], ...] = ()
    direct_converge: bool = False
    failed_stay_eval_start_turn: int = 2

    @property
    def signature(self) -> Tuple[Any, ...]:
        return (
            self.oracle,
            self.t0.signature,
            self.t1.signature,
            self.t2.signature,
            tuple(
                tuple(candidate.signature for candidate in batch)
                for batch in self._turn_batches()[3:]
            ),
        )

    def _turn_batches(self) -> List[Tuple[MeasurementCandidate, ...]]:
        batches = [(self.t0,), (self.t1,), (self.t2,)]
        if self.post_interference_batches:
            batches.extend(self.post_interference_batches)
        else:
            batches.extend((candidate,) for candidate in self.post_interference)
        return batches

    def as_template_case(self) -> Dict[str, Any]:
        turn_batches = self._turn_batches()
        ground_truth_survivors: List[List[str]] = []
        accumulated: Optional[Set[str]] = None
        topology = get_topology(self.t0.circuit_type)
        for batch in turn_batches:
            candidate_survivors = set(topology.fault_ids)
            for candidate in batch:
                candidate_survivors &= set(candidate.survivors)
            accumulated = (
                set(candidate_survivors)
                if accumulated is None
                else accumulated & candidate_survivors
            )
            ground_truth_survivors.append(sorted(accumulated))

        payload = {
            f"t{turn_idx}": {
                "meas": [candidate.as_mapping() for candidate in batch]
            }
            for turn_idx, batch in enumerate(turn_batches)
        }
        payload.update({
            "auto_generated": {
                "ground_truth_survivors": ground_truth_survivors,
                "measurement_modes": {
                    f"t{turn_idx}": ",".join(
                        sorted({candidate.mode for candidate in batch})
                    )
                    for turn_idx, batch in enumerate(turn_batches)
                },
                "post_interference_rounds": max(0, len(turn_batches) - 3),
                "post_interference_blocks_per_round": [
                    len(batch) for batch in turn_batches[3:]
                ],
                "direct_converge": self.direct_converge,
                "failed_stay_eval_start_turn": self.failed_stay_eval_start_turn,
            },
        })
        return payload


def _candidate_keys(candidate: MeasurementCandidate) -> Set[str]:
    return {key for key, _ in candidate.measurements}


def _key_targets_component(key: str, component: str) -> bool:
    """True if a measurement key reads on the given component (e.g. voltmeter_r2)."""
    if not component:
        return False
    return key.endswith(f"_{component}") or key == component


def _multi_candidate_set_failed_stay_case_score_v9(
    t0: MeasurementCandidate,
    t1: MeasurementCandidate,
    t2: MeasurementCandidate,
    t0_survivors: Set[str],
    oracle: str,
) -> Tuple[Any, ...]:
    """Rank cases that keep full-history truth at a small candidate set.

    The prefix should establish a stable two- or three-fault belief, while T2
    presents a broader local reading that can tempt the model to expand or
    collapse the set if it overweights the latest turn.
    """
    paired_fault = _paired_branch_fault(t0.circuit_type, oracle)
    target_set = _multi_candidate_target_set(t0.circuit_type, oracle)
    keys0 = _candidate_keys(t0)
    keys1 = _candidate_keys(t1)
    keys2 = _candidate_keys(t2)
    t1_survivors = set(t1.survivors)
    t2_survivors = set(t2.survivors)
    oracle_component = _fault_component(t0.circuit_type, oracle)
    paired_component = _fault_component(t0.circuit_type, paired_fault)
    extra_count = len(t2_survivors - target_set)
    targets_pair_component = any(
        _key_targets_component(key, oracle_component)
        or _key_targets_component(key, paired_component)
        for key in keys2
    )
    uses_global_key = bool(keys2 & {"ammeter_main", "voltmeter_battery"})

    return (
        abs(len(t0_survivors) - 4),
        abs(len(t1_survivors) - 4),
        int(extra_count <= 0),
        abs(extra_count - 1),
        int(not targets_pair_component),
        int(uses_global_key),
        len(keys2 & (keys0 | keys1)),
        abs(t2.condition_count - 2),
        t0.condition_count,
        t1.condition_count,
        t2.condition_count,
        t0.signature,
        t1.signature,
        t2.signature,
    )


def _singleton_converge_failed_stay_case_score_v1(
    t0: MeasurementCandidate,
    t1: MeasurementCandidate,
    t2: MeasurementCandidate,
    oracle: str,
) -> Tuple[Any, ...]:
    """Rank singleton-converged failed_stay cases.

    T0/T1 establish the oracle singleton; T2 is locally broader and can tempt
    recency-driven expansion, but the cumulative survivor set stays singleton.
    """
    keys0 = _candidate_keys(t0)
    keys1 = _candidate_keys(t1)
    keys2 = _candidate_keys(t2)
    oracle_component = _fault_component(t0.circuit_type, oracle)
    paired_fault = _paired_branch_fault(t0.circuit_type, oracle)
    paired_component = _fault_component(t0.circuit_type, paired_fault)
    targets_oracle_component = any(
        _key_targets_component(key, oracle_component) for key in keys2
    )
    targets_paired_component = any(
        _key_targets_component(key, paired_component) for key in keys2
    )
    uses_global_key = bool(keys2 & {"ammeter_main", "voltmeter_battery"})
    t2_survivors = set(t2.survivors)
    extra_count = len(t2_survivors - {oracle})

    return (
        abs(len(t0.survivors) - 3),
        abs(len(t1.survivors) - 3),
        int(extra_count <= 0),
        abs(extra_count - 2),
        int(not targets_paired_component),
        int(targets_oracle_component),
        int(uses_global_key),
        len(keys2 & (keys0 | keys1)),
        abs(t2.condition_count - 2),
        t0.condition_count,
        t1.condition_count,
        t2.condition_count,
        t0.signature,
        t1.signature,
        t2.signature,
    )


def _benchmark_protocol_for_topology(circuit_type: str) -> Dict[str, Any]:
    protocol = get_benchmark_protocol()
    protocol["fault_count"] = len(get_topology(circuit_type).fault_ids)
    return protocol


def _pick_misleading_target(circuit_type: str, oracle: str) -> str:
    target_set = _multi_candidate_target_set(circuit_type, oracle)
    for fault_id in sorted(target_set):
        if fault_id != oracle:
            return fault_id
    paired_fault = _paired_branch_fault(circuit_type, oracle)
    if paired_fault:
        return paired_fault
    for fault_id in get_topology(circuit_type).fault_ids:
        if fault_id != oracle:
            return fault_id
    return oracle


def _build_event(
    turn: int,
    event_type: str,
    measurements: Mapping[str, str],
) -> Dict[str, Any]:
    return enrich_event(
        {
            "turn": turn,
            "type": event_type,
            "measurements": dict(measurements),
        }
    )


def _copy_ground_truth_step(step: Mapping[str, Any]) -> Dict[str, Any]:
    copied: Dict[str, Any] = {
        "turn": step["turn"],
        "survivors": sorted(step["survivors"]),
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


def _build_failed_stay_event(
    turn: int,
    event_type: str,
    candidate: MeasurementCandidate,
) -> Dict[str, Any]:
    event = _build_event(turn, event_type, candidate.as_mapping())
    event["local_survivors"] = list(candidate.survivors)
    return event


def _candidate_measurements_are_compatible(
    candidates: Sequence[MeasurementCandidate],
) -> bool:
    merged: Dict[str, str] = {}
    for candidate in candidates:
        for key, value in candidate.measurements:
            if key in merged and merged[key] != value:
                return False
            merged[key] = value
    return True


def _merge_candidate_measurements(
    candidates: Sequence[MeasurementCandidate],
) -> Dict[str, str]:
    if not _candidate_measurements_are_compatible(candidates):
        raise ValueError("Cannot merge incompatible measurement candidates")
    merged: Dict[str, str] = {}
    for candidate in candidates:
        merged.update(candidate.as_mapping())
    return dict(sorted(merged.items()))


def _batch_local_survivors(
    candidates: Sequence[MeasurementCandidate],
) -> List[str]:
    if not candidates:
        return []
    topology = get_topology(candidates[0].circuit_type)
    survivors = set(topology.fault_ids)
    for candidate in candidates:
        survivors &= set(candidate.survivors)
    return sorted(survivors)


def _build_failed_stay_batch_event(
    turn: int,
    event_type: str,
    candidates: Sequence[MeasurementCandidate],
) -> Dict[str, Any]:
    event = _build_event(turn, event_type, _merge_candidate_measurements(candidates))
    event["measurement_blocks"] = [
        candidate.as_mapping() for candidate in candidates
    ]
    event["local_block_survivors"] = [
        list(candidate.survivors) for candidate in candidates
    ]
    event["local_survivors"] = _batch_local_survivors(candidates)
    return event


class FailedStayChallengeSequence:
    """Three-turn d7 revision-confirm measurement challenge for failed_stay evaluation."""

    def __init__(
        self,
        circuit_type: str,
        oracle: str,
        template_idx: int,
        case: FailedStayTemplateCase,
        prefix_rule_prediction_turns: int = 0,
    ):
        self.circuit_type = circuit_type
        self.oracle = oracle
        topology = get_topology(circuit_type)
        self.oracle_description = topology.fault_options[oracle]
        self.template_idx = template_idx
        self.case = case
        self.prefix_rule_prediction_turns = max(0, int(prefix_rule_prediction_turns))
        self.benchmark_protocol = _benchmark_protocol_for_topology(circuit_type)
        self.misleading_target = _pick_misleading_target(circuit_type, oracle)
        self.symptom: Optional[dict] = None
        self.events = [
            _build_failed_stay_event(0, "initial_measurement", case.t0),
            _build_failed_stay_event(
                1,
                "failed_stay_probe_measurement" if case.direct_converge else "confirm_measurement",
                case.t1,
            ),
            _build_failed_stay_event(2, "failed_stay_probe_measurement", case.t2),
        ]
        post_batches = (
            list(case.post_interference_batches)
            if case.post_interference_batches
            else [(candidate,) for candidate in case.post_interference]
        )
        for offset, batch in enumerate(post_batches, start=3):
            if len(batch) == 1:
                self.events.append(
                    _build_failed_stay_event(offset, "failed_stay_probe_measurement", batch[0])
                )
            else:
                self.events.append(
                    _build_failed_stay_batch_event(offset, "failed_stay_probe_measurement", batch)
                )
        self._attach_prefix_rule_predictions()
        computed = compute_ground_truth_history(self.events, circuit_type)
        self.ground_truth = [_copy_ground_truth_step(step) for step in computed]
        self.total_turns = len(self.events)

        ok, errors = validate_challenge_against_rules(
            self.as_dict(),
            require_support_compatibility=False,
        )
        if not ok:
            raise ValueError(
                f"FailedStayChallengeSequence validation failed "
                f"({circuit_type}:{oracle}:tpl{template_idx}): "
                + "; ".join(errors)
            )

    def _attach_prefix_rule_predictions(self) -> None:
        if self.prefix_rule_prediction_turns <= 0:
            return

        accumulated: Dict[str, str] = {}
        for event in self.events:
            measurements = event.get("measurements") or {}
            accumulated.update(dict(measurements))
            if int(event["turn"]) < self.prefix_rule_prediction_turns:
                event["prefix_rule_prediction"] = {
                    "circuit_type": self.circuit_type,
                    "measurements": dict(accumulated),
                }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "circuit_type": self.circuit_type,
            "oracle": self.oracle,
            "oracle_description": self.oracle_description,
            "template_idx": self.template_idx,
            "total_turns": self.total_turns,
            "misleading_target": self.misleading_target,
            "prefix_rule_prediction_turns": self.prefix_rule_prediction_turns,
            "benchmark_protocol": dict(self.benchmark_protocol),
            "events": [dict(event) for event in self.events],
            "ground_truth": [dict(item) for item in self.ground_truth],
            "direct_converge": self.case.direct_converge,
            "failed_stay_eval_start_turn": self.case.failed_stay_eval_start_turn,
            "single_condition_mode": all(
                len(event["measurements"]) == 1
                for event in self.events
                if event["measurements"]
            ),
            "uses_external_templates": False,
            "challenge_family": "belief_failed_stay_measurement",
            "template_case": self.case.as_template_case(),
        }


def enumerate_failed_stay_template_cases(
    circuit_type: str,
    oracle: str,
    *,
    max_powered_conditions: int = 2,
    include_isolation_healthy: bool = True,
    strategy: str = "multi_candidate_set_failed_stay_v9",
) -> List[FailedStayTemplateCase]:
    if strategy not in {"multi_candidate_set_failed_stay_v9", "singleton_converge_broad_v1"}:
        raise ValueError(
            "Unsupported failed_stay case strategy after cleanup: "
            f"{strategy}. Use multi_candidate_set_failed_stay_v9 or "
            "singleton_converge_broad_v1."
        )

    candidates = enumerate_measurement_candidates(
        circuit_type,
        max_powered_conditions=max_powered_conditions,
        include_isolation_healthy=include_isolation_healthy,
    )
    powered_candidates = [
        candidate for candidate in candidates if candidate.mode == "powered"
    ]

    cases: List[FailedStayTemplateCase] = []
    seen_signatures = set()
    if strategy == "singleton_converge_broad_v1":
        for t0 in powered_candidates:
            t0_survivors = set(t0.survivors)
            if oracle not in t0_survivors:
                continue
            if not (2 <= len(t0_survivors) <= 4):
                continue

            for t1 in powered_candidates:
                if t1.signature == t0.signature:
                    continue
                t1_survivors = set(t1.survivors)
                if oracle not in t1_survivors:
                    continue
                if not (2 <= len(t1_survivors) <= 4):
                    continue
                if t0_survivors & t1_survivors != {oracle}:
                    continue

                for t2 in powered_candidates:
                    if t2.signature in {t0.signature, t1.signature}:
                        continue
                    t2_survivors = set(t2.survivors)
                    if oracle not in t2_survivors:
                        continue
                    if not (2 <= len(t2_survivors) <= 5):
                        continue
                    if (t0_survivors & t1_survivors & t2_survivors) != {oracle}:
                        continue

                    case = FailedStayTemplateCase(
                        oracle=oracle,
                        t0=t0,
                        t1=t1,
                        t2=t2,
                        score=_singleton_converge_failed_stay_case_score_v1(
                            t0,
                            t1,
                            t2,
                            oracle,
                        ),
                        failed_stay_eval_start_turn=2,
                    )
                    if case.signature in seen_signatures:
                        continue
                    seen_signatures.add(case.signature)
                    cases.append(case)

        cases.sort(key=lambda case: case.score)
        return cases

    for t0 in powered_candidates:
        t0_survivors = set(t0.survivors)
        target_set = _multi_candidate_target_set(t0.circuit_type, oracle)
        if len(target_set) < 2:
            continue
        if not target_set <= t0_survivors:
            continue
        if not (len(target_set) <= len(t0_survivors) <= len(target_set) + 3):
            continue

        for t1 in powered_candidates:
            if t1.signature == t0.signature:
                continue
            t1_survivors = set(t1.survivors)
            if not target_set <= t1_survivors:
                continue
            if not (len(target_set) <= len(t1_survivors) <= len(target_set) + 3):
                continue
            t1_accumulated = t0_survivors & t1_survivors
            if t1_accumulated != target_set:
                continue

            for t2 in powered_candidates:
                if t2.signature in {t0.signature, t1.signature}:
                    continue
                t2_survivors = set(t2.survivors)
                if not target_set <= t2_survivors:
                    continue
                if not (
                    len(target_set) + 1
                    <= len(t2_survivors)
                    <= len(target_set) + 3
                ):
                    continue
                if t1_accumulated & t2_survivors != target_set:
                    continue

                case = FailedStayTemplateCase(
                    oracle=oracle,
                    t0=t0,
                    t1=t1,
                    t2=t2,
                    score=_multi_candidate_set_failed_stay_case_score_v9(
                        t0,
                        t1,
                        t2,
                        t0_survivors,
                        oracle,
                    ),
                    failed_stay_eval_start_turn=2,
                )
                if case.signature in seen_signatures:
                    continue
                seen_signatures.add(case.signature)
                cases.append(case)

    cases.sort(key=lambda case: case.score)
    return cases


def _add_post_convergence_interference(
    selected: Sequence[FailedStayTemplateCase],
    all_cases: Sequence[FailedStayTemplateCase],
    rounds: int,
    blocks_per_turn: int = 1,
    coherent_distractor: bool = False,
    prefer_broad_distractor: bool = False,
    extra_candidates: Optional[Sequence[MeasurementCandidate]] = None,
) -> List[FailedStayTemplateCase]:
    """Append Scenario-A-style singleton-preserving interference turns.

    The extra turns share the same T0/T1 singleton-convergence prefix as the
    base case. Each extra measurement still contains the oracle and its paired
    branch fault locally, but after T1 the accumulated survivor set remains the
    oracle singleton.
    """
    if rounds <= 0:
        return list(selected)
    if blocks_per_turn <= 0:
        raise ValueError("blocks_per_turn must be >= 1")

    by_prefix: Dict[
        Tuple[str, Tuple[Any, ...], Tuple[Any, ...]],
        List[MeasurementCandidate],
    ] = {}
    for case in all_cases:
        prefix = (case.oracle, case.t0.signature, case.t1.signature)
        by_prefix.setdefault(prefix, []).append(case.t2)

    augmented: List[FailedStayTemplateCase] = []
    for case in selected:
        prefix = (case.oracle, case.t0.signature, case.t1.signature)
        used = {case.t0.signature, case.t1.signature, case.t2.signature}
        seen_prefix_keys = _candidate_keys(case.t0) | _candidate_keys(case.t1)
        t1_keys = _candidate_keys(case.t1)
        oracle_component = _fault_component(case.t0.circuit_type, case.oracle)
        paired_fault = _paired_branch_fault(case.t0.circuit_type, case.oracle)
        paired_component = _fault_component(case.t0.circuit_type, paired_fault)
        target_survivors = (
            _multi_candidate_target_set(case.t0.circuit_type, case.oracle)
            if prefer_broad_distractor
            else {case.oracle, paired_fault}
            if paired_fault
            else {case.oracle}
        )
        extras: List[MeasurementCandidate] = []

        def post_candidate_rank(candidate: MeasurementCandidate) -> Tuple[Any, ...]:
            keys = _candidate_keys(candidate)
            survivors = set(candidate.survivors)
            targets_paired_component = any(
                _key_targets_component(key, paired_component) for key in keys
            )
            targets_oracle_component = any(
                _key_targets_component(key, oracle_component) for key in keys
            )
            uses_global_key = bool(keys & {"ammeter_main", "voltmeter_battery"})
            if coherent_distractor:
                if prefer_broad_distractor:
                    contains_pair = target_survivors <= survivors
                    broad_with_extra = contains_pair and bool(survivors - target_survivors)
                    return (
                        int(not broad_with_extra),
                        abs(len(survivors) - (len(target_survivors) + 2)),
                        int(not targets_paired_component),
                        int(targets_oracle_component),
                        int(uses_global_key),
                        len(keys & t1_keys),
                        len(keys & seen_prefix_keys),
                        abs(candidate.condition_count - 2),
                        candidate.signature,
                    )
                return (
                    int(survivors != target_survivors),
                    abs(len(survivors) - len(target_survivors)),
                    int(not targets_paired_component),
                    int(targets_oracle_component),
                    int(uses_global_key),
                    len(keys & t1_keys),
                    len(keys & seen_prefix_keys),
                    abs(candidate.condition_count - 2),
                    candidate.signature,
                )
            return (
                int(not targets_paired_component),
                len(keys & t1_keys),
                len(keys & seen_prefix_keys),
                int(targets_oracle_component),
                -candidate.condition_count,
                candidate.signature,
            )

        candidate_pool = (
            list(extra_candidates)
            if extra_candidates is not None
            else by_prefix.get(prefix, [])
        )
        ranked_candidates = sorted(candidate_pool, key=post_candidate_rank)
        post_batches: List[Tuple[MeasurementCandidate, ...]] = []
        current_batch: List[MeasurementCandidate] = []
        for candidate in ranked_candidates:
            if candidate.signature in used:
                continue
            if coherent_distractor:
                survivors = set(candidate.survivors)
                if prefer_broad_distractor:
                    if not target_survivors <= survivors:
                        continue
                elif case.oracle not in survivors or paired_fault not in survivors:
                    continue
                if prefer_broad_distractor and not (survivors - target_survivors):
                    continue

            proposed_batch = [*current_batch, candidate]
            if current_batch and (
                len(current_batch) >= blocks_per_turn
                or not _candidate_measurements_are_compatible(proposed_batch)
            ):
                post_batches.append(tuple(current_batch))
                current_batch = []
                if len(post_batches) >= rounds:
                    break
                proposed_batch = [candidate]

            current_batch.append(candidate)
            extras.append(candidate)
            used.add(candidate.signature)
            if len(current_batch) >= blocks_per_turn:
                post_batches.append(tuple(current_batch))
                current_batch = []
                if len(post_batches) >= rounds:
                    break

        if current_batch and len(post_batches) < rounds:
            post_batches.append(tuple(current_batch))

        augmented.append(
            FailedStayTemplateCase(
                oracle=case.oracle,
                t0=case.t0,
                t1=case.t1,
                t2=case.t2,
                score=case.score,
                post_interference=tuple(extras),
                post_interference_batches=tuple(post_batches),
                direct_converge=case.direct_converge,
                failed_stay_eval_start_turn=case.failed_stay_eval_start_turn,
            )
        )

    return augmented


def _load_failed_stay_cases(
    circuit_type: str,
    fault: str,
    args: argparse.Namespace,
    template_db: Optional[Dict[str, Any]] = None,
) -> List[FailedStayTemplateCase]:
    if template_db is not None:
        raw_cases = list(template_db.get(fault, []))
        if args.max_cases_per_fault > 0:
            raw_cases = raw_cases[: args.max_cases_per_fault]
        return [
            _failed_stay_case_from_template_payload(circuit_type, fault, index, payload)
            for index, payload in enumerate(raw_cases)
        ]

    cases = enumerate_failed_stay_template_cases(
        circuit_type,
        fault,
        max_powered_conditions=args.max_powered_conditions,
        include_isolation_healthy=args.include_isolation_healthy,
        strategy=args.failed_stay_case_strategy,
    )
    selected = cases[: args.max_cases_per_fault]
    if not selected:
        return []
    extra_candidates = [
        candidate
        for candidate in enumerate_measurement_candidates(
            circuit_type,
            max_powered_conditions=args.max_powered_conditions,
            include_isolation_healthy=args.include_isolation_healthy,
        )
        if candidate.mode == "powered"
    ]
    return _add_post_convergence_interference(
        selected,
        cases,
        rounds=args.failed_stay_interference_rounds,
        blocks_per_turn=getattr(args, "failed_stay_blocks_per_post_turn", 1),
        coherent_distractor=True,
        prefer_broad_distractor=True,
        extra_candidates=extra_candidates,
    )


def _measurement_candidate_from_payload(
    circuit_type: str,
    measurements: Mapping[str, str],
    *,
    mode: str = "powered",
) -> MeasurementCandidate:
    normalized = {
        str(key): str(value)
        for key, value in measurements.items()
    }
    return MeasurementCandidate(
        circuit_type=circuit_type,
        mode=mode,
        measurements=tuple(sorted(normalized.items())),
        survivors=tuple(sorted(supported_faults_for_condition(normalized, circuit_type))),
    )


def _template_turn_candidates(
    circuit_type: str,
    payload: Mapping[str, Any],
    turn_name: str,
    measurement_modes: Mapping[str, str],
) -> Tuple[MeasurementCandidate, ...]:
    turn_payload = payload.get(turn_name)
    if not isinstance(turn_payload, Mapping):
        raise ValueError(f"FailedStay template missing {turn_name}")
    meas_list = turn_payload.get("meas")
    if not isinstance(meas_list, Sequence) or isinstance(meas_list, (str, bytes)) or not meas_list:
        raise ValueError(f"FailedStay template {turn_name}.meas must be a non-empty list")

    mode = str(measurement_modes.get(turn_name, "powered"))
    candidates: List[MeasurementCandidate] = []
    for measurements in meas_list:
        if not isinstance(measurements, Mapping):
            raise ValueError(f"FailedStay template {turn_name}.meas entries must be mappings")
        candidates.append(
            _measurement_candidate_from_payload(
                circuit_type,
                measurements,
                mode=mode,
            )
        )
    return tuple(candidates)


def _failed_stay_case_from_template_payload(
    circuit_type: str,
    fault: str,
    template_idx: int,
    payload: Mapping[str, Any],
) -> FailedStayTemplateCase:
    auto = payload.get("auto_generated", {})
    if not isinstance(auto, Mapping):
        auto = {}
    measurement_modes = auto.get("measurement_modes", {})
    if not isinstance(measurement_modes, Mapping):
        measurement_modes = {}

    prefix_batches = [
        _template_turn_candidates(circuit_type, payload, turn_name, measurement_modes)
        for turn_name in ("t0", "t1", "t2")
    ]
    if any(len(batch) != 1 for batch in prefix_batches):
        raise ValueError(
            f"FailedStay template {circuit_type}:{fault}:tpl{template_idx} must have "
            "exactly one measurement block for t0/t1/t2"
        )

    post_batches: List[Tuple[MeasurementCandidate, ...]] = []
    turn_index = 3
    while f"t{turn_index}" in payload:
        post_batches.append(
            _template_turn_candidates(
                circuit_type,
                payload,
                f"t{turn_index}",
                measurement_modes,
            )
        )
        turn_index += 1

    post_interference = tuple(
        candidate for batch in post_batches for candidate in batch
    )
    return FailedStayTemplateCase(
        oracle=fault,
        t0=prefix_batches[0][0],
        t1=prefix_batches[1][0],
        t2=prefix_batches[2][0],
        score=(template_idx,),
        post_interference=post_interference,
        post_interference_batches=tuple(post_batches),
        direct_converge=bool(auto.get("direct_converge", False)),
        failed_stay_eval_start_turn=int(auto.get("failed_stay_eval_start_turn", 2)),
    )


def _prepare_failed_stay_task(
    *,
    circuit_type: str,
    fault: str,
    args: argparse.Namespace,
    template_db: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    from task_b.runtime.orchestrator import CircuitOrchestrator

    template_cases = _load_failed_stay_cases(circuit_type, fault, args, template_db=template_db)
    if not template_cases:
        return [], [], 0
    allowed_indices = _allowed_template_indices(
        circuit_type=circuit_type,
        fault=fault,
        args=args,
    )
    if allowed_indices is not None:
        print(
            f"Template filter for {circuit_type}:{fault}: "
            f"{len(allowed_indices)} requested indices"
        )
    template_items = [
        (template_idx, case)
        for template_idx, case in enumerate(template_cases)
        if allowed_indices is None or template_idx in allowed_indices
    ]
    if not template_items:
        return [], [], 0
    template_count = len(template_items)
    task_label = f"{circuit_type}:{fault}"
    point_specs = [
        (point_index, run_idx, template_idx, case)
        for point_index, (run_idx, template_idx, case) in enumerate(
            (run_idx, template_idx, case)
            for run_idx in range(args.num_runs)
            for template_idx, case in template_items
        )
    ]

    def build_point(item: Tuple[int, int, int, FailedStayTemplateCase]) -> Tuple[Dict[str, Any], Any]:
        point_index, run_idx, template_idx, case = item
        exp_id = f"stats_{circuit_type}_{fault}_tpl{template_idx}_run{run_idx}"
        challenge = FailedStayChallengeSequence(
            circuit_type,
            fault,
            template_idx,
            case,
            prefix_rule_prediction_turns=args.failed_stay_prefix_rule_prediction_turns,
        )
        challenge_dict = challenge.as_dict()

        point = {
            "point_index": point_index,
            "experiment_id": exp_id,
            "task_label": task_label,
            "circuit_type": circuit_type,
            "fault": fault,
            "template_idx": template_idx,
            "run_idx": run_idx,
            "template_count": template_count,
            "challenge": challenge_dict,
        }
        return point, challenge

    built_points = _thread_map_ordered(
        build_point,
        point_specs,
        desc=f"prepare {task_label}",
        workers=args.prepare_workers,
    )

    points: List[Dict[str, Any]] = []
    sessions: List[Dict[str, Any]] = []
    for point, challenge in built_points:
        points.append(point)
        for repeat_index in range(args.repeats):
            orchestrator = CircuitOrchestrator(
                challenge,
                None,
                prompt_style=args.prompt_style,
                temperature=args.agent_temperature,
                max_tokens=args.agent_max_tokens,
                context_turns=args.context_turns,
                include_fault_predictions=args.include_fault_predictions,
                model_type=args.model_type,
            )
            sessions.append({
                "task_label": task_label,
                "point_index": point["point_index"],
                "repeat_index": repeat_index,
                "orchestrator": orchestrator,
            })

    return points, sessions, template_count


def _prepare_noise_task(
    *,
    circuit_type: str,
    fault: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    from task_b.runtime.environment import NoiseChallengeSequence
    from task_b.runtime.orchestrator import CircuitOrchestrator

    task_label = f"{circuit_type}:{fault}"
    template_count = 1
    point_specs = [
        (point_index, run_idx)
        for point_index, run_idx in enumerate(range(args.num_runs))
    ]

    def build_point(item: Tuple[int, int]) -> Tuple[Dict[str, Any], Any]:
        point_index, run_idx = item
        exp_id = f"stats_{circuit_type}_{fault}_noise_run{run_idx}"
        challenge = NoiseChallengeSequence(
            circuit_type,
            fault,
            template_idx=0,
            max_turns=args.noise_max_turns,
            add_host_comment=args.add_host_comment,
        )
        challenge_dict = challenge.as_dict()

        point = {
            "point_index": point_index,
            "experiment_id": exp_id,
            "task_label": task_label,
            "circuit_type": circuit_type,
            "fault": fault,
            "template_idx": 0,
            "run_idx": run_idx,
            "template_count": template_count,
            "challenge": challenge_dict,
        }
        return point, challenge

    built_points = _thread_map_ordered(
        build_point,
        point_specs,
        desc=f"prepare {task_label}",
        workers=args.prepare_workers,
    )

    points: List[Dict[str, Any]] = []
    sessions: List[Dict[str, Any]] = []
    for point, challenge in built_points:
        points.append(point)
        for repeat_index in range(args.repeats):
            orchestrator = CircuitOrchestrator(
                challenge,
                None,
                prompt_style=args.prompt_style,
                temperature=args.agent_temperature,
                max_tokens=args.agent_max_tokens,
                context_turns=args.context_turns,
                include_fault_predictions=args.include_fault_predictions,
                model_type=args.model_type,
            )
            sessions.append({
                "task_label": task_label,
                "point_index": point["point_index"],
                "repeat_index": repeat_index,
                "orchestrator": orchestrator,
            })

    return points, sessions, template_count


def _prepare_task_data(
    *,
    circuit_type: str,
    fault: str,
    args: argparse.Namespace,
    template_db: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    if args.mode == "failed_update":
        return _prepare_failed_update_task(
            circuit_type=circuit_type,
            fault=fault,
            args=args,
            template_db=template_db,
        )
    if args.mode == "noise":
        return _prepare_noise_task(
            circuit_type=circuit_type,
            fault=fault,
            args=args,
        )
    if args.mode == "failed_isolation":
        # FailedIsolation reuses failed_stay sequence generation but injects a misleading
        # host comment line at every turn. We attach a deterministic per-session
        # rng to each orchestrator so prompt construction can pick templates.
        import random as _random
        points, sessions, template_count = _prepare_failed_stay_task(
            circuit_type=circuit_type,
            fault=fault,
            args=args,
            template_db=template_db,
        )
        for point in points:
            point.setdefault("challenge", {})["mode"] = "failed_isolation"
            point.setdefault("challenge", {})["strict_failed_isolation_prefix_turns"] = int(
                getattr(args, "strict_failed_isolation_prefix_turns", 0) or 0
            )

        for session in sessions:
            orchestrator = session["orchestrator"]
            challenge = orchestrator.challenge
            seed_str = (
                f"failed_isolation:{circuit_type}:{fault}:{session.get('point_index')}:"
                f"{session.get('repeat_index')}:{getattr(args, 'seed', 0)}"
            )
            session_rng = _random.Random(sum(ord(ch) for ch in seed_str))
            orchestrator.failed_isolation_options = {
                "rng": session_rng,
                "oracle": challenge.oracle,
                "candidate_names": sorted(orchestrator.valid_faults),
                "add_failed_isolation_comment": bool(getattr(args, "add_failed_isolation_comment", True)),
                "comment_start_turn": int(getattr(args, "failed_isolation_comment_start_turn", 0) or 0),
                "preserve_turn_message": bool(getattr(args, "preserve_failed_isolation_turn_message", False)),
            }
        return points, sessions, template_count
    return _prepare_failed_stay_task(
        circuit_type=circuit_type,
        fault=fault,
        args=args,
        template_db=template_db,
    )

def _save_task_outputs(
    *,
    task_label: str,
    points: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    category_dirs: Dict[str, str],
    repeats: int,
) -> Dict[str, Any]:
    sessions_by_point: Dict[int, List[Dict[str, Any]]] = {}
    for session in sessions:
        sessions_by_point.setdefault(session["point_index"], []).append(session)

    counts = {category: 0 for category in CATEGORIES}

    for completed_count, point in enumerate(points, start=1):
        per_run_categories: List[str] = []
        repeat_trajectories: List[Dict[str, Any]] = []

        point_sessions = sorted(
            sessions_by_point.get(point["point_index"], []),
            key=lambda item: item["repeat_index"],
        )
        for session in point_sessions:
            trajectory = session["trajectory"]
            category = classify_trajectory(trajectory, point["challenge"])
            per_run_categories.append(category)
            repeat_trajectories.append({
                "repeat_index": session["repeat_index"],
                "category": category,
                "trajectory": sanitize_trajectory_for_export(trajectory),
            })

        category = aggregate_repeat_categories(per_run_categories)
        counts[category] += 1

        save_json(
            os.path.join(category_dirs[category], f"{point['experiment_id']}.json"),
            compact_case_export(
                challenge_dict=point["challenge"],
                repeat_trajectories=repeat_trajectories,
            ),
        )

        detail = ", ".join(
            f"{label}={count}"
            for label, count in sorted(Counter(per_run_categories).items())
        )
        print(
            f"  [{completed_count}/{len(points)}] {point['experiment_id']}: "
            f"{category} ({detail})"
        )

    return {
        "task_label": task_label,
        "circuit_type": points[0]["circuit_type"] if points else "",
        "fault": points[0]["fault"] if points else "",
        "template_count": points[0]["template_count"] if points else 0,
        "counts": counts,
        "completed": len(points),
    }


def _print_run_configuration(
    *,
    model_name: str,
    gpu_ids: Sequence[int],
    task_labels: Sequence[str],
    args: argparse.Namespace,
) -> None:
    print(f"Mode: {args.mode}")
    print(f"Model: {model_name}")
    print(f"Prompt style: {args.prompt_style}")
    if args.mode == "failed_update":
        print(
            f"Templates: {args.templates_json} (verified)"
            if args.templates_json
            else "Templates: builtin defaults (verified)"
        )
        if args.max_templates_per_fault > 0:
            print(f"Max failed_update templates per fault: {args.max_templates_per_fault}")
        print(f"FailedUpdate case strategy: {args.failed_update_case_strategy}")
    elif args.mode == "noise":
        print("Templates: online noise querying (no fixed templates)")
        print(f"Noise max turns: {args.noise_max_turns}")
    else:
        if args.mode == "failed_isolation":
            print(
                f"FailedIsolation mode (failed_stay sequence + misleading host comments, "
                f"add_failed_isolation_comment={getattr(args, 'add_failed_isolation_comment', True)})"
            )
        print(
            f"Templates: {args.templates_json} (verified)"
            if args.templates_json
            else "Templates: auto-generated d7 revision-confirm failed_stay cases"
        )
        print(f"Max failed_stay templates per fault: {args.max_cases_per_fault}")
        print(f"FailedStay case strategy: {args.failed_stay_case_strategy}")
        print(f"FailedStay post-convergence interference rounds: {args.failed_stay_interference_rounds}")
        print(f"FailedStay blocks per post turn: {args.failed_stay_blocks_per_post_turn}")
        print(f"FailedStay prefix rule prediction turns: {args.failed_stay_prefix_rule_prediction_turns}")
    print(f"GPUs: {list(gpu_ids)} (single vLLM, tensor_parallel_size={len(gpu_ids)})")
    print(
        f"Tasks: {len(task_labels)} (circuit_type x fault), "
        f"Runs per task: {args.num_runs}, Repeats: {args.repeats}"
    )
    print(f"Sampling: temp={args.agent_temperature}, top_p={args.agent_top_p}, top_k={args.agent_top_k}, min_p={args.agent_min_p}, presence_penalty={args.agent_presence_penalty}, repetition_penalty={args.agent_repetition_penalty}")
    print(f"Max generation tokens: {args.agent_max_tokens}")
    print("Each run covers all templates exactly once per task.")


def _print_overall_report(
    *,
    model_name: str,
    prompt_style: str,
    mode: str,
    task_labels: Sequence[str],
    all_stats: Mapping[str, Mapping[str, int]],
    template_counts: Mapping[str, int],
) -> None:
    print(f"\n{'=' * 70}")
    print(f"OVERALL REPORT  (mode: {mode}, model: {model_name}, prompt: {prompt_style})")
    print(f"{'=' * 70}")

    col_w = max(len(label) for label in task_labels) + 2
    header = f"{'Task':<{col_w}}  {'tpl':>4}"
    for category in CATEGORIES:
        header += f"  {category:>12}"
    print(header)
    print("-" * len(header))

    total_counts = {category: 0 for category in CATEGORIES}
    for label in task_labels:
        counts = all_stats.get(label, {category: 0 for category in CATEGORIES})
        n_total = sum(counts.values())
        row = f"{label:<{col_w}}  {template_counts.get(label, 0):>4}"
        for category in CATEGORIES:
            n = counts[category]
            pct = n / n_total * 100 if n_total > 0 else 0
            row += f"  {n:>4} ({pct:>5.1f}%)"
            total_counts[category] += n
        print(row)

    grand_total = sum(total_counts.values())
    print("-" * len(header))
    row = f"{'TOTAL':<{col_w}}  {'-':>4}"
    for category in CATEGORIES:
        n = total_counts[category]
        pct = n / grand_total * 100 if grand_total > 0 else 0
        row += f"  {n:>4} ({pct:>5.1f}%)"
    print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scenario B belief statistics (failed_update / failed_stay / noise)",
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--backend", choices=["vllm", "api"], default="vllm")
    parser.add_argument("--model-type", choices=["local", "api_qwen35"], default="local", help="Model type for template selection")
    parser.add_argument("--api-base-url", type=str, default="")
    parser.add_argument("--api-model", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument(
        "--agent-model-path",
        type=str,
        default=os.environ.get("AGENT_MODEL_PATH", "models/Qwen3-30B-A3B-Instruct-2507"),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs used by one vLLM instance",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=0)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=0)
    parser.add_argument(
        "--vllm-language-model-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run multimodal-capable models in text-only mode.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of full template sweeps per (circuit_type, fault) pair.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repeats per data point; majority > 50%% decides category",
    )
    parser.add_argument("--circuit-types", nargs="+", default=CIRCUIT_TYPES)
    parser.add_argument("--faults", nargs="+", default=FAULT_IDS)
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="neutral",
        choices=[
            "neutral",
            "neutral_no_think",
            "minimal_no_think",
            "noise",
        ],
    )
    parser.add_argument(
        "--include-fault-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In failed_update/failed_stay turn prompts, include a per-fault prediction "
            "table for the current measurements."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-temperature", type=float, default=0.3)
    parser.add_argument("--agent-top-p", type=float, default=0.95)
    parser.add_argument("--agent-top-k", type=int, default=20)
    parser.add_argument("--agent-min-p", type=float, default=0.0)
    parser.add_argument("--agent-presence-penalty", type=float, default=1.5)
    parser.add_argument("--agent-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--agent-max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--context-turns",
        type=int,
        default=-1,
        help=(
            "Number of recent user/assistant turns to replay. "
            "-1 keeps full history; 1 keeps previous assistant answer plus current user turn."
        ),
    )
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=_default_prepare_workers(),
        help="Thread workers for CPU-side challenge/session preparation. Use 1 for serial.",
    )
    parser.add_argument(
        "--save-turn-checkpoints",
        action="store_true",
        help="Save per-turn checkpoint jsonl files. Disabled by default to keep outputs compact.",
    )
    parser.add_argument(
        "--templates-json",
        type=str,
        default="",
        help="FailedUpdate/failed_stay only: optional path to verified generated templates JSON.",
    )
    parser.add_argument(
        "--max-templates-per-fault",
        type=int,
        default=0,
        help="FailedUpdate only: use at most this many unique templates per fault; 0 uses all.",
    )
    parser.add_argument(
        "--pure-retraction",
        action="store_true",
        help="FailedUpdate only: Turn 2 only retracts Turn 1 without adding a new measurement.",
    )
    parser.add_argument(
        "--failed_update-case-strategy",
        type=str,
        default="default",
        choices=INERTIA_CASE_STRATEGIES,
        help="FailedUpdate only: template ordering strategy before max-templates truncation.",
    )
    parser.add_argument(
        "--failed_update-powered-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FailedUpdate only: discard templates containing isolation/ohmmeter checks.",
    )
    parser.add_argument(
        "--max-cases-per-fault",
        type=int,
        default=12,
        help="FailedStay only: maximum number of auto-generated templates retained per fault.",
    )
    parser.add_argument(
        "--max-powered-conditions",
        type=int,
        default=2,
        help="FailedStay only: maximum powered measurements combined into one candidate event.",
    )
    parser.add_argument(
        "--include-isolation-healthy",
        action="store_true",
        help="FailedStay only: include healthy isolation checks when enumerating candidates.",
    )
    parser.add_argument(
        "--failed_stay-case-strategy",
        type=str,
        default="multi_candidate_set_failed_stay_v9",
        choices=["multi_candidate_set_failed_stay_v9", "singleton_converge_broad_v1"],
        help="FailedStay only: fixed template selection strategy.",
    )
    parser.add_argument(
        "--failed_stay-interference-rounds",
        type=int,
        default=0,
        help=(
            "FailedStay only: append this many singleton-preserving post-convergence "
            "interference turns after the normal 3-turn failed_stay prefix."
        ),
    )
    parser.add_argument(
        "--failed_stay-blocks-per-post-turn",
        type=int,
        default=1,
        help=(
            "FailedStay only: pack up to this many post-convergence measurement "
            "blocks into each appended interference turn."
        ),
    )
    parser.add_argument(
        "--failed_stay-prefix-rule-prediction-turns",
        type=int,
        default=0,
        help=(
            "FailedStay only: include neutral rule-prediction tables for the first "
            "N turns only, using the measured keys accumulated so far."
        ),
    )
    parser.add_argument(
        "--noise-max-turns",
        type=int,
        default=6,
        help="Noise only: maximum interactive rounds before forced stop.",
    )
    parser.add_argument(
        "--only-template-indices-json",
        type=str,
        default="",
        help=(
            "Optional JSON object mapping task labels such as "
            "'parallel_r12_series:F' to template_idx lists. Used to rerun a "
            "selected subset, e.g. previously valid cases."
        ),
    )
    parser.set_defaults(add_host_comment=True)
    parser.add_argument(
        "--add-host-comment",
        dest="add_host_comment",
        action="store_true",
        help="Add host comments to noise mode feedback (default: True).",
    )
    parser.add_argument(
        "--no-add-host-comment",
        dest="add_host_comment",
        action="store_false",
        help="Disable host comments in noise mode feedback.",
    )
    parser.set_defaults(add_failed_isolation_comment=True)
    parser.add_argument(
        "--add-failed_isolation-comment",
        dest="add_failed_isolation_comment",
        action="store_true",
        help="Append a misleading host comment to every failed_isolation mode turn (default: True).",
    )
    parser.add_argument(
        "--no-add-failed_isolation-comment",
        dest="add_failed_isolation_comment",
        action="store_false",
        help="Disable failed_isolation mode host comment injection.",
    )
    parser.add_argument(
        "--strict-failed_isolation-prefix-turns",
        type=int,
        default=0,
        help=(
            "FailedIsolation only: require exact matches for this many prefix turns before "
            "counting later wrong hypotheses as valid failed_isolation data. "
            "0 preserves the original final-answer-only classifier."
        ),
    )
    parser.add_argument(
        "--failed_isolation-comment-start-turn",
        type=int,
        default=0,
        help="FailedIsolation only: start appending misleading comments at this zero-based turn.",
    )
    parser.add_argument(
        "--preserve-failed_isolation-turn-message",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "FailedIsolation only: preserve the normal turn prompt, including fault "
            "prediction tables, and append the misleading comment instead of "
            "rewriting the prompt to the short A-style clue format."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be >= 1")
    if args.repeats <= 0:
        raise ValueError("--repeats must be >= 1")
    if args.strict_failed_isolation_prefix_turns < 0:
        raise ValueError("--strict-failed_isolation-prefix-turns must be >= 0")
    if args.failed_isolation_comment_start_turn < 0:
        raise ValueError("--failed_isolation-comment-start-turn must be >= 0")
    if args.max_cases_per_fault <= 0:
        raise ValueError("--max-cases-per-fault must be >= 1")
    if args.max_powered_conditions <= 0:
        raise ValueError("--max-powered-conditions must be >= 1")
    if args.failed_stay_interference_rounds < 0:
        raise ValueError("--failed_stay-interference-rounds must be >= 0")
    if args.failed_stay_blocks_per_post_turn <= 0:
        raise ValueError("--failed_stay-blocks-per-post-turn must be >= 1")
    if args.failed_stay_prefix_rule_prediction_turns < 0:
        raise ValueError("--failed_stay-prefix-rule-prediction-turns must be >= 0")
    if args.noise_max_turns <= 0:
        raise ValueError("--noise-max-turns must be >= 1")
    if args.max_templates_per_fault < 0:
        raise ValueError("--max-templates-per-fault must be >= 0")
    if args.agent_max_tokens <= 0:
        raise ValueError("--agent-max-tokens must be >= 1")
    if args.context_turns < -1:
        raise ValueError("--context-turns must be >= -1")
    if not 0.0 <= args.agent_top_p <= 1.0:
        raise ValueError("--agent-top-p must be in [0, 1]")
    if args.agent_top_k < -1:
        raise ValueError("--agent-top-k must be >= -1")
    if args.agent_min_p < 0.0 or args.agent_min_p > 1.0:
        raise ValueError("--agent-min-p must be in [0, 1]")
    if args.prepare_workers <= 0:
        raise ValueError("--prepare-workers must be >= 1")
    if args.vllm_max_model_len <= 0:
        raise ValueError("--vllm-max-model-len must be >= 1")
    if args.vllm_max_num_seqs < 0:
        raise ValueError("--vllm-max-num-seqs must be >= 0")
    if args.vllm_max_num_batched_tokens < 0:
        raise ValueError("--vllm-max-num-batched-tokens must be >= 0")
    if args.context_turns == -1:
        args.context_turns = None
    if args.backend == "api" and not args.api_base_url:
        raise ValueError("--api-base-url is required when --backend api")
    if args.mode == "noise" and args.prompt_style != "noise":
        args.prompt_style = "noise"


def main() -> None:
    args = parse_args()
    _validate_args(args)
    args.only_template_indices = _load_template_index_filter(
        args.only_template_indices_json
    )

    if args.backend == "vllm":
        gpu_ids = _parse_gpu_ids(args.gpus)
        gpu_label = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_label
    else:
        gpu_ids = [0]
        gpu_label = "api"

    use_topology_default_faults = list(args.faults) == list(FAULT_IDS)
    all_tasks: List[Tuple[str, str]] = []
    for circuit_type in args.circuit_types:
        topology_faults = list(get_topology(circuit_type).fault_ids)
        selected_faults = topology_faults if use_topology_default_faults else [
            fault for fault in args.faults if fault in topology_faults
        ]
        all_tasks.extend((circuit_type, fault) for fault in selected_faults)
    task_labels = [f"{circuit_type}:{fault}" for circuit_type, fault in all_tasks]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.agent_model_path)
    output_dir = args.output_dir or os.path.join(
        "task_b",
        "outputs",
        f"{args.mode}_stats_{model_name}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    category_dirs = build_category_dirs(output_dir)
    for category_dir in category_dirs.values():
        os.makedirs(category_dir, exist_ok=True)

    _print_run_configuration(
        model_name=model_name,
        gpu_ids=gpu_ids,
        task_labels=task_labels,
        args=args,
    )

    template_db = (
        _load_template_db(args.templates_json)
        if args.mode in {"failed_update", "failed_stay", "failed_isolation"} and args.templates_json
        else None
    )

    points_by_task: Dict[str, List[Dict[str, Any]]] = {}
    sessions_by_task: Dict[str, List[Dict[str, Any]]] = {}
    template_counts: Dict[str, int] = {}
    all_sessions: List[Dict[str, Any]] = []
    active_task_labels: List[str] = []
    skipped_tasks: List[Dict[str, str]] = []

    for circuit_type, fault in all_tasks:
        task_label = f"{circuit_type}:{fault}"
        points, sessions, template_count = _prepare_task_data(
            circuit_type=circuit_type,
            fault=fault,
            args=args,
            template_db=template_db,
        )
        if not points:
            skipped_tasks.append({
                "task_label": task_label,
                "reason": f"no {args.mode} task points prepared",
            })
            print(
                f"Skipped {task_label}: no {args.mode} task points prepared"
            )
            continue

        points_by_task[task_label] = points
        sessions_by_task[task_label] = sessions
        template_counts[task_label] = template_count
        active_task_labels.append(task_label)
        all_sessions.extend(sessions)
        print(
            f"Prepared {task_label}: points={len(points)}, "
            f"sessions={len(sessions)}, templates={template_count}"
        )

    if not active_task_labels:
        raise ValueError(
            "No executable tasks prepared for the selected mode. "
            "Please relax filtering constraints or adjust mode-specific generation settings."
        )

    print(
        f"\n[GPU {gpu_label}] Session preparation complete: "
        f"mode={args.mode}, tasks={len(active_task_labels)}"
        f" (skipped={len(skipped_tasks)}), sessions={len(all_sessions)}"
    )
    if args.backend == "vllm":
        print(
            f"Loading vLLM: {args.agent_model_path} "
            f"(tensor_parallel_size={len(gpu_ids)})"
        )
        backend = VLLMBackend(
            model_path=args.agent_model_path,
            dtype="bf16",
            max_model_len=args.vllm_max_model_len,
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            language_model_only=args.vllm_language_model_only,
            max_num_seqs=args.vllm_max_num_seqs or None,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens or None,
        )
        print(f"[GPU {gpu_label}] vLLM backend ready.", flush=True)
    else:
        api_key = args.api_key or os.environ.get(args.api_key_env, "")
        api_model = args.api_model or args.agent_model_path
        print(f"Loading API backend: model={api_model}, endpoint={args.api_base_url}")
        backend = APIBackend(
            api_base_url=args.api_base_url,
            model_name=api_model,
            api_key=api_key or None,
        )
        print("[API] backend ready.", flush=True)

    backend.sampling_overrides = {
        "top_p": args.agent_top_p,
        "top_k": args.agent_top_k,
        "min_p": args.agent_min_p,
        "presence_penalty": args.agent_presence_penalty,
        "repetition_penalty": args.agent_repetition_penalty,
    }

    all_stats: Dict[str, Dict[str, int]] = {}

    def _print_task_summary(task_label: str) -> None:
        total_points = len(points_by_task[task_label])
        template_count = template_counts[task_label]
        print(
            f"\n  --- {task_label} summary "
            f"({args.num_runs} runs x {template_count} templates x {args.repeats} repeats) ---"
        )
        for category in CATEGORIES:
            n = all_stats[task_label][category]
            pct = n / total_points * 100 if total_points > 0 else 0
            print(f"  {category}: {n}/{total_points} ({pct:.1f}%)")

    if args.mode == "failed_update":
        run_semantics = "each run executes every template once"
    elif args.mode == "noise":
        run_semantics = "each run executes one interactive noise challenge"
    elif args.mode == "failed_isolation":
        run_semantics = (
            "each run executes every auto-generated failed_stay template once "
            "with a misleading host comment appended each turn"
        )
    else:
        run_semantics = "each run executes every auto-generated failed_stay template once"

    def _save_progress_report(completed_task_labels: Sequence[str]) -> None:
        report = build_stats_report(
            model=model_name,
            prompt_style=args.prompt_style,
            num_runs_per_task=args.num_runs,
            run_semantics=run_semantics,
            repeats=args.repeats,
            gpus=gpu_ids,
            circuit_types=args.circuit_types,
            faults=args.faults,
            per_task_counts=all_stats,
            template_counts=template_counts,
        )
        report["mode"] = args.mode
        report["debug_config"] = {
            "agent_temperature": args.agent_temperature,
            "agent_top_p": args.agent_top_p,
            "agent_top_k": args.agent_top_k,
            "agent_min_p": args.agent_min_p,
            "agent_presence_penalty": args.agent_presence_penalty,
            "agent_repetition_penalty": args.agent_repetition_penalty,
            "agent_max_tokens": args.agent_max_tokens,
            "context_turns": args.context_turns,
            "failed_update_case_strategy": args.failed_update_case_strategy,
            "max_templates_per_fault": args.max_templates_per_fault,
            "failed_update_powered_only": args.failed_update_powered_only,
            "noise_max_turns": args.noise_max_turns,
            "failed_stay_blocks_per_post_turn": args.failed_stay_blocks_per_post_turn,
            "failed_stay_prefix_rule_prediction_turns": args.failed_stay_prefix_rule_prediction_turns,
        }
        report["skipped_tasks"] = skipped_tasks
        report["completed_tasks"] = list(completed_task_labels)
        save_json(os.path.join(output_dir, "stats_report.json"), report)

    if args.mode == "failed_isolation":
        print(
            f"\n[GPU {gpu_label}] Starting task-wise inference: "
            f"mode={args.mode}, tasks={len(active_task_labels)}, sessions={len(all_sessions)}"
        )
        completed_task_labels: List[str] = []
        total_tasks = len(active_task_labels)
        for task_idx, task_label in enumerate(active_task_labels, start=1):
            task_sessions = sessions_by_task[task_label]
            task_checkpoint_dir = None
            if args.save_turn_checkpoints:
                task_checkpoint_dir = os.path.join(
                    output_dir,
                    "turn_checkpoints",
                    task_label.replace(":", "_"),
                )
            run_sessions_batched(
                run_label=f"task_b {args.mode} [{task_idx}/{total_tasks}] {task_label}",
                sessions=task_sessions,
                backend=backend,
                temperature=args.agent_temperature,
                max_tokens=args.agent_max_tokens,
                checkpoint_dir=task_checkpoint_dir,
            )
            item = _save_task_outputs(
                task_label=task_label,
                points=points_by_task[task_label],
                sessions=task_sessions,
                category_dirs=category_dirs,
                repeats=args.repeats,
            )
            all_stats[task_label] = item["counts"]
            completed_task_labels.append(task_label)
            _print_task_summary(task_label)
            _save_progress_report(completed_task_labels)
            print(
                f"[save-progress] task {task_idx}/{total_tasks} saved: {task_label}",
                flush=True,
            )

        _print_overall_report(
            model_name=model_name,
            prompt_style=args.prompt_style,
            mode=args.mode,
            task_labels=active_task_labels,
            all_stats=all_stats,
            template_counts=template_counts,
        )
        _save_progress_report(active_task_labels)
    else:
        print(
            f"\n[GPU {gpu_label}] Starting global inference: "
            f"mode={args.mode}, tasks={len(active_task_labels)}, sessions={len(all_sessions)}"
        )
        if args.mode == "noise":
            # Noise challenges are interactive query/feedback loops and should use
            # the orchestrator's noise runner to preserve mode-specific semantics.
            for idx, session in enumerate(all_sessions, start=1):
                orchestrator = session["orchestrator"]
                orchestrator.backend = backend
                session["trajectory"] = orchestrator.run()
                if idx % 10 == 0 or idx == len(all_sessions):
                    print(
                        f"  [noise] completed sessions: {idx}/{len(all_sessions)}",
                        flush=True,
                    )
        else:
            run_sessions_batched(
                run_label=f"task_b {args.mode} ({len(active_task_labels)} tasks)",
                sessions=all_sessions,
                backend=backend,
                temperature=args.agent_temperature,
                max_tokens=args.agent_max_tokens,
                checkpoint_dir=(
                    os.path.join(output_dir, "turn_checkpoints")
                    if args.save_turn_checkpoints
                    else None
                ),
            )

        for task_label in active_task_labels:
            item = _save_task_outputs(
                task_label=task_label,
                points=points_by_task[task_label],
                sessions=sessions_by_task[task_label],
                category_dirs=category_dirs,
                repeats=args.repeats,
            )
            all_stats[task_label] = item["counts"]
            _print_task_summary(task_label)

        _print_overall_report(
            model_name=model_name,
            prompt_style=args.prompt_style,
            mode=args.mode,
            task_labels=active_task_labels,
            all_stats=all_stats,
            template_counts=template_counts,
        )
        _save_progress_report(active_task_labels)

    report = build_stats_report(
        model=model_name,
        prompt_style=args.prompt_style,
        num_runs_per_task=args.num_runs,
        run_semantics=run_semantics,
        repeats=args.repeats,
        gpus=gpu_ids,
        circuit_types=args.circuit_types,
        faults=args.faults,
        per_task_counts=all_stats,
        template_counts=template_counts,
    )
    report["mode"] = args.mode
    report["debug_config"] = {
        "agent_temperature": args.agent_temperature,
        "agent_top_p": args.agent_top_p,
        "agent_top_k": args.agent_top_k,
        "agent_min_p": args.agent_min_p,
        "agent_presence_penalty": args.agent_presence_penalty,
        "agent_repetition_penalty": args.agent_repetition_penalty,
        "agent_max_tokens": args.agent_max_tokens,
        "context_turns": args.context_turns,
        "failed_update_case_strategy": args.failed_update_case_strategy,
        "max_templates_per_fault": args.max_templates_per_fault,
        "failed_update_powered_only": args.failed_update_powered_only,
        "noise_max_turns": args.noise_max_turns,
        "failed_stay_blocks_per_post_turn": args.failed_stay_blocks_per_post_turn,
        "failed_stay_prefix_rule_prediction_turns": args.failed_stay_prefix_rule_prediction_turns,
        "strict_failed_isolation_prefix_turns": args.strict_failed_isolation_prefix_turns,
        "failed_isolation_comment_start_turn": args.failed_isolation_comment_start_turn,
        "preserve_failed_isolation_turn_message": args.preserve_failed_isolation_turn_message,
    }
    report["skipped_tasks"] = skipped_tasks
    save_json(os.path.join(output_dir, "stats_report.json"), report)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
