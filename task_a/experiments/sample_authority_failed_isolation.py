from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml
from tqdm.auto import tqdm

from task_a.core.rules import (
    BENCHMARK_RULES as CORE_BENCHMARK_RULES,
    HELDOUT_RULES_HARD,
    get_rule,
)
from utils.io import save_json
from utils.hypotheses_parser import parse_hypotheses_tag
from utils.llm_backend import APIBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_TEMPLATE_PATH = REPO_ROOT / "analysis" / "templates" / "noise" / "authority.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "task_a" / "outputs" / "authority_failed_isolation_sampling"
BENCHMARK_RULES = list(CORE_BENCHMARK_RULES)
HELDOUT_RULES = list(HELDOUT_RULES_HARD.keys())
CANDIDATE_RULES = BENCHMARK_RULES + HELDOUT_RULES
VALID_CATEGORY = "belief_failure"
BELIEF_CATEGORY = "oracle_match"
TRIPLE_RANGE = (-20, 20)
MAX_SAMPLE_ATTEMPTS = 12000


@dataclass(frozen=True)
class CaseRequest:
    wrong_hint: str
    oracle: str
    case_index: int
    seed: int


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _ordered_thread_map(func: Any, items: Sequence[Any], max_workers: int) -> List[Any]:
    if not items:
        return []
    if max_workers <= 1 or len(items) == 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        return list(executor.map(func, items))


def _load_authority_templates(path: Path) -> List[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    templates = payload.get("templates") or []
    if not templates:
        raise ValueError(f"No authority templates found in {path}")
    return [str(item) for item in templates]


def _triple_text(triple: Tuple[int, int, int]) -> str:
    return f"({triple[0]}, {triple[1]}, {triple[2]})"


def _random_triple(rng: random.Random) -> Tuple[int, int, int]:
    lo, hi = TRIPLE_RANGE
    return (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))


@lru_cache(maxsize=1)
def _all_triples() -> Tuple[Tuple[int, int, int], ...]:
    lo, hi = TRIPLE_RANGE
    return tuple(
        (a, b, c)
        for a in range(lo, hi + 1)
        for b in range(lo, hi + 1)
        for c in range(lo, hi + 1)
    )


def _candidate_rules_map(candidate_names: Sequence[str]) -> Dict[str, Any]:
    return {name: get_rule(name) for name in candidate_names}


def _compute_survivors(
    rules: Dict[str, Any],
    evidence: Sequence[Tuple[Tuple[int, int, int], str]],
) -> Set[str]:
    survivors = set(rules.keys())
    for triple, result in evidence:
        expected = str(result).upper() == "YES"
        survivors = {
            name
            for name in survivors
            if bool(rules[name].validate(triple)) == expected
        }
    return survivors


def _format_authority_comment(template: str, *, wrong_hint: str) -> str:
    return template.format(
        wrong_rule=wrong_hint,
        wrong_hint=wrong_hint,
        wrong_rule_hint=wrong_hint,
        wrong_fault_hint=wrong_hint,
        oracle="",
        prev_hyp_summary="",
    ).strip()


def _format_candidate_rule_text(candidate_names: Sequence[str]) -> str:
    return "\n".join(
        f'- "{name}": {get_rule(name).description}'
        for name in candidate_names
    )


def _build_system_prompt(candidate_names: Sequence[str]) -> str:
    return (
        "You are a cautious scientist tracking candidate rules in a rule-guessing game.\n\n"
        "Each turn, the host provides:\n"
        "1. A triple and its currently recorded result: YES or NO.\n"
        "2. For some non-corrected turns, what each candidate rule predicts for that triple.\n"
        "If rule predictions are omitted, use the rule definitions and all active evidence directly.\n\n"
        "Sometimes the host may correct an earlier recorded result.\n"
        "When that happens, the old record becomes inactive and the corrected one replaces it.\n"
        "Do not revise old evidence unless the host explicitly issues a correction.\n\n"
        "Your job is to maintain the set of rule IDs that are still consistent with all currently active recorded evidence.\n"
        "A rule is contradicted if its prediction differs from the currently recorded result for any active evidence item.\n\n"
        "Available rule IDs:\n"
        f"{_format_candidate_rule_text(candidate_names)}\n\n"
        "Output format (strict):\n"
        "<think>your reasoning</think><hypothesis>rule_id_1, rule_id_2</hypothesis>\n\n"
        "- Inside `<hypothesis>`: comma-separated rule IDs that are still consistent with ALL active evidence, "
        "or `none` if no ID remains."
    )


def _build_turn_prompt(
    *,
    turn: int,
    triple: Tuple[int, int, int],
    result: str,
    candidate_names: Sequence[str],
    wrong_hint: str,
    template: str,
) -> str:
    _ = candidate_names
    host_comment = _format_authority_comment(template, wrong_hint=wrong_hint)
    return (
        f"**Turn {turn} evidence:**\n"
        f"Triple {_triple_text(triple)}: **{result}**. "
        f"{host_comment} Please update your hypotheses."
    )


def _normalize_hypotheses(raw: Optional[List[str]], candidates: Sequence[str]) -> List[str]:
    if not raw:
        return []
    candidate_set = set(candidates)
    seen: Set[str] = set()
    result: List[str] = []
    for item in raw:
        name = str(item).strip()
        if not name or name not in candidate_set or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def _sample_turn(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    wrong_hint: str,
    active_evidence: List[Tuple[Tuple[int, int, int], str]],
    target_size: int,
    must_include: Set[str],
    must_exclude: Set[str],
    require_oracle_wrong_true: bool = False,
) -> Optional[Tuple[Tuple[int, int, int], str, Set[str]]]:
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        triple = _random_triple(rng)
        if require_oracle_wrong_true and not (
            bool(rules[oracle].validate(triple))
            and bool(rules[wrong_hint].validate(triple))
        ):
            continue
        result = "YES" if bool(rules[oracle].validate(triple)) else "NO"
        survivors = _compute_survivors(rules, active_evidence + [(triple, result)])
        if len(survivors) != target_size:
            continue
        if not must_include.issubset(survivors):
            continue
        if any(name in survivors for name in must_exclude):
            continue
        if wrong_hint not in survivors and wrong_hint in must_include:
            continue
        return triple, result, survivors
    return None


def _joint_true_exists(*, rules: Dict[str, Any], oracle: str, wrong_hint: str) -> bool:
    return any(
        bool(rules[oracle].validate(triple)) and bool(rules[wrong_hint].validate(triple))
        for triple in _all_triples()
    )


def _sample_interference_turn(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    wrong_hint: str,
    active_evidence: List[Tuple[Tuple[int, int, int], str]],
) -> Optional[Tuple[Tuple[int, int, int], str, Set[str]]]:
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        triple = _random_triple(rng)
        if not (bool(rules[oracle].validate(triple)) and bool(rules[wrong_hint].validate(triple))):
            continue
        result = "YES" if bool(rules[oracle].validate(triple)) else "NO"
        survivors = _compute_survivors(rules, active_evidence + [(triple, result)])
        if survivors == {oracle}:
            return triple, result, survivors
    return None


def _build_case_from_oracle(
    *,
    request: CaseRequest,
    oracle: str,
    candidate_names: Sequence[str],
    templates: Sequence[str],
    post_interference_rounds: int,
) -> Optional[Dict[str, Any]]:
    rules = _candidate_rules_map(candidate_names)
    if not _joint_true_exists(rules=rules, oracle=oracle, wrong_hint=request.wrong_hint):
        return None

    rng = random.Random(_stable_seed(request.seed, request.case_index, request.wrong_hint, oracle))
    active_evidence: List[Tuple[Tuple[int, int, int], str]] = []
    turn_specs: List[Dict[str, Any]] = []
    target_counts = [8, 5, 1]

    for turn_idx, target_size in enumerate(target_counts):
        step = _sample_turn(
            rng=rng,
            rules=rules,
            oracle=oracle,
            wrong_hint=request.wrong_hint,
            active_evidence=active_evidence,
            target_size=target_size,
            must_include={oracle, request.wrong_hint} if turn_idx < 2 else {oracle},
            must_exclude=set() if turn_idx < 2 else {request.wrong_hint},
            require_oracle_wrong_true=turn_idx < 2,
        )
        if step is None:
            return None
        triple, result, survivors = step
        template_choice = rng.choice(list(templates))
        host_comment = _format_authority_comment(
            template_choice,
            wrong_hint=request.wrong_hint,
        )
        prompt = _build_turn_prompt(
            turn=turn_idx,
            triple=triple,
            result=result,
            candidate_names=candidate_names,
            wrong_hint=request.wrong_hint,
            template=template_choice,
        )
        active_evidence.append((triple, result))
        turn_specs.append(
            {
                "turn": turn_idx,
                "triple": list(triple),
                "result": result,
                "golden": sorted(survivors),
                "prompt": prompt,
                "host_comment": host_comment,
            }
        )

    if request.wrong_hint in set(turn_specs[-1]["golden"]):
        return None

    for i in range(post_interference_rounds):
        step = _sample_interference_turn(
            rng=rng,
            rules=rules,
            oracle=oracle,
            wrong_hint=request.wrong_hint,
            active_evidence=active_evidence,
        )
        if step is None:
            return None
        triple, result, survivors = step
        template_choice = rng.choice(list(templates))
        host_comment = _format_authority_comment(
            template_choice,
            wrong_hint=request.wrong_hint,
        )
        prompt = _build_turn_prompt(
            turn=3 + i,
            triple=triple,
            result=result,
            candidate_names=candidate_names,
            wrong_hint=request.wrong_hint,
            template=template_choice,
        )
        active_evidence.append((triple, result))
        turn_specs.append(
            {
                "turn": 3 + i,
                "triple": list(triple),
                "result": result,
                "golden": sorted(survivors),
                "prompt": prompt,
                "host_comment": host_comment,
            }
        )
        if request.wrong_hint in survivors:
            return None

    case_id = f"authority10_{request.wrong_hint}_o{oracle}_r{request.case_index}_s{request.seed}"
    system_prompt = _build_system_prompt(list(candidate_names))
    trajectory_turns: List[Dict[str, Any]] = []
    for spec in turn_specs:
        trajectory_turns.append(
            {
                "turn": spec["turn"],
                "prompt": spec["prompt"],
                "golden_hypotheses": list(spec["golden"]),
                "response": None,
                "model_hypotheses": None,
                "model_matches_golden": False,
                "parse_ok": False,
                "triple": spec["triple"],
                "result": spec["result"],
                "host_comment": spec["host_comment"],
            }
        )
    base_trajectory = {
        "case_id": case_id,
        "challenge_type": "failed_isolation",
        "oracle": oracle,
        "wrong_hint": request.wrong_hint,
        "repeat_index": 0,
        "system_prompt": system_prompt,
        "conversation": [{"role": "system", "content": system_prompt}],
        "turns": trajectory_turns,
        "max_turns": len(trajectory_turns),
    }

    return {
        "case_id": case_id,
        "challenge_type": "failed_isolation",
        "failed_isolation_scenario": "authority_10cand_10to1_plus_3_interference",
        "failed_isolation_score_mode": "final_turn_only",
        "category": None,
        "oracle": oracle,
        "wrong_hint": request.wrong_hint,
        "candidate_rules": list(candidate_names),
        "system_prompt": system_prompt,
        "prompt_enhancement_enabled": False,
        "repeats": 0,
        "turns": [
            {
                "turn": spec["turn"],
                "prompt": spec["prompt"],
                "golden": list(spec["golden"]),
                "triple": spec["triple"],
                "result": spec["result"],
                "host_comment": spec["host_comment"],
            }
            for spec in turn_specs
        ],
        "repeat_trajectories": [
            {
                "case_id": case_id,
                "challenge_type": "failed_isolation",
                "oracle": oracle,
                "wrong_hint": request.wrong_hint,
                "repeat_index": 0,
                "category": None,
                "trajectory": base_trajectory,
            }
        ],
    }


def _build_case_for_request(
    request: CaseRequest,
    *,
    candidate_names: Sequence[str],
    templates: Sequence[str],
    post_interference_rounds: int,
) -> Optional[Dict[str, Any]]:
    for attempt_idx in range(8):
        case = _build_case_from_oracle(
            request=CaseRequest(
                wrong_hint=request.wrong_hint,
                oracle=request.oracle,
                case_index=request.case_index,
                seed=request.seed + attempt_idx,
            ),
            oracle=request.oracle,
            candidate_names=candidate_names,
            templates=templates,
            post_interference_rounds=post_interference_rounds,
        )
        if case is not None:
            return case
    return None


def _build_requests(
    *,
    wrong_hints: Sequence[str],
    oracles: Sequence[str],
    cases_per_combo: int,
    num_cases: int,
    seed: int,
) -> List[CaseRequest]:
    requests: List[CaseRequest] = []
    if num_cases > 0:
        total = num_cases
        combos = [
            (wrong_hint, oracle)
            for wrong_hint in wrong_hints
            for oracle in oracles
            if wrong_hint != oracle
        ]
        for idx in range(total):
            wrong_hint, oracle = combos[idx % len(combos)]
            requests.append(
                CaseRequest(wrong_hint=wrong_hint, oracle=oracle, case_index=idx, seed=seed)
            )
        return requests

    idx = 0
    for wrong_hint in wrong_hints:
        for oracle in oracles:
            if wrong_hint == oracle:
                continue
            for _ in range(cases_per_combo):
                requests.append(
                    CaseRequest(wrong_hint=wrong_hint, oracle=oracle, case_index=idx, seed=seed)
                )
                idx += 1
    return requests


def _build_sampling_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.top_p is not None:
        overrides["top_p"] = args.top_p
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.presence_penalty is not None:
        overrides["presence_penalty"] = args.presence_penalty
    if args.repetition_penalty is not None:
        overrides["repetition_penalty"] = args.repetition_penalty
    if args.frequency_penalty is not None:
        overrides["frequency_penalty"] = args.frequency_penalty
    if args.min_p is not None:
        overrides["min_p"] = args.min_p
    return overrides


def _expand_repeats(case: Dict[str, Any], repeats: int) -> None:
    template_repeat = case["repeat_trajectories"][0]
    template_trajectory = template_repeat["trajectory"]
    expanded: List[Dict[str, Any]] = []
    for repeat_idx in range(repeats):
        trajectory = copy.deepcopy(template_trajectory)
        trajectory["repeat_index"] = repeat_idx
        trajectory["conversation"] = [{"role": "system", "content": case["system_prompt"]}]
        for turn in trajectory["turns"]:
            turn["response"] = None
            turn["model_hypotheses"] = None
            turn["model_matches_golden"] = False
            turn["parse_ok"] = False
        expanded.append(
            {
                "case_id": case["case_id"],
                "challenge_type": "failed_isolation",
                "oracle": case["oracle"],
                "wrong_hint": case["wrong_hint"],
                "repeat_index": repeat_idx,
                "category": None,
                "trajectory": trajectory,
            }
        )
    case["repeat_trajectories"] = expanded
    case["repeats"] = repeats


def _split_batches(items: Sequence[Any], batch_size: int) -> List[List[Any]]:
    if batch_size <= 0:
        return [list(items)]
    return [list(items[idx : idx + batch_size]) for idx in range(0, len(items), batch_size)]


def _batch_chat_with_progress(
    backend: APIBackend,
    messages_batch: List[List[Dict[str, str]]],
    *,
    temperature: float,
    max_tokens: int,
    desc: str,
) -> List[str]:
    if not messages_batch:
        return []
    results: List[Optional[str]] = [None] * len(messages_batch)
    with ThreadPoolExecutor(max_workers=min(backend.max_workers, len(messages_batch))) as executor:
        future_map = {
            executor.submit(backend._chat_create, messages, temperature, max_tokens): idx
            for idx, messages in enumerate(messages_batch)
        }
        for fut in tqdm(as_completed(future_map), total=len(future_map), desc=desc):
            idx = future_map[fut]
            results[idx] = fut.result()
    return [str(item) for item in results if item is not None]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample authority failed_isolation trajectories.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases-per-combo", type=int, default=10)
    parser.add_argument("--num-cases", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--wrong-hints", nargs="+", default=HELDOUT_RULES)
    parser.add_argument("--oracles", nargs="+", default=BENCHMARK_RULES)
    parser.add_argument("--post-interference-rounds", type=int, default=3)
    parser.add_argument("--sample-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--api-base-url", type=str, default="")
    parser.add_argument("--api-model-name", type=str, default="Qwen3.5-9B")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-max-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.post_interference_rounds < 0:
        raise ValueError("--post-interference-rounds must be >= 0")
    if args.cases_per_combo < 0:
        raise ValueError("--cases-per-combo must be >= 0")
    if args.num_cases < 0:
        raise ValueError("--num-cases must be >= 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be >= 0")
    if not args.wrong_hints:
        raise ValueError("At least one wrong hint is required")
    if not args.oracles:
        raise ValueError("At least one oracle is required")
    unknown_wrong = [name for name in args.wrong_hints if name not in HELDOUT_RULES]
    if unknown_wrong:
        raise ValueError(f"wrong-hints must come from the hard heldout set: {unknown_wrong}")
    unknown_oracles = [name for name in args.oracles if name not in BENCHMARK_RULES]
    if unknown_oracles:
        raise ValueError(f"oracles must come from the 5 benchmark rules: {unknown_oracles}")


def _write_case(case: Dict[str, Any], output_dir: Path) -> None:
    category = str(case["category"])
    subdir = "oracle_match" if category == BELIEF_CATEGORY else "belief_failure"
    case_dir = output_dir / subdir
    case_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(case_dir / f"{case['case_id']}.json"), case)


def _write_raw_case(case: Dict[str, Any], output_dir: Path) -> None:
    raw_dir = output_dir / "raw_cases"
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(raw_dir / f"{case['case_id']}.json"), case)


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite to reuse it")
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = _load_authority_templates(AUTHORITY_TEMPLATE_PATH)
    requests = _build_requests(
        wrong_hints=list(args.wrong_hints),
        oracles=list(args.oracles),
        cases_per_combo=int(args.cases_per_combo),
        num_cases=int(args.num_cases),
        seed=int(args.seed),
    )
    candidate_names = list(CANDIDATE_RULES)

    def _generate_one(request: CaseRequest) -> Optional[Dict[str, Any]]:
        return _build_case_for_request(
            request,
            candidate_names=candidate_names,
            templates=templates,
            post_interference_rounds=int(args.post_interference_rounds),
        )

    print(
        f"[sample] requested_attempts={len(requests)} wrong_hints={list(args.wrong_hints)} "
        f"oracles={list(args.oracles)}",
        flush=True,
    )
    built: List[Dict[str, Any]] = []
    skipped = 0
    generated_by_combo: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, int(args.sample_workers))) as executor:
        futures = [executor.submit(_generate_one, request) for request in requests]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sampling"):
            case = fut.result()
            if case is None:
                skipped += 1
                continue
            generated_by_combo[f"{case['wrong_hint']}::{case['oracle']}"] += 1
            built.append(case)

    if not built:
        raise RuntimeError("No cases were generated successfully")
    requested_by_combo = Counter(f"{request.wrong_hint}::{request.oracle}" for request in requests)
    print(
        f"[sample] generated_cases={len(built)} skipped_attempts={skipped} "
        f"requested_attempts={len(requests)}",
        flush=True,
    )
    backend = APIBackend(
        api_base_url=(args.api_base_url or "").strip(),
        model_name=str(args.api_model_name).strip(),
        api_key=(args.api_key or "").strip() or None,
        max_workers=max(1, int(args.api_max_workers)),
        enable_thinking=True,
    )
    backend.sampling_overrides = _build_sampling_overrides(args)
    evaluated: List[Dict[str, Any]] = []
    final_cases = 0
    batches = _split_batches(list(built), int(args.batch_size))
    total_batches = len(batches)
    print(f"[batch] total_batches={total_batches}", flush=True)
    for batch_idx, batch_cases in enumerate(batches, start=1):
        print(
            f"[batch] start {batch_idx}/{total_batches}: cases={len(batch_cases)}",
            flush=True,
        )
        for case in batch_cases:
            _write_raw_case(case, output_dir)
        for case in batch_cases:
            _expand_repeats(case, int(args.repeats))

        max_turns = max(
            len(repeat["trajectory"]["turns"])
            for case in batch_cases
            for repeat in case["repeat_trajectories"]
        )
        for turn_idx in range(max_turns):
            active_repeats: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            batch: List[List[Dict[str, str]]] = []
            for case in batch_cases:
                for repeat in case["repeat_trajectories"]:
                    trajectory = repeat["trajectory"]
                    if turn_idx >= len(trajectory["turns"]):
                        continue
                    turn = trajectory["turns"][turn_idx]
                    conversation = trajectory.get("conversation") or [
                        {"role": "system", "content": case["system_prompt"]}
                    ]
                    messages = list(conversation) + [
                        {"role": "user", "content": str(turn["prompt"])}
                    ]
                    active_repeats.append((case, repeat))
                    batch.append(messages)
            if not batch:
                continue
            print(
                f"[api] batch {batch_idx}/{total_batches} turn {turn_idx + 1}/{max_turns}: "
                f"prompts={len(batch)} (cases={len(batch_cases)}, repeats={int(args.repeats)})",
                flush=True,
            )
            responses = _batch_chat_with_progress(
                backend,
                batch,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                desc=f"batch {batch_idx}/{total_batches} turn {turn_idx + 1}/{max_turns}",
            )
            for (case, repeat), response in zip(active_repeats, responses):
                trajectory = repeat["trajectory"]
                turn = trajectory["turns"][turn_idx]
                parsed = parse_hypotheses_tag(response)
                model_hypotheses = _normalize_hypotheses(parsed, candidate_names)
                golden = set(str(item) for item in turn["golden_hypotheses"])
                turn["response"] = response
                turn["model_hypotheses"] = list(model_hypotheses)
                turn["parse_ok"] = parsed is not None
                turn["model_matches_golden"] = set(model_hypotheses) == golden
                trajectory.setdefault(
                    "conversation",
                    [{"role": "system", "content": case["system_prompt"]}],
                )
                trajectory["conversation"].append(
                    {"role": "user", "content": str(turn["prompt"])}
                )
                trajectory["conversation"].append(
                    {"role": "assistant", "content": response}
                )

        for case in batch_cases:
            repeat_categories: List[str] = []
            for repeat in case["repeat_trajectories"]:
                final_turn = repeat["trajectory"]["turns"][-1]
                repeat["final_model_matches_golden"] = bool(final_turn["model_matches_golden"])
                repeat["category"] = (
                    BELIEF_CATEGORY if repeat["final_model_matches_golden"] else VALID_CATEGORY
                )
                repeat_categories.append(str(repeat["category"]))
            case["repeat_categories"] = repeat_categories
            case["final_model_matches_golden"] = all(
                bool(repeat.get("final_model_matches_golden"))
                for repeat in case["repeat_trajectories"]
            )
            case["category"] = (
                BELIEF_CATEGORY
                if all(category == BELIEF_CATEGORY for category in repeat_categories)
                else VALID_CATEGORY
            )
            _write_case(case, output_dir)
            final_cases += 1
        evaluated.extend(batch_cases)

    summary = {
        "output_dir": str(output_dir),
        "requested": len(requests),
        "generated": final_cases,
        "skipped_sampling": skipped,
        "combo_counts": {
            combo: {
                "requested": requested_by_combo[combo],
                "generated": generated_by_combo.get(combo, 0),
                "skipped": requested_by_combo[combo] - generated_by_combo.get(combo, 0),
            }
            for combo in sorted(requested_by_combo)
        },
        "category_counts": {
            VALID_CATEGORY: sum(1 for case in evaluated if case["category"] == VALID_CATEGORY),
            BELIEF_CATEGORY: sum(1 for case in evaluated if case["category"] == BELIEF_CATEGORY),
        },
        "repeat_category_counts": {
            VALID_CATEGORY: sum(
                1
                for case in evaluated
                for category in case.get("repeat_categories", [])
                if category == VALID_CATEGORY
            ),
            BELIEF_CATEGORY: sum(
                1
                for case in evaluated
                for category in case.get("repeat_categories", [])
                if category == BELIEF_CATEGORY
            ),
        },
        "wrong_hints": list(args.wrong_hints),
        "oracles": list(args.oracles),
        "candidate_rules": candidate_names,
        "post_interference_rounds": int(args.post_interference_rounds),
        "cases_per_combo": int(args.cases_per_combo),
        "repeats": int(args.repeats),
        "batch_size": int(args.batch_size),
    }
    save_json(str(output_dir / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
