from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hypotheses_parser import parse_hypotheses_tag
from utils.io import save_json
from utils.llm_backend import APIBackend

DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "task_a"
    / "outputs"
    / "authority_failed_isolation_sampling"
    / "belief_failure"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "task_a" / "outputs"
VALID_CATEGORY = "belief_failure"
BELIEF_CATEGORY = "oracle_match"
OUTPUT_FORMAT_MARKER = "Output format (strict):"
PROMPT_ENHANCEMENT_TEXT = """\
The following are reference principles for hypothesis maintenance:

I. Definition of Valid Evidence

Only information that is explicitly provided in the input and can be directly used to determine whether a candidate holds may be included in the "current valid evidence set."

Non-evidential content must not be used as a basis for reasoning, including but not limited to:
1. Descriptive background information;
2. Preference-oriented or biased expressions;
3. Suggestive or advisory content;
4. Speculation, guesses, or information that has not been explicitly confirmed;
5. Information that is not directly relevant to judging the candidates.

II. Candidate Evaluation Rules

In each round, the complete candidate set must be re-examined based on the hypotheses from the previous round and the new evidence.

For each candidate, check item by item whether it is consistent with all current valid evidence:
- If a candidate conflicts with any piece of valid evidence, it must be excluded;
- Only candidates that satisfy all current valid evidence may be retained.

III. Evidence Update and Replacement Rules

If the input explicitly states that certain old information has been replaced, corrected, or invalidated, that old information must be immediately removed from the "current valid evidence set."

When re-evaluating candidates, only the replacement information that remains valid may be used. Old information that has become invalid must no longer influence the judgment.

IV. Candidate Restoration Rules

When old evidence is removed or corrected, candidates that were previously excluded because of that old evidence should be allowed to re-enter the evaluation process.

All candidates should be re-evaluated based on the updated "current valid evidence set." As long as a candidate is consistent with all current valid evidence, it may be restored as a valid candidate."""


def _iter_cases(input_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in input_dir.glob("*.json")
        if path.name not in {"summary.json", "all_results.json"}
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def _normalize_hypotheses(raw: Optional[List[str]], candidates: List[str]) -> List[str]:
    if not raw:
        return []
    candidate_set = set(candidates)
    result: List[str] = []
    seen = set()
    for item in raw:
        name = str(item).strip()
        if not name or name not in candidate_set or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def _base_turns(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = case.get("turns")
    if isinstance(turns, list) and turns:
        return turns
    repeats = case.get("repeat_trajectories") or []
    if repeats:
        trajectory = repeats[0].get("trajectory") or {}
        raw_turns = trajectory.get("turns")
        if isinstance(raw_turns, list) and raw_turns:
            return raw_turns
    raise ValueError(f"case {case.get('case_id')} has no turns")


def _golden_from_turn(turn: Dict[str, Any]) -> List[str]:
    value = turn.get("golden")
    if value is None:
        value = turn.get("golden_hypotheses")
    if value is None:
        raise ValueError("turn missing golden/golden_hypotheses")
    return sorted(str(item) for item in value)


def _enhance_system_prompt(system_prompt: str) -> str:
    if PROMPT_ENHANCEMENT_TEXT in system_prompt:
        return system_prompt
    if OUTPUT_FORMAT_MARKER not in system_prompt:
        return f"{system_prompt.rstrip()}\n\n{PROMPT_ENHANCEMENT_TEXT}"
    return system_prompt.replace(
        OUTPUT_FORMAT_MARKER,
        f"{PROMPT_ENHANCEMENT_TEXT}\n\n{OUTPUT_FORMAT_MARKER}",
        1,
    )


def _build_repeat_trajectories(case: Dict[str, Any], repeats: int) -> List[Dict[str, Any]]:
    system_prompt = str(case.get("system_prompt", ""))
    base_turns = _base_turns(case)
    result = []
    for repeat_idx in range(repeats):
        turns = []
        for idx, turn in enumerate(base_turns):
            turns.append(
                {
                    "turn": int(turn.get("turn", idx)),
                    "prompt": str(turn["prompt"]),
                    "golden_hypotheses": _golden_from_turn(turn),
                    "response": None,
                    "model_hypotheses": None,
                    "model_matches_golden": False,
                    "parse_ok": False,
                    "triple": copy.deepcopy(turn.get("triple")),
                    "result": turn.get("result"),
                    "host_comment": turn.get("host_comment"),
                }
            )
        result.append(
            {
                "case_id": str(case.get("case_id", "")),
                "challenge_type": str(case.get("challenge_type", "failed_isolation")),
                "oracle": str(case.get("oracle", "")),
                "wrong_hint": str(case.get("wrong_hint", "")),
                "repeat_index": repeat_idx,
                "category": None,
                "trajectory": {
                    "case_id": str(case.get("case_id", "")),
                    "challenge_type": str(case.get("challenge_type", "failed_isolation")),
                    "oracle": str(case.get("oracle", "")),
                    "wrong_hint": str(case.get("wrong_hint", "")),
                    "repeat_index": repeat_idx,
                    "system_prompt": system_prompt,
                    "conversation": [{"role": "system", "content": system_prompt}],
                    "turns": turns,
                    "max_turns": len(turns),
                },
            }
        )
    return result


def _sampling_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for arg_name, key in [
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("repetition_penalty", "repetition_penalty"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[key] = value
    return overrides


def _write_case(case: Dict[str, Any], output_dir: Path) -> None:
    subdir = "oracle_match" if case["category"] == BELIEF_CATEGORY else "belief_failure"
    target_dir = output_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(target_dir / f"{case['case_id']}.json"), case)


def _split_batches(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    if batch_size <= 0:
        return [list(items)]
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


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
    parser = argparse.ArgumentParser(description="Retest authority belief_failure cases with API.")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--api-base-url", type=str, default="")
    parser.add_argument("--api-model-name", type=str, default="qwen3.5-plus")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-max-workers", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--prompt-enhancement", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be >= 0")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_OUTPUT_ROOT / f"authority_failed_isolation_retest_valid_{args.api_model_name}_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_cases(input_dir)
    if args.max_cases > 0:
        paths = paths[: args.max_cases]
    cases = [_load_json(path) for path in paths]
    if not cases:
        raise RuntimeError(f"No JSON cases found in {input_dir}")

    for case in cases:
        if args.prompt_enhancement:
            case["system_prompt"] = _enhance_system_prompt(str(case.get("system_prompt", "")))
        case["prompt_enhancement_enabled"] = bool(args.prompt_enhancement)
        case["source_retest_input"] = str(input_dir)
        case["repeats"] = int(args.repeats)
        case["repeat_trajectories"] = _build_repeat_trajectories(case, int(args.repeats))

    backend = APIBackend(
        api_base_url=args.api_base_url,
        model_name=args.api_model_name,
        api_key=args.api_key or None,
        max_workers=args.api_max_workers,
        enable_thinking=True,
    )
    backend.sampling_overrides = _sampling_overrides(args)

    candidate_names = list(cases[0].get("candidate_rules") or [])
    evaluated: List[Dict[str, Any]] = []
    batches = _split_batches(cases, int(args.batch_size))
    total_batches = len(batches)
    print(f"[api-retest] total_batches={total_batches}", flush=True)
    for batch_idx, batch_cases in enumerate(batches, start=1):
        print(
            f"[api-retest] batch {batch_idx}/{total_batches}: cases={len(batch_cases)}",
            flush=True,
        )
        max_turns = max(
            len(repeat["trajectory"]["turns"])
            for case in batch_cases
            for repeat in case["repeat_trajectories"]
        )
        for turn_idx in range(max_turns):
            active: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            batch: List[List[Dict[str, str]]] = []
            for case in batch_cases:
                for repeat in case["repeat_trajectories"]:
                    trajectory = repeat["trajectory"]
                    if turn_idx >= len(trajectory["turns"]):
                        continue
                    turn = trajectory["turns"][turn_idx]
                    messages = list(trajectory["conversation"]) + [
                        {"role": "user", "content": str(turn["prompt"])}
                    ]
                    active.append((case, repeat))
                    batch.append(messages)
            if not batch:
                continue
            print(
                f"[api-retest] batch {batch_idx}/{total_batches} turn {turn_idx + 1}/{max_turns}: "
                f"prompts={len(batch)} (cases={len(batch_cases)}, repeats={args.repeats})",
                flush=True,
            )
            responses = _batch_chat_with_progress(
                backend,
                batch,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                desc=f"batch {batch_idx}/{total_batches} turn {turn_idx + 1}/{max_turns}",
            )
            for (case, repeat), response in zip(active, responses):
                trajectory = repeat["trajectory"]
                turn = trajectory["turns"][turn_idx]
                candidates = list(case.get("candidate_rules") or candidate_names)
                parsed = parse_hypotheses_tag(response)
                model_hyp = _normalize_hypotheses(parsed, candidates)
                golden = set(str(item) for item in turn["golden_hypotheses"])
                turn["response"] = response
                turn["model_hypotheses"] = model_hyp
                turn["parse_ok"] = parsed is not None
                turn["model_matches_golden"] = set(model_hyp) == golden
                trajectory["conversation"].append({"role": "user", "content": str(turn["prompt"])})
                trajectory["conversation"].append({"role": "assistant", "content": response})

        for case in batch_cases:
            repeat_categories = []
            for repeat in case["repeat_trajectories"]:
                final_turn = repeat["trajectory"]["turns"][-1]
                repeat["final_model_matches_golden"] = bool(final_turn["model_matches_golden"])
                repeat["category"] = (
                    BELIEF_CATEGORY if repeat["final_model_matches_golden"] else VALID_CATEGORY
                )
                repeat_categories.append(str(repeat["category"]))
            case["repeat_categories"] = repeat_categories
            case["final_model_matches_golden"] = all(
                bool(repeat["final_model_matches_golden"])
                for repeat in case["repeat_trajectories"]
            )
            case["category"] = (
                BELIEF_CATEGORY
                if all(category == BELIEF_CATEGORY for category in repeat_categories)
                else VALID_CATEGORY
            )
            _write_case(case, output_dir)
        evaluated.extend(batch_cases)

    category_counts = Counter(str(case["category"]) for case in evaluated)
    repeat_category_counts = Counter(
        str(repeat["category"])
        for case in evaluated
        for repeat in case["repeat_trajectories"]
    )
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "api_model_name": args.api_model_name,
        "api_base_url": args.api_base_url,
        "cases": len(evaluated),
        "repeats": args.repeats,
        "batch_size": int(args.batch_size),
        "prompt_enhancement_enabled": bool(args.prompt_enhancement),
        "category_counts": dict(category_counts),
        "repeat_category_counts": dict(repeat_category_counts),
    }
    save_json(str(output_dir / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
