from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hypotheses_parser import parse_hypotheses_tag  # noqa: E402
from utils.llm_backend import VLLMBackend  # noqa: E402
from utils.belieftrack_constants import (  # noqa: E402
    BELIEF_FAILURE,
    CBM_CHALLENGE_TYPES,
    CBM_EVAL_CATEGORIES,
    FAILED_ISOLATION,
    FAILED_UPDATE,
    ORACLE_MATCH,
    canonicalize_category_counts,
    cbm_failure_metric,
    normalize_cbm_challenge_type,
    percentages_from_counts,
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HYP_LINE_RE = re.compile(r"(?ims)^[ \t]*<hypothesis>(.*?)</hypothesis>[ \t]*$")
_HYP_ANY_RE = re.compile(r"(?is)(?<!`)<hypothesis>(.*?)(?<!`)</hypothesis>")
_ITEM_SEP_RE = re.compile(r"[,，\s]+")

CATEGORY_INSUFFICIENT = "insufficient_capability"
CATEGORY_VALID = BELIEF_FAILURE
CATEGORY_BELIEF = ORACLE_MATCH
CATEGORY_PARSE_ERROR = "parse_error"
EVAL_CATEGORIES = list(CBM_EVAL_CATEGORIES)
EVAL_TYPES = CBM_CHALLENGE_TYPES
SKIP_CASE_FILENAMES = {"stats_report.json", "summary.json", "comparison.json", "train_cases_7B.json", "train_cases_9B.json"}
PROMPT_ENHANCEMENT_MARKER = "Output format (strict):"
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


def _build_eval_category_dirs(output_dir: str) -> Dict[str, str]:
    return {
        CATEGORY_INSUFFICIENT: f"{output_dir}/insufficient_capability",
        CATEGORY_BELIEF: f"{output_dir}/{CATEGORY_BELIEF}",
        CATEGORY_VALID: f"{output_dir}/{CATEGORY_VALID}",
        CATEGORY_PARSE_ERROR: f"{output_dir}/parse_error",
    }


def _aggregate_repeat_categories_eval(per_run_categories: List[str]) -> str:
    if any(category == CATEGORY_VALID for category in per_run_categories):
        return CATEGORY_VALID
    if any(category == CATEGORY_INSUFFICIENT for category in per_run_categories):
        return CATEGORY_INSUFFICIENT
    return CATEGORY_BELIEF


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scenario A test cases with belief_stats-style direct vLLM inference.")
    parser.add_argument("--test-data", required=True, type=str)
    parser.add_argument(
        "--eval-types",
        nargs="+",
        choices=EVAL_TYPES,
        default=list(EVAL_TYPES),
        help="Challenge type subdirectories to evaluate when --test-data is a directory.",
    )
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--eval-target", choices=("base", "lora", "both"), default="both")
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--lora-source-type", choices=("merged", "adapter"), default="merged")
    parser.add_argument("--lora-model-path", type=str, default=None)
    parser.add_argument("--lora-adapter-path", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--model-dtype", type=str, default="bfloat16")
    parser.add_argument("--disable-custom-all-reduce", action="store_true", default=False)
    parser.add_argument("--agent-temperature", "--temperature", dest="agent_temperature", type=float, default=0.3)
    parser.add_argument("--agent-max-tokens", "--max-output-tokens", "--max-tokens", dest="agent_max_tokens", type=int, default=8192)
    parser.add_argument("--sampling-top-p", "--top-p", dest="sampling_top_p", type=float, default=None)
    parser.add_argument("--sampling-top-k", "--top-k", dest="sampling_top_k", type=int, default=None)
    parser.add_argument("--sampling-presence-penalty", "--presence-penalty", dest="sampling_presence_penalty", type=float, default=None)
    parser.add_argument(
        "--sampling-repetition-penalty",
        "--repetition-penalty",
        dest="sampling_repetition_penalty",
        type=float,
        default=None,
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--enable-prompt-enhancement", action="store_true", default=False)
    parser.add_argument("--eval-mode", choices=("belief", "failed_isolation"), default="belief")
    parser.add_argument(
        "--enable-thinking",
        choices=("auto", "true", "false"),
        default="auto",
        help="Pass enable_thinking to chat templates that support it.",
    )
    parser.add_argument("--failed_isolation-scenario", choices=("auto", "a", "b"), default="auto")
    parser.add_argument("--failed_isolation-strict-prefix-turns", type=int, default=-1)
    parser.add_argument(
        "--failed_isolation-score-mode",
        choices=("auto", "strict-prefix", "final-oracle"),
        default="auto",
    )
    parser.add_argument("--failed_isolation-aggregate", choices=("majority", "any", "all"), default="majority")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate at most N cases after loading (stable order: sort by case_id). Omit or use 0 for all.",
    )
    return parser.parse_args()


def _apply_max_cases(cases: List[Dict[str, Any]], args: argparse.Namespace | None) -> List[Dict[str, Any]]:
    if args is None:
        return cases
    limit = getattr(args, "max_cases", None)
    if limit is None or limit <= 0 or len(cases) <= limit:
        return cases
    sorted_cases = sorted(cases, key=lambda c: str(c.get("case_id") or c.get("id") or ""))
    selected = sorted_cases[:limit]
    print(
        f"[eval-test-cases] max-cases={limit}: using {len(selected)}/{len(cases)} (sorted by case_id)",
        flush=True,
    )
    return selected


def _load_cases(path: str, args: argparse.Namespace | None = None) -> List[Dict[str, Any]]:
    root = Path(path)
    if root.is_dir():
        cases = _load_cases_from_dir(root, args)
    else:
        payload = json.loads(root.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} is not a case list")
        cases = payload
    return _apply_max_cases(cases, args)


def _case_type_dir_name(challenge_type: str) -> str:
    return _normalize_challenge_type(challenge_type)


def _load_case_file(path: Path, challenge_type: str) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        cases = payload
    elif isinstance(payload, dict):
        cases = [payload]
    else:
        raise ValueError(f"{path} is not a case object or case list")
    normalized = _case_type_dir_name(challenge_type)
    result: List[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"{path} contains a non-object case")
        case = dict(case)
        case["cbm_challenge_type"] = normalized
        result.append(case)
    return result


def _load_cases_from_dir(root: Path, args: argparse.Namespace | None = None) -> List[Dict[str, Any]]:
    eval_types = list(getattr(args, "eval_types", EVAL_TYPES) or EVAL_TYPES)
    cases: List[Dict[str, Any]] = []
    for challenge_type in eval_types:
        type_dir = root / challenge_type
        if not type_dir.exists():
            print(f"[eval-test-cases] skip missing test type dir: {type_dir}", flush=True)
            continue
        for file_path in sorted(type_dir.rglob("*.json")):
            if file_path.name in SKIP_CASE_FILENAMES:
                continue
            cases.extend(_load_case_file(file_path, challenge_type))
    if not cases:
        raise FileNotFoundError(f"No test case JSON files found under {root} for eval_types={eval_types}")
    return cases


def _find_failed_isolation_valid_files(path: str) -> List[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    valid_roots = [root] if root.name == BELIEF_FAILURE else sorted(p for p in root.rglob(BELIEF_FAILURE) if p.is_dir())
    files: List[Path] = []
    for valid_root in valid_roots:
        files.extend(sorted(valid_root.glob("*.json")))
    if not files:
        raise FileNotFoundError(f"No {BELIEF_FAILURE}/*.json found under {path}")
    return files


def _role_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
        if message.get("role") in {"system", "user", "assistant"}
    ]


def _assistant_records(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [message for message in messages if message.get("role") == "assistant"]


def _infer_failed_isolation_scenario(path: Path, payload: Dict[str, Any], task_arg: str) -> str:
    if task_arg != "auto":
        return task_arg
    if "repeat_trajectories" in payload:
        return "a"
    if "conversation" in payload:
        return "b"
    text = str(path)
    if "task_a" in text:
        return "a"
    if "task_b" in text:
        return "b"
    raise ValueError(f"Cannot infer failed_isolation scenario for {path}")


def _infer_prefix_from_matches(matches: List[bool]) -> int:
    for idx, match in enumerate(matches):
        if not match:
            return idx
    return max(0, len(matches) - 1)


def _infer_failed_isolation_score_mode(path: Path, payload: Dict[str, Any], args: argparse.Namespace) -> str:
    if args.failed_isolation_score_mode != "auto":
        return args.failed_isolation_score_mode
    text = str(path).lower()
    if "qwen2.5" in text or "7b" in text:
        return "strict-prefix"
    if "qwen3.5" in text:
        return "final-oracle"
    for repeat in payload.get("repeat_trajectories") or []:
        trajectory = repeat.get("trajectory") or {}
        if int(trajectory.get("strict_failed_isolation_prefix_turns") or 0) > 0:
            return "strict-prefix"
    return "final-oracle"


def _infer_failed_isolation_prefix(
    *,
    path: Path,
    payload: Dict[str, Any],
    scenario: str,
    source_matches: List[bool],
    score_mode: str,
    args: argparse.Namespace,
) -> int:
    if args.failed_isolation_strict_prefix_turns >= 0:
        return args.failed_isolation_strict_prefix_turns
    if score_mode != "strict-prefix":
        return 0
    if scenario == "a":
        for repeat in payload.get("repeat_trajectories") or []:
            trajectory = repeat.get("trajectory") or {}
            value = trajectory.get("strict_failed_isolation_prefix_turns")
            if isinstance(value, int) and value > 0:
                return value
    text = str(path).lower()
    if "qwen2.5" in text or "7b" in text:
        return 2
    return _infer_prefix_from_matches(source_matches)


def _oracle_from_failed_isolation_b_filename(name: str) -> str | None:
    match = re.search(r"_([A-Z])_tpl", name)
    return match.group(1) if match else None


def _failed_isolation_case_from_a(path: Path, payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    repeats = payload.get("repeat_trajectories") or []
    if not repeats:
        raise ValueError(f"{path} has no repeat_trajectories")
    source = repeats[0].get("trajectory") or {}
    messages = _role_messages(source.get("messages") or [])
    assistants = _assistant_records(source.get("messages") or [])
    gold = [list(message.get("golden_hypotheses") or []) for message in assistants]
    source_matches = [
        set(message.get("model_hypotheses") or []) == set(message.get("golden_hypotheses") or [])
        for message in assistants
    ]
    if not messages or not gold:
        raise ValueError(f"{path} has no replayable messages/gold")
    system = next((message["content"] for message in messages if message["role"] == "system"), None)
    if system is None:
        raise ValueError(f"{path} has no system message")
    score_mode = _infer_failed_isolation_score_mode(path, payload, args)
    prefix_turns = _infer_failed_isolation_prefix(
        path=path,
        payload=payload,
        scenario="a",
        source_matches=source_matches,
        score_mode=score_mode,
        args=args,
    )
    return {
        "case_id": path.stem,
        "experiment_id": str(payload.get("experiment_id") or path.stem),
        "cbm_challenge_type": FAILED_ISOLATION,
        "failed_isolation_scenario": "a",
        "source_file": str(path),
        "oracle": payload.get("oracle") or source.get("oracle"),
        "system_prompt": system,
        "failed_isolation_prefix_turns": int(prefix_turns),
        "failed_isolation_score_mode": score_mode,
        "turns": [
            {"prompt": message["content"], "golden": list(golden)}
            for message, golden in zip([m for m in messages if m["role"] == "user"], gold, strict=False)
        ],
    }


def _failed_isolation_case_from_b(path: Path, payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    repeats = payload.get("conversation") or []
    if not repeats:
        raise ValueError(f"{path} has no conversation repeats")
    source = repeats[0]
    messages = _role_messages(source.get("conversation") or [])
    survivors = source.get("turn_survivors") or []
    gold = [list(turn.get("golden_survivors") or []) for turn in survivors]
    sampled = [list(turn.get("sampled_survivors") or []) for turn in survivors]
    source_matches = [set(pred) == set(gt) for pred, gt in zip(sampled, gold, strict=False)]
    if not messages or not gold:
        raise ValueError(f"{path} has no replayable messages/gold")
    system = next((message["content"] for message in messages if message["role"] == "system"), None)
    if system is None:
        raise ValueError(f"{path} has no system message")
    score_mode = _infer_failed_isolation_score_mode(path, payload, args)
    prefix_turns = _infer_failed_isolation_prefix(
        path=path,
        payload=payload,
        scenario="b",
        source_matches=source_matches,
        score_mode=score_mode,
        args=args,
    )
    return {
        "case_id": path.stem,
        "cbm_challenge_type": FAILED_ISOLATION,
        "failed_isolation_scenario": "b",
        "source_file": str(path),
        "oracle": _oracle_from_failed_isolation_b_filename(path.name),
        "system_prompt": system,
        "failed_isolation_prefix_turns": int(prefix_turns),
        "failed_isolation_score_mode": score_mode,
        "turns": [
            {"prompt": message["content"], "golden": list(golden)}
            for message, golden in zip([m for m in messages if m["role"] == "user"], gold, strict=False)
        ],
    }


def _load_failed_isolation_cases(path: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for file_path in _find_failed_isolation_valid_files(path):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{file_path} is not a JSON object")
        scenario = _infer_failed_isolation_scenario(file_path, payload, args.failed_isolation_scenario)
        if scenario == "a":
            cases.append(_failed_isolation_case_from_a(file_path, payload, args))
        else:
            cases.append(_failed_isolation_case_from_b(file_path, payload, args))
    return cases


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_hypotheses(text: str) -> List[str] | None:
    return parse_hypotheses_tag(text)


def _normalize_challenge_type(raw: str) -> str:
    return normalize_cbm_challenge_type(raw)


def _model_output_label(model_label: str, args: argparse.Namespace) -> str:
    if args.enable_prompt_enhancement:
        return f"{model_label}_prompt_enhanced"
    return model_label


def _enhance_system_prompt(system_prompt: str) -> str:
    if PROMPT_ENHANCEMENT_TEXT in system_prompt:
        return system_prompt
    if PROMPT_ENHANCEMENT_MARKER not in system_prompt:
        return f"{system_prompt.rstrip()}\n\n{PROMPT_ENHANCEMENT_TEXT}"
    return system_prompt.replace(
        PROMPT_ENHANCEMENT_MARKER,
        f"{PROMPT_ENHANCEMENT_TEXT}\n\n{PROMPT_ENHANCEMENT_MARKER}",
        1,
    )


def _build_sampling_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.sampling_top_p is not None:
        overrides["top_p"] = args.sampling_top_p
    if args.sampling_top_k is not None:
        overrides["top_k"] = args.sampling_top_k
    if args.sampling_presence_penalty is not None:
        overrides["presence_penalty"] = args.sampling_presence_penalty
    if args.sampling_repetition_penalty is not None:
        overrides["repetition_penalty"] = args.sampling_repetition_penalty
    return overrides


def _resolve_enable_thinking(args: argparse.Namespace) -> bool | None:
    value = getattr(args, "enable_thinking", "auto")
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _build_backend_for_base(args: argparse.Namespace) -> VLLMBackend:
    if not args.base_model_path:
        raise ValueError("--base-model-path is required")
    print(f"[eval-test-cases] loading base model: {args.base_model_path}", flush=True)
    backend = VLLMBackend(
        model_path=args.base_model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.model_dtype,
        trust_remote_code=True,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enable_thinking=_resolve_enable_thinking(args),
    )
    backend.sampling_overrides = _build_sampling_overrides(args)
    return backend


def _build_backend_for_lora(args: argparse.Namespace) -> VLLMBackend:
    if args.lora_source_type == "merged":
        if not args.lora_model_path:
            raise ValueError("--lora-model-path is required when --lora-source-type merged")
        print(f"[eval-test-cases] loading merged lora model: {args.lora_model_path}", flush=True)
        backend = VLLMBackend(
            model_path=args.lora_model_path,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.model_dtype,
            trust_remote_code=True,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            enable_thinking=_resolve_enable_thinking(args),
        )
        backend.sampling_overrides = _build_sampling_overrides(args)
        return backend

    if not args.base_model_path:
        raise ValueError("--base-model-path is required when --lora-source-type adapter")
    if not args.lora_adapter_path:
        raise ValueError("--lora-adapter-path is required when --lora-source-type adapter")
    print(
        f"[eval-test-cases] loading base model with lora adapter: base={args.base_model_path} "
        f"adapter={args.lora_adapter_path}",
        flush=True,
    )
    backend = VLLMBackend(
        model_path=args.base_model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.model_dtype,
        trust_remote_code=True,
        adapter_path=args.lora_adapter_path,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enable_thinking=_resolve_enable_thinking(args),
    )
    backend.sampling_overrides = _build_sampling_overrides(args)
    return backend


def _init_repeat_session(
    case: Dict[str, Any],
    repeat_index: int,
    *,
    enable_prompt_enhancement: bool = False,
) -> Dict[str, Any]:
    turns = []
    for idx, turn in enumerate(case["turns"]):
        turns.append(
            {
                "turn": idx,
                "prompt": str(turn["prompt"]),
                "golden_hypotheses": list(turn["golden"]),
                "response": None,
                "model_hypotheses": None,
                "model_matches_golden": False,
                "parse_ok": False,
            }
        )
    system_prompt = str(case["system_prompt"])
    if enable_prompt_enhancement:
        system_prompt = _enhance_system_prompt(system_prompt)
    return {
        "case_id": str(case["case_id"]),
        "cbm_challenge_type": _normalize_challenge_type(case.get("cbm_challenge_type", "")),
        "oracle": str(case.get("oracle", "")),
        "repeat_index": int(repeat_index),
        "system_prompt": system_prompt,
        "conversation": [],
        "turns": turns,
        "max_turns": len(turns),
    }


def _init_sessions(
    cases: Sequence[Dict[str, Any]],
    repeats: int,
    *,
    enable_prompt_enhancement: bool = False,
) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    for case in cases:
        for repeat_index in range(repeats):
            sessions.append(
                _init_repeat_session(
                    case,
                    repeat_index,
                    enable_prompt_enhancement=enable_prompt_enhancement,
                )
            )
    return sessions


def _prepare_turn_prompt(session: Dict[str, Any], turn_idx: int) -> List[Dict[str, str]] | None:
    if turn_idx >= session["max_turns"]:
        return None
    conversation = session["conversation"]
    prompt = session["turns"][turn_idx]["prompt"]
    if turn_idx == 0 and not conversation:
        conversation.extend(
            [
                {"role": "system", "content": session["system_prompt"]},
                {"role": "user", "content": prompt},
            ]
        )
    else:
        conversation.append({"role": "user", "content": prompt})
    return conversation


def _apply_turn_response(session: Dict[str, Any], turn_idx: int, response: str) -> None:
    conversation = session["conversation"]
    conversation.append({"role": "assistant", "content": response})

    turn_record = session["turns"][turn_idx]
    parsed = _parse_hypotheses(response)
    golden = set(turn_record["golden_hypotheses"])
    model_hyp = [] if parsed is None else list(dict.fromkeys(parsed))

    turn_record["response"] = response
    turn_record["model_hypotheses"] = model_hyp
    turn_record["parse_ok"] = parsed is not None
    turn_record["model_matches_golden"] = set(model_hyp) == golden


def _response_fingerprint(response: str) -> str:
    return hashlib.sha256(response.encode("utf-8", errors="replace")).hexdigest()[:12]


def _run_sessions(
    *,
    backend: VLLMBackend,
    sessions: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    run_label: str,
) -> None:
    max_turns = max((session["max_turns"] for session in sessions), default=0)
    for turn_idx in range(max_turns):
        active_sessions: List[Dict[str, Any]] = []
        messages_batch: List[List[Dict[str, str]]] = []

        for session in sessions:
            messages = _prepare_turn_prompt(session, turn_idx)
            if messages is None:
                continue
            active_sessions.append(session)
            messages_batch.append(messages)

        if not messages_batch:
            continue

        print(
            f"[vLLM] {run_label}: turn {turn_idx + 1}/{max_turns}, prompts={len(messages_batch)}",
            flush=True,
        )
        responses = backend.batch_chat_completion(
            messages_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            use_tqdm=True,
        )
        if len(responses) != len(active_sessions):
            raise RuntimeError(
                f"response count mismatch at turn {turn_idx}: responses={len(responses)} active_sessions={len(active_sessions)}"
            )

        for session, response in zip(active_sessions, responses):
            _apply_turn_response(session, turn_idx, response)

        if responses:
            print(
                f"[vLLM] {run_label}: turn {turn_idx + 1} first_response_sha256={_response_fingerprint(responses[0])}",
                flush=True,
            )


def _classify_repeat_trajectory(case: Dict[str, Any], repeat_state: Dict[str, Any]) -> str:
    challenge_type = _normalize_challenge_type(str(case.get("cbm_challenge_type", "")))
    if challenge_type == FAILED_ISOLATION:
        return _classify_failed_isolation_repeat_trajectory(case, repeat_state)

    turns = repeat_state["turns"]
    matches = [bool(turn["model_matches_golden"]) for turn in turns]

    if challenge_type == FAILED_UPDATE:
        if len(matches) < 4:
            return CATEGORY_INSUFFICIENT
        if not matches[2]:
            return CATEGORY_INSUFFICIENT
        return CATEGORY_BELIEF if matches[3] else CATEGORY_VALID

    if len(matches) < 4:
        return CATEGORY_INSUFFICIENT
    if not matches[0]:
        return CATEGORY_INSUFFICIENT
    if not matches[1]:
        turn1_gold = set(turns[0]["golden_hypotheses"])
        turn2_model = set(turns[1]["model_hypotheses"] or [])
        reintroduced = any(rule_id not in turn1_gold for rule_id in turn2_model)
        return CATEGORY_VALID if reintroduced else CATEGORY_INSUFFICIENT
    return CATEGORY_BELIEF if matches[2] and matches[3] else CATEGORY_VALID


def _classify_failed_isolation_repeat_trajectory(case: Dict[str, Any], repeat_state: Dict[str, Any]) -> str:
    turns = repeat_state["turns"]
    if not turns:
        return CATEGORY_INSUFFICIENT
    return CATEGORY_BELIEF if bool(turns[-1].get("model_matches_golden")) else CATEGORY_VALID


def _build_model_summary(
    model_label: str,
    cases: Sequence[Dict[str, Any]],
    sessions: Sequence[Dict[str, Any]],
    output_dir: Path,
    repeats: int,
    model_info: Dict[str, Any],
    failed_isolation_aggregate: str = "majority",
) -> Dict[str, Any]:
    model_dir = output_dir / model_label
    model_dir.mkdir(parents=True, exist_ok=True)

    case_lookup = {str(case["case_id"]): case for case in cases}
    sessions_by_case: Dict[str, List[Dict[str, Any]]] = {}
    for session in sessions:
        sessions_by_case.setdefault(session["case_id"], []).append(session)

    current_challenge_types = {
        _normalize_challenge_type(case_lookup[case_id].get("cbm_challenge_type", ""))
        for case_id in sessions_by_case
    }
    for challenge_type in current_challenge_types:
        challenge_dir = model_dir / challenge_type
        if challenge_dir.exists():
            shutil.rmtree(challenge_dir)

    per_type: Dict[str, Dict[str, Any]] = {}

    for case_id, case_sessions in sorted(sessions_by_case.items()):
        case = case_lookup[case_id]
        challenge_type = _normalize_challenge_type(case.get("cbm_challenge_type", ""))
        if challenge_type not in per_type:
            challenge_dir = model_dir / challenge_type
            challenge_dir.mkdir(parents=True, exist_ok=True)
            category_dirs = {name: Path(path) for name, path in _build_eval_category_dirs(str(challenge_dir)).items()}
            for category_dir in category_dirs.values():
                category_dir.mkdir(parents=True, exist_ok=True)
            per_type[challenge_type] = {
                "category_dirs": category_dirs,
                "cbm_category_counts": {category: 0 for category in EVAL_CATEGORIES},
                "sample_results": [],
                "num_cases": 0,
            }

        ordered_sessions = sorted(case_sessions, key=lambda item: int(item["repeat_index"]))
        per_run_categories = [_classify_repeat_trajectory(case, session) for session in ordered_sessions]
        final_category = _aggregate_repeat_categories_eval(per_run_categories)

        per_type[challenge_type]["cbm_category_counts"][final_category] += 1
        per_type[challenge_type]["num_cases"] += 1

        payload = {
            "case_id": case_id,
            "model_info": model_info,
            "cbm_challenge_type": normalize_cbm_challenge_type(challenge_type),
            "failure_metric": cbm_failure_metric(challenge_type),
            "oracle": case.get("oracle"),
            "source_file": case.get("source_file"),
            "failed_isolation_scenario": case.get("failed_isolation_scenario"),
            "failed_isolation_prefix_turns": case.get("failed_isolation_prefix_turns"),
            "failed_isolation_score_mode": case.get("failed_isolation_score_mode"),
            "system_prompt": ordered_sessions[0].get("system_prompt") if ordered_sessions else case.get("system_prompt"),
            "prompt_enhancement_enabled": bool(model_info.get("prompt_enhancement_enabled")),
            "turns": case.get("turns"),
            "category": final_category,
            "repeats": repeats,
            "repeat_trajectories": [
                {
                    "repeat_index": session["repeat_index"],
                    "category": per_run_categories[idx],
                    "trajectory": session,
                }
                for idx, session in enumerate(ordered_sessions)
            ],
        }
        _write_json(per_type[challenge_type]["category_dirs"][final_category] / f"{case_id}.json", payload)

        sample_item = {
            "case_id": case_id,
            "category": final_category,
            "cbm_challenge_type": normalize_cbm_challenge_type(challenge_type),
            "failure_metric": cbm_failure_metric(challenge_type),
            "oracle": case.get("oracle"),
            "source_file": case.get("source_file"),
            "failed_isolation_scenario": case.get("failed_isolation_scenario"),
            "failed_isolation_prefix_turns": case.get("failed_isolation_prefix_turns"),
            "failed_isolation_score_mode": case.get("failed_isolation_score_mode"),
            "repeat_categories": per_run_categories,
            "repeat_response_hashes": [
                [
                    _response_fingerprint(str(turn.get("response") or ""))
                    for turn in session["turns"]
                ]
                for session in ordered_sessions
            ],
        }
        per_type[challenge_type]["sample_results"].append(sample_item)

    for challenge_type, challenge_payload in per_type.items():
        num_cases = challenge_payload["num_cases"]
        cbm_counts = canonicalize_category_counts(challenge_payload["cbm_category_counts"])
        challenge_stats = {
            "model_label": model_label,
            "model_info": model_info,
            "cbm_challenge_type": normalize_cbm_challenge_type(challenge_type),
            "failure_metric": cbm_failure_metric(challenge_type),
            "num_cases": num_cases,
            "repeats": repeats,
            "cbm_category_counts": cbm_counts,
            "cbm_category_percentages": percentages_from_counts(cbm_counts),
            "sample_results": challenge_payload["sample_results"],
        }
        _write_json(model_dir / challenge_type / "stats_report.json", challenge_stats)

    all_type_stats: Dict[str, Any] = {}
    for challenge_type in EVAL_TYPES:
        type_stats = _read_json_if_exists(model_dir / challenge_type / "stats_report.json")
        if type_stats is not None:
            all_type_stats[challenge_type] = type_stats

    aggregate_num_cases = 0
    aggregate_category_counts = {category: 0 for category in EVAL_CATEGORIES}
    aggregate_sample_results: List[Dict[str, Any]] = []
    for type_stats in all_type_stats.values():
        aggregate_num_cases += int(type_stats.get("num_cases", 0))
        type_counts = type_stats.get("cbm_category_counts", {})
        for category in EVAL_CATEGORIES:
            aggregate_category_counts[category] += int(type_counts.get(category, 0))
        aggregate_sample_results.extend(type_stats.get("sample_results", []))

    cbm_aggregate_counts = canonicalize_category_counts(aggregate_category_counts)
    stats = {
        "model_label": model_label,
        "model_info": model_info,
        "num_cases": aggregate_num_cases,
        "repeats": repeats,
        "cbm_category_counts": cbm_aggregate_counts,
        "cbm_category_percentages": percentages_from_counts(cbm_aggregate_counts),
        "sample_results": aggregate_sample_results,
        "cbm_challenge_type_stats": all_type_stats,
    }
    _write_json(model_dir / "stats_report.json", stats)
    return stats


def _read_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _category_counts(stats: Dict[str, Any]) -> Dict[str, int]:
    raw_counts = stats.get("cbm_category_counts", {})
    return {category: int(raw_counts.get(category, 0)) for category in EVAL_CATEGORIES}


def _category_percentages(stats: Dict[str, Any]) -> Dict[str, float]:
    raw_percentages = stats.get("cbm_category_percentages", {})
    return {category: float(raw_percentages.get(category, 0.0)) for category in EVAL_CATEGORIES}


def _build_type_comparison(challenge_type: str, base_stats: Dict[str, Any], lora_stats: Dict[str, Any]) -> Dict[str, Any]:
    base_counts = _category_counts(base_stats)
    lora_counts = _category_counts(lora_stats)
    base_percentages = _category_percentages(base_stats)
    lora_percentages = _category_percentages(lora_stats)
    base_samples = {str(item["case_id"]): item for item in base_stats.get("sample_results", [])}
    lora_samples = {str(item["case_id"]): item for item in lora_stats.get("sample_results", [])}
    shared_case_ids = sorted(set(base_samples) & set(lora_samples))
    identical_response_hash_cases = [
        case_id
        for case_id in shared_case_ids
        if base_samples[case_id].get("repeat_response_hashes") == lora_samples[case_id].get("repeat_response_hashes")
    ]
    return {
        "cbm_challenge_type": challenge_type,
        "num_cases": base_stats.get("num_cases"),
        "repeats": base_stats.get("repeats"),
        "identical_response_hash_case_count": len(identical_response_hash_cases),
        "identical_response_hash_case_percentage": round(
            len(identical_response_hash_cases) / max(len(shared_case_ids), 1) * 100,
            2,
        ),
        "identical_response_hash_case_ids": identical_response_hash_cases,
        "base": {
            "model_info": base_stats.get("model_info"),
            "cbm_category_counts": base_counts,
            "cbm_category_percentages": base_percentages,
        },
        "lora": {
            "model_info": lora_stats.get("model_info"),
            "cbm_category_counts": lora_counts,
            "cbm_category_percentages": lora_percentages,
        },
        "delta_percentage_points": {
            category: round(lora_percentages[category] - base_percentages[category], 2)
            for category in EVAL_CATEGORIES
        },
    }


def _print_comparison(label: str, comparison: Dict[str, Any]) -> None:
    print(
        f"[eval-test-cases] {label}: "
        + ", ".join(
                f"{category}: base={comparison['base']['cbm_category_percentages'][category]}% "
                f"lora={comparison['lora']['cbm_category_percentages'][category]}% "
                f"delta={comparison['delta_percentage_points'][category]}pt"
            for category in EVAL_CATEGORIES
        )
    )
    print(
        f"[eval-test-cases] {label}: identical_response_hash_cases="
        f"{comparison.get('identical_response_hash_case_count', 0)}/"
        f"{comparison.get('num_cases', 0)} "
        f"({comparison.get('identical_response_hash_case_percentage', 0)}%)"
    )


def _comparison_output_name(base_label: str, lora_label: str) -> str:
    if base_label == "base" and lora_label == "lora":
        return "comparison.json"
    if base_label == "base_prompt_enhanced" and lora_label == "lora_prompt_enhanced":
        return "comparison_prompt_enhanced.json"
    return f"comparison_{base_label}_vs_{lora_label}.json"


def _comparison_model_labels(args: argparse.Namespace) -> tuple[str, str]:
    return _model_output_label("base", args), _model_output_label("lora", args)


def _build_comparison(output_dir: Path, base_label: str = "base", lora_label: str = "lora") -> Dict[str, Any] | None:
    comparison = {"base_label": base_label, "lora_label": lora_label, "cbm_challenge_types": {}}
    for challenge_type in EVAL_TYPES:
        base_type_stats = _read_json_if_exists(output_dir / base_label / challenge_type / "stats_report.json")
        lora_type_stats = _read_json_if_exists(output_dir / lora_label / challenge_type / "stats_report.json")
        if base_type_stats is None or lora_type_stats is None:
            continue
        comparison["cbm_challenge_types"][challenge_type] = _build_type_comparison(
            challenge_type,
            base_type_stats,
            lora_type_stats,
        )
    if not comparison["cbm_challenge_types"]:
        return None
    _write_json(output_dir / _comparison_output_name(base_label, lora_label), comparison)
    return comparison


def _cleanup_backend(backend: VLLMBackend | None) -> None:
    if backend is None:
        return
    try:
        llm = getattr(backend, "llm", None)
        if llm is not None:
            del llm
    finally:
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _model_info(model_label: str, args: argparse.Namespace) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model_label": model_label,
        "eval_target": args.eval_target,
        "lora_source_type": args.lora_source_type,
        "base_model_path": args.base_model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "model_dtype": args.model_dtype,
        "prompt_enhancement_enabled": args.enable_prompt_enhancement,
        "eval_mode": args.eval_mode,
        "enable_thinking": args.enable_thinking,
        "failed_isolation_scenario": args.failed_isolation_scenario,
        "failed_isolation_score_mode": args.failed_isolation_score_mode,
        "failed_isolation_strict_prefix_turns": args.failed_isolation_strict_prefix_turns,
        "failed_isolation_aggregate": args.failed_isolation_aggregate,
        "max_cases": getattr(args, "max_cases", None),
    }
    if model_label.startswith("lora"):
        info["lora_model_path"] = args.lora_model_path
        info["lora_adapter_path"] = args.lora_adapter_path
    return info


def _evaluate_single_model(model_label: str, args: argparse.Namespace) -> None:
    cases = _load_cases(args.test_data, args)
    output_label = _model_output_label(model_label, args)

    if model_label == "base":
        backend = _build_backend_for_base(args)
    else:
        backend = _build_backend_for_lora(args)

    try:
        sessions = _init_sessions(
            cases,
            args.repeats,
            enable_prompt_enhancement=args.enable_prompt_enhancement,
        )
        _run_sessions(
            backend=backend,
            sessions=sessions,
            temperature=args.agent_temperature,
            max_tokens=args.agent_max_tokens,
            run_label=output_label,
        )
        stats = _build_model_summary(
            output_label,
            cases,
            sessions,
            Path(args.output_dir),
            args.repeats,
            _model_info(output_label, args),
            args.failed_isolation_aggregate,
        )
        print(
            f"[eval-test-cases] {output_label} done: "
            + ", ".join(
                f"{category}={stats['cbm_category_counts'][category]} ({stats['cbm_category_percentages'][category]}%)"
                for category in EVAL_CATEGORIES
            )
        )
    finally:
        _cleanup_backend(backend)

    if not args.skip_compare:
        comparison = _build_comparison(Path(args.output_dir), *_comparison_model_labels(args))
        if comparison is not None:
            for challenge_type in EVAL_TYPES:
                if challenge_type in comparison["cbm_challenge_types"]:
                    _print_comparison(f"{challenge_type} comparison", comparison["cbm_challenge_types"][challenge_type])


def _spawn_child(target: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--test-data",
        args.test_data,
        "--output-dir",
        args.output_dir,
        "--eval-target",
        target,
        "--base-model-path",
        str(args.base_model_path),
        "--lora-source-type",
        args.lora_source_type,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--model-dtype",
        str(args.model_dtype),
        "--agent-temperature",
        str(args.agent_temperature),
        "--agent-max-tokens",
        str(args.agent_max_tokens),
        "--repeats",
        str(args.repeats),
        "--skip-compare",
        "--eval-mode",
        args.eval_mode,
        "--enable-thinking",
        args.enable_thinking,
        "--failed_isolation-scenario",
        args.failed_isolation_scenario,
        "--failed_isolation-strict-prefix-turns",
        str(args.failed_isolation_strict_prefix_turns),
        "--failed_isolation-score-mode",
        args.failed_isolation_score_mode,
        "--failed_isolation-aggregate",
        args.failed_isolation_aggregate,
    ]
    cmd.extend(["--eval-types", *args.eval_types])
    if args.disable_custom_all_reduce:
        cmd.append("--disable-custom-all-reduce")
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.sampling_top_p is not None:
        cmd.extend(["--sampling-top-p", str(args.sampling_top_p)])
    if args.sampling_top_k is not None:
        cmd.extend(["--sampling-top-k", str(args.sampling_top_k)])
    if args.sampling_presence_penalty is not None:
        cmd.extend(["--sampling-presence-penalty", str(args.sampling_presence_penalty)])
    if args.sampling_repetition_penalty is not None:
        cmd.extend(["--sampling-repetition-penalty", str(args.sampling_repetition_penalty)])
    if args.enable_prompt_enhancement:
        cmd.append("--enable-prompt-enhancement")
    if args.lora_model_path is not None:
        cmd.extend(["--lora-model-path", args.lora_model_path])
    if args.lora_adapter_path is not None:
        cmd.extend(["--lora-adapter-path", args.lora_adapter_path])
    if getattr(args, "max_cases", None) is not None and args.max_cases > 0:
        cmd.extend(["--max-cases", str(args.max_cases)])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_target == "both":
        _spawn_child("base", args)
        _spawn_child("lora", args)
        if not args.skip_compare:
            comparison = _build_comparison(output_dir, *_comparison_model_labels(args))
            if comparison is not None:
                for challenge_type in EVAL_TYPES:
                    if challenge_type in comparison["cbm_challenge_types"]:
                        _print_comparison(f"{challenge_type} comparison", comparison["cbm_challenge_types"][challenge_type])
        return

    _evaluate_single_model(args.eval_target, args)

if __name__ == "__main__":
    main()
