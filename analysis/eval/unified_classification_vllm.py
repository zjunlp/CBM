"""analysis-specific vLLM eval entry helpers with unified classification rules."""

from __future__ import annotations

import subprocess
import sys
import os
import inspect
import re
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict, List

from utils.llm_backend import APIBackend

_OUTPUT_FORMAT_BLOCK_RE = re.compile(r"Output format \(strict\):\n.*\Z", re.DOTALL)
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


def _classify_failed_isolation_repeat_trajectory(base: Any, case: Dict[str, Any], repeat_state: Dict[str, Any]) -> str:
    turns = repeat_state["turns"]
    if not turns:
        return base.CATEGORY_VALID
    return base.CATEGORY_BELIEF if bool(turns[-1].get("model_matches_golden")) else base.CATEGORY_VALID


def _classify_repeat_trajectory(base: Any, case: Dict[str, Any], repeat_state: Dict[str, Any]) -> str:
    challenge_type = base._normalize_challenge_type(str(case.get("cbm_challenge_type", "")))
    if challenge_type == "failed_isolation":
        return _classify_failed_isolation_repeat_trajectory(base, case, repeat_state)

    turns = repeat_state["turns"]
    matches = [bool(turn["model_matches_golden"]) for turn in turns]

    if challenge_type == "failed_update":
        if len(matches) < 2:
            return base.CATEGORY_INSUFFICIENT
        if not matches[-2]:
            return base.CATEGORY_INSUFFICIENT
        return base.CATEGORY_BELIEF if matches[-1] else base.CATEGORY_VALID

    if len(matches) < 2:
        return base.CATEGORY_INSUFFICIENT
    if not matches[0]:
        return base.CATEGORY_INSUFFICIENT
    if not matches[1]:
        first_turn_gold = set(turns[0]["golden_hypotheses"])
        second_turn_model = set(turns[1]["model_hypotheses"] or [])
        reintroduced = any(rule_id not in first_turn_gold for rule_id in second_turn_model)
        return base.CATEGORY_VALID if reintroduced else base.CATEGORY_INSUFFICIENT
    return base.CATEGORY_VALID if any(not match for match in matches[2:]) else base.CATEGORY_BELIEF


def apply_unified_classification(base: Any) -> None:
    base.PROMPT_ENHANCEMENT_MARKER = PROMPT_ENHANCEMENT_MARKER
    base.PROMPT_ENHANCEMENT_TEXT = PROMPT_ENHANCEMENT_TEXT
    base._classify_repeat_trajectory = lambda case, repeat_state: _classify_repeat_trajectory(
        base,
        case,
        repeat_state,
    )
    base._classify_failed_isolation_repeat_trajectory = lambda case, repeat_state: _classify_failed_isolation_repeat_trajectory(
        base,
        case,
        repeat_state,
    )
    apply_analysis_source_case_limit(base)


def _analysis_source_case_id(case: Dict[str, Any]) -> str:
    augmentation = case.get("augmentation") or {}
    source_case_id = augmentation.get("source_case_id")
    if source_case_id:
        return str(source_case_id)
    return str(case.get("case_id") or case.get("id") or "")


def _limit_by_source_case(cases: List[Dict[str, Any]], limit: int | None) -> List[Dict[str, Any]]:
    if limit is None or limit <= 0:
        return cases
    selected_sources = set()
    selected: List[Dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: (_analysis_source_case_id(item), str(item.get("case_id") or item.get("id") or ""))):
        source_case_id = _analysis_source_case_id(case)
        if source_case_id not in selected_sources:
            if len(selected_sources) >= limit:
                continue
            selected_sources.add(source_case_id)
        if source_case_id in selected_sources:
            selected.append(case)
    print(
        f"[analysis-eval] max-cases={limit}: using {len(selected)}/{len(cases)} "
        f"augmented cases from {len(selected_sources)} source cases"
    )
    return selected


def apply_analysis_source_case_limit(base: Any) -> None:
    if getattr(base, "_analysis_source_case_limit_applied", False):
        return
    original_load_cases = base._load_cases

    def _load_cases_with_source_limit(path: str, args: Any | None = None) -> List[Dict[str, Any]]:
        if args is None or getattr(args, "max_cases", None) in (None, 0):
            return original_load_cases(path, args)
        unrestricted_args = copy(args)
        unrestricted_args.max_cases = 0
        cases = original_load_cases(path, unrestricted_args)
        return _limit_by_source_case(cases, int(args.max_cases))

    base._load_cases = _load_cases_with_source_limit
    base._analysis_source_case_limit_applied = True


def _spawn_child(wrapper_file: str, target: str, args: Any) -> None:
    cmd = [
        sys.executable,
        str(Path(wrapper_file).resolve()),
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
    ]
    if getattr(args, "max_cases", None) is not None and args.max_cases > 0:
        cmd.extend(["--max-cases", str(args.max_cases)])
    cmd.extend(
        [
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
    )
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
    cmd.extend(["--eval-types", *args.eval_types])
    if args.lora_model_path is not None:
        cmd.extend(["--lora-model-path", args.lora_model_path])
    if args.lora_adapter_path is not None:
        cmd.extend(["--lora-adapter-path", args.lora_adapter_path])
    subprocess.run(cmd, check=True)


def main(base: Any, wrapper_file: str) -> None:
    apply_unified_classification(base)
    args = base._parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_target == "both":
        _spawn_child(wrapper_file, "base", args)
        _spawn_child(wrapper_file, "lora", args)
        if not args.skip_compare:
            comparison = base._build_comparison(output_dir)
            if comparison is not None:
                for challenge_type in base.EVAL_TYPES:
                    if challenge_type in comparison["cbm_challenge_types"]:
                        base._print_comparison(
                            f"{challenge_type} comparison",
                            comparison["cbm_challenge_types"][challenge_type],
                        )
        return

    base._evaluate_single_model(args.eval_target, args)


def _rewrite_api_argv() -> None:
    updated = list(sys.argv)
    for idx, arg in enumerate(updated):
        if arg == "--eval-target" and idx + 1 < len(updated) and updated[idx + 1] == "api":
            updated[idx + 1] = "base"
        elif arg == "--eval-target=api":
            updated[idx] = "--eval-target=base"
    sys.argv = updated


def _api_model_info(output_label: str, args: Any, api_base_url: str, api_model_name: str) -> Dict[str, Any]:
    return {
        "model_label": output_label,
        "backend": "api",
        "api_base_url": api_base_url,
        "api_model_name": api_model_name,
        "prompt_enhancement_enabled": bool(args.enable_prompt_enhancement),
    }


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _optional_float_env(name: str) -> float | None:
    value = _optional_env(name)
    return None if value is None else float(value)


def _optional_int_env(name: str) -> int | None:
    value = _optional_env(name)
    return None if value is None else int(value)


def _optional_bool_env(name: str, default: bool | None = None) -> bool | None:
    value = _optional_env(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be true/false when set, got {value!r}")


def _api_sampling_overrides() -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    mapping = {
        "API_SAMPLING_TOP_P": ("top_p", float),
        "API_SAMPLING_TOP_K": ("top_k", int),
        "API_SAMPLING_PRESENCE_PENALTY": ("presence_penalty", float),
        "API_SAMPLING_REPETITION_PENALTY": ("repetition_penalty", float),
        "API_SAMPLING_FREQUENCY_PENALTY": ("frequency_penalty", float),
        "API_SAMPLING_MIN_P": ("min_p", float),
    }
    for env_name, (override_name, caster) in mapping.items():
        value = _optional_env(env_name)
        if value is not None:
            overrides[override_name] = caster(value)
    return overrides


def _api_output_format_text(system_prompt: str) -> str:
    if "fault_id_1" in system_prompt or "Fault Space:" in system_prompt:
        id_example = "fault_id_1, fault_id_2"
        id_name = "fault IDs"
    else:
        id_example = "rule_id_1, rule_id_2"
        id_name = "rule IDs"
    return (
        "Output format (strict):\n"
        f"<think>your reasoning</think><hypothesis>{id_example}</hypothesis>\n\n"
        f"- Inside `<hypothesis>`: comma-separated {id_name} that are still consistent with ALL active evidence, "
        "or `none` if no ID remains."
    )


def rewrite_api_system_prompt(system_prompt: str) -> str:
    replacement = _api_output_format_text(system_prompt)
    if "Output format (strict):" not in system_prompt:
        return f"{system_prompt.rstrip()}\n\n{replacement}"
    return _OUTPUT_FORMAT_BLOCK_RE.sub(replacement, system_prompt.rstrip())


def prepare_api_cases(cases: Any) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for case in cases:
        item = deepcopy(case)
        item["system_prompt"] = rewrite_api_system_prompt(str(item.get("system_prompt", "")))
        prepared.append(item)
    return prepared


def _build_api_summary(
    base: Any,
    output_label: str,
    cases: Any,
    sessions: Any,
    args: Any,
    api_base_url: str,
    api_model_name: str,
) -> Dict[str, Any]:
    signature = inspect.signature(base._build_model_summary)
    kwargs = {"failed_isolation_aggregate": args.failed_isolation_aggregate}
    if "model_info" in signature.parameters:
        return base._build_model_summary(
            output_label,
            cases,
            sessions,
            Path(args.output_dir),
            args.repeats,
            _api_model_info(output_label, args, api_base_url, api_model_name),
            **kwargs,
        )
    return base._build_model_summary(
        output_label,
        cases,
        sessions,
        Path(args.output_dir),
        args.repeats,
        prompt_enhancement_enabled=args.enable_prompt_enhancement,
        **kwargs,
    )


def main_api(base: Any) -> None:
    apply_unified_classification(base)
    _rewrite_api_argv()
    args = base._parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_base_url = os.environ.get("API_BASE_URL", "").strip()
    if not api_base_url:
        raise ValueError("API_BASE_URL is required when EVAL_TARGET=api")
    api_model_name = os.environ.get("API_MODEL_NAME", "Qwen3.5-9B").strip()
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_max_workers = int(os.environ.get("API_MAX_WORKERS", "8"))
    api_enable_thinking = _optional_bool_env("API_ENABLE_THINKING", True)
    args.agent_temperature = _optional_float_env("API_AGENT_TEMPERATURE")
    args.agent_max_tokens = _optional_int_env("API_AGENT_MAX_TOKENS")

    cases = prepare_api_cases(base._load_cases(args.test_data, args))
    output_label = base._model_output_label("api", args)
    backend = APIBackend(
        api_base_url=api_base_url,
        model_name=api_model_name,
        api_key=api_key,
        max_workers=api_max_workers,
        enable_thinking=api_enable_thinking,
    )
    backend.sampling_overrides = _api_sampling_overrides()

    sessions = base._init_sessions(
        cases,
        args.repeats,
        enable_prompt_enhancement=args.enable_prompt_enhancement,
    )
    base._run_sessions(
        backend=backend,
        sessions=sessions,
        temperature=args.agent_temperature,
        max_tokens=args.agent_max_tokens,
        run_label=output_label,
    )
    stats = _build_api_summary(
        base,
        output_label,
        cases,
        sessions,
        args,
        api_base_url,
        api_model_name,
    )
    print(
        f"[eval-test-cases] {output_label} done: "
        + ", ".join(
            f"{category}={stats['cbm_category_counts'][category]} ({stats['cbm_category_percentages'][category]}%)"
            for category in base.EVAL_CATEGORIES
        )
    )

    if not args.skip_compare:
        comparison = base._build_comparison(Path(args.output_dir))
        if comparison is not None:
            for challenge_type in base.EVAL_TYPES:
                if challenge_type in comparison["cbm_challenge_types"]:
                    base._print_comparison(f"{challenge_type} comparison", comparison["cbm_challenge_types"][challenge_type])
