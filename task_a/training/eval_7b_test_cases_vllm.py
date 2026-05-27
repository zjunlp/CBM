from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_a.training import eval_9b_test_cases_vllm as base


def _classify_repeat_trajectory(case: Dict[str, Any], repeat_state: Dict[str, Any]) -> str:
    challenge_type = base._normalize_challenge_type(str(case.get("cbm_challenge_type", "")))
    if challenge_type == "failed_isolation":
        return base._classify_failed_isolation_repeat_trajectory(case, repeat_state)

    turns = repeat_state["turns"]
    matches = [bool(turn["model_matches_golden"]) for turn in turns]

    if challenge_type == "failed_update":
        if len(matches) < 3:
            return base.CATEGORY_INSUFFICIENT
        if not matches[1]:
            return base.CATEGORY_INSUFFICIENT
        return base.CATEGORY_VALID if not matches[2] else base.CATEGORY_BELIEF

    if len(matches) < 3:
        return base.CATEGORY_INSUFFICIENT
    if not matches[0]:
        return base.CATEGORY_INSUFFICIENT
    if not matches[1]:
        turn0_gold = set(turns[0]["golden_hypotheses"])
        turn1_model = set(turns[1]["model_hypotheses"] or [])
        reintroduced = any(rule_id not in turn0_gold for rule_id in turn1_model)
        return base.CATEGORY_VALID if reintroduced else base.CATEGORY_INSUFFICIENT
    return base.CATEGORY_BELIEF if matches[2] else base.CATEGORY_VALID


def _spawn_child(target: str, args: Any) -> None:
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


def main() -> None:
    base._classify_repeat_trajectory = _classify_repeat_trajectory
    args = base._parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_target == "both":
        _spawn_child("base", args)
        _spawn_child("lora", args)
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


if __name__ == "__main__":
    main()
