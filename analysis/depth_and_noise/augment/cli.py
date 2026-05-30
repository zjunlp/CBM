"""Unified CLI for analysis augmentation pipelines."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.depth_and_noise.augment.cd_failed_stay_depth import augment_cd_failed_stay_depth
from analysis.depth_and_noise.augment.cd_failed_update_depth import augment_cd_failed_update_depth
from analysis.depth_and_noise.augment.cd_noise_typology import augment_noise_typology
from analysis.depth_and_noise.augment.failed_isolation_source import normalize_failed_isolation_source_case
from analysis.depth_and_noise.augment.rd_failed_stay_depth import augment_rd_failed_stay_depth
from analysis.depth_and_noise.augment.rd_failed_update_depth import augment_rd_failed_update_depth
from analysis.depth_and_noise.augment.schema import (
    iter_input_cases,
    load_case,
    manifest_record,
    stable_case_hash,
    write_augment_summary,
    write_case,
    write_manifest_line,
)

PIPELINES = {
    "rd_failed_stay_depth": "failed_stay",
    "cd_failed_stay_depth": "failed_stay",
    "rd_failed_update_depth": "failed_update",
    "cd_failed_update_depth": "failed_update",
    "noise_typology": "failed_isolation",
}

TASK_NAMES = {"task_a": "Task_A", "task_b": "Task_B"}


def _default_data_root() -> Path:
    env_root = os.environ.get("BELIEFTRACK_DATA_ROOT")
    if env_root:
        return Path(env_root)

    legacy_root = REPO_ROOT / "data" / "belief_training_task_dataset"
    if legacy_root.exists():
        return legacy_root

    hf_snapshot_root = REPO_ROOT / "data" / "BeliefTrackDataset"
    if hf_snapshot_root.exists():
        return hf_snapshot_root

    return legacy_root


def _normalize_task(value: Any) -> str:
    text = str(value or "task_a").strip().lower()
    if text in TASK_NAMES:
        return text
    raise ValueError(f"unsupported task: {value}")


def _normalize_case_fields(case: Dict[str, Any], task: str) -> Dict[str, Any]:
    out = dict(case)
    out["task"] = _normalize_task(out.get("task") or task)
    return out


def _failed_stay_pipeline_for_task(task: str) -> str:
    return "cd_failed_stay_depth" if task == "task_b" else "rd_failed_stay_depth"


def _failed_update_pipeline_for_task(task: str) -> str:
    return "cd_failed_update_depth" if task == "task_b" else "rd_failed_update_depth"


def _case_output_dir(output_dir: Path, pipeline: str) -> Path:
    """Place cases under challenge-type subdirs for vLLM eval compatibility."""
    challenge_type = PIPELINES.get(pipeline)
    if challenge_type:
        return output_dir / challenge_type
    return output_dir


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict(payload)


def _resolve_input_dir(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    if args.input_dir:
        return Path(args.input_dir)
    task = _normalize_task(args.task or config.get("task", "task_a"))
    model = args.model or config.get("model", "7B")
    split = args.split or config.get("split", "test")
    challenge = args.challenge_type or config.get("challenge_type")
    task_name = TASK_NAMES[task]
    base = _default_data_root() / task_name / model / split
    if challenge:
        return base / str(challenge)
    return base


def _output_pipeline_name(pipeline: str) -> str:
    if pipeline in {"rd_failed_stay_depth", "cd_failed_stay_depth"}:
        return "failed_stay_depth"
    if pipeline in {"rd_failed_update_depth", "cd_failed_update_depth"}:
        return "failed_update_depth"
    return pipeline


def _resolve_output_dir(args: argparse.Namespace, config: Dict[str, Any], pipeline: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    task = _normalize_task(args.task or config.get("task", "task_a"))
    model = args.model or config.get("model", "7B")
    folder = _output_pipeline_name(pipeline)
    return REPO_ROOT / "analysis" / "depth_and_noise" / "outputs" / task / model / folder


def _augment_one(
    case: Dict[str, Any],
    *,
    pipeline: str,
    config: Dict[str, Any],
    seed: int,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if pipeline == "rd_failed_stay_depth":
        n = int(config.get("n_redundant_single", config.get("n_redundant", [3])[0]))
        return augment_rd_failed_stay_depth(case, n_redundant=n, seed=seed)
    if pipeline == "cd_failed_stay_depth":
        n = int(config.get("n_redundant_single", config.get("n_redundant", [3])[0]))
        return augment_cd_failed_stay_depth(case, n_redundant=n, seed=seed)
    if pipeline == "rd_failed_update_depth":
        delay = int(config.get("delay_turns_single", config.get("delay_turns", [3])[0]))
        return augment_rd_failed_update_depth(case, delay_turns=delay, seed=seed)
    if pipeline == "cd_failed_update_depth":
        delay = int(config.get("delay_turns_single", config.get("delay_turns", [3])[0]))
        return augment_cd_failed_update_depth(case, delay_turns=delay, seed=seed)
    if pipeline == "noise_typology":
        noise_type = str(config.get("noise_type", "sycophancy"))
        turn_policy = str(config.get("turn_policy", "host_comment"))
        template_dir = config.get("template_dir")
        tpl_path = Path(template_dir) if template_dir else None
        return augment_noise_typology(
            case,
            noise_type=noise_type,
            seed=seed,
            turn_policy=turn_policy,
            template_dir=tpl_path,
        )
    raise ValueError(f"Unknown pipeline: {pipeline}")


def run_augment(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    pipeline = args.pipeline or config.get("pipeline")
    if not pipeline:
        raise ValueError("pipeline is required")
    task = _normalize_task(args.task or config.get("task", "task_a"))
    model = args.model or config.get("model", "7B")
    if pipeline == "failed_stay_depth":
        pipeline = _failed_stay_pipeline_for_task(task)
    elif pipeline == "failed_update_depth":
        pipeline = _failed_update_pipeline_for_task(task)

    input_dir = _resolve_input_dir(args, config)
    output_dir = _resolve_output_dir(args, config, pipeline)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    limit = args.limit if args.limit is not None else config.get("limit")
    max_source_cases = args.max_source_cases
    if max_source_cases is None:
        max_source_cases = config.get("max_source_cases")

    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists() and args.overwrite:
        manifest_path.unlink()

    records: List[Dict[str, Any]] = []
    count = 0
    attempted_source_count = 0
    successful_source_count = 0
    for path in iter_input_cases(input_dir):
        case = _normalize_case_fields(load_case(path), task)
        if pipeline == "noise_typology":
            case = normalize_failed_isolation_source_case(path, case, task)
            case = _normalize_case_fields(case, task)
        expected_type = PIPELINES.get(pipeline)
        if expected_type and str(case.get("cbm_challenge_type", "")).lower() != expected_type:
            continue
        if max_source_cases is not None and successful_source_count >= int(max_source_cases):
            break
        attempted_source_count += 1

        param_grid: List[Dict[str, Any]] = [dict(config)]
        if pipeline in {"rd_failed_stay_depth", "cd_failed_stay_depth"} and isinstance(config.get("n_redundant"), list):
            depths = [int(n) for n in config["n_redundant"]]
            param_grid = [{**config, "n_redundant_single": max(depths), "eval_depths": depths}]
        elif pipeline in {"rd_failed_update_depth", "cd_failed_update_depth"} and isinstance(config.get("delay_turns"), list):
            param_grid = [{**config, "delay_turns_single": d} for d in config["delay_turns"]]
        elif pipeline == "noise_typology" and isinstance(config.get("noise_types"), list):
            noise_types = list(config["noise_types"])
            if str(model) == "api":
                noise_types = [nt for nt in noise_types if str(nt) != "none"]
            param_grid = [{**config, "noise_type": nt} for nt in noise_types]

        pending_records: List[Dict[str, Any]] = []
        pending_writes: List[Tuple[Path, Dict[str, Any]]] = []
        source_has_skip = False
        for idx, run_config in enumerate(param_grid):
            run_seed = seed + idx
            augmented, skip_reason = _augment_one(case, pipeline=pipeline, config=run_config, seed=run_seed)
            if augmented is None:
                source_has_skip = True
                pending_records.append(
                    manifest_record(
                        case_id=str(case.get("case_id", path.stem)),
                        source_path=str(path),
                        output_path="",
                        pipeline=pipeline,
                        params=run_config,
                        case_hash="",
                        status="skipped",
                        skip_reason=skip_reason,
                    )
                )
                continue
            if pipeline in {"rd_failed_stay_depth", "cd_failed_stay_depth"} and run_config.get("eval_depths"):
                augmented.setdefault("augmentation", {})["eval_depths"] = list(run_config["eval_depths"])
                augmented["augmentation"].setdefault("params", {})["eval_depths"] = list(run_config["eval_depths"])

            out_name = f"{augmented['case_id']}.json"
            out_path = _case_output_dir(output_dir, pipeline) / out_name
            case_hash = stable_case_hash(augmented)
            pending_writes.append((out_path, augmented))
            pending_records.append(
                manifest_record(
                    case_id=str(augmented["case_id"]),
                    source_path=str(path),
                    output_path=str(out_path),
                    pipeline=pipeline,
                    params={k: run_config.get(k) for k in run_config if not k.endswith("_single")},
                    case_hash=case_hash,
                    status="ok",
                )
            )

        require_full_grid = pipeline in {"rd_failed_update_depth", "cd_failed_update_depth"}
        source_complete = bool(pending_writes) and (not require_full_grid or len(pending_writes) == len(param_grid))

        if source_complete:
            for out_path, augmented in pending_writes:
                write_case(out_path, augmented)
            records.extend(pending_records)
            count += len(pending_writes)
            successful_source_count += 1
        elif source_has_skip:
            for record in pending_records:
                if record.get("status") == "skipped":
                    records.append(record)

        if limit is not None and count >= int(limit):
            break

    for record in records:
        write_manifest_line(manifest_path, record)
    write_augment_summary(output_dir, records)
    print(
        json.dumps(
            {
                "pipeline": pipeline,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "attempted_source_cases": attempted_source_count,
                "successful_source_cases": successful_source_count,
                "written": count,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BELIEFSHIFT analysis augmentation CLI")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        choices=list(PIPELINES.keys()) + ["failed_stay_depth", "failed_update_depth"],
    )
    parser.add_argument("--task", choices=["task_a", "task_b"], default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--challenge-type", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-source-cases",
        type=int,
        default=None,
        help="Limit source cases before expanding each case across the full parameter grid.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_augment(args)


if __name__ == "__main__":
    raise SystemExit(main())
