#!/usr/bin/env python3
"""Run belief-probe ranking prompts and save compact JSON results."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llm_backend import APIBackend, VLLMBackend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "belief_probe_dataset_task_a.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "task_a" / "9B" / "probing" / "base"
DEFAULT_MODEL_PATH = "models/Qwen3.5-9B"

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


def read_json_records(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [item for item in payload["records"] if isinstance(item, dict)]
    raise ValueError(f"unsupported JSON record file: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_wrappers(text: str) -> str:
    text = THINK_RE.sub("", str(text or "")).strip()
    return CODE_FENCE_RE.sub("", text).strip()


def parse_ranking(text: str) -> Tuple[List[str], bool]:
    cleaned = strip_wrappers(text)
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        candidates.append(cleaned[cleaned.find("{") : cleaned.rfind("}") + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        ranking = payload.get("ranking") if isinstance(payload, dict) else None
        if not isinstance(ranking, list):
            return [], isinstance(payload, dict)
        output: List[str] = []
        seen: set[str] = set()
        for item in ranking:
            value = str(item).strip()
            if value and value not in seen:
                seen.add(value)
                output.append(value)
        return output, True
    return [], False


def complete_ranking(ranking: Sequence[str], candidates: Sequence[str]) -> List[str]:
    allowed = set(candidates)
    cleaned = [item for item in ranking if item in allowed]
    seen = set(cleaned)
    return cleaned + [item for item in candidates if item not in seen]


def ranking_metrics(record: Dict[str, Any], ranking: Sequence[str]) -> Dict[str, Any]:
    candidates = [str(item) for item in record.get("candidate_rules") or []]
    oracle_value = str(record.get("oracle") or "").strip()
    completed = complete_ranking(ranking, candidates)
    default_rank = len(candidates) + 1
    oracle_rank = (completed.index(oracle_value) + 1) if oracle_value in completed else default_rank
    return {
        "completed_ranking": completed,
        "oracle_rank": oracle_rank,
        "oracle_top1": bool(oracle_rank == 1),
        "oracle_top3": bool(oracle_rank <= 3),
    }


def make_backend(args: argparse.Namespace):
    if args.backend == "api":
        return APIBackend(
            api_base_url=args.api_base_url,
            model_name=args.api_model_name,
            api_key=args.api_key,
            max_workers=args.max_workers,
            enable_thinking=args.enable_thinking,
        )

    env_lib = os.environ.get("BELIEF_TRAINING_ENV", ".conda/envs/swift") + "/lib"
    os.environ["LD_LIBRARY_PATH"] = f"{env_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    return VLLMBackend(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs if args.max_num_seqs > 0 else None,
        max_num_batched_tokens=args.max_num_batched_tokens if args.max_num_batched_tokens > 0 else None,
        disable_custom_all_reduce=True,
        enforce_eager=True,
        enable_thinking=args.enable_thinking,
    )


def trajectory_key(record: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(record.get("source_file", "")),
            str(record.get("case_id", "")),
            str(record.get("repeat_index", 0)),
        ]
    )


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in results:
        grouped[trajectory_key(record)].append(record)

    final_wrong = []
    for items in grouped.values():
        items.sort(key=lambda item: int(item.get("turn_index", 0) or 0))
        if items and not bool(items[-1].get("model_matches_golden")):
            final_wrong.append(items[-1])

    return {
        "records": len(results),
        "trajectories": len(grouped),
        "final_wrong_trajectories": len(final_wrong),
        "probe_position": "post_answer",
    }


def run(args: argparse.Namespace) -> None:
    records = read_json_records(args.input)
    if args.limit > 0:
        records = records[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "probe_ranking_results.json"
    summary_path = args.output_dir / "probe_ranking_summary.json"
    config_path = args.output_dir / "probe_ranking_run_config.json"
    if args.overwrite and results_path.exists():
        results_path.unlink()

    results = read_json_records(results_path) if results_path.exists() else []
    done = {str(item.get("probe_id")) for item in results}
    pending = [record for record in records if str(record.get("probe_id")) not in done]
    write_json(config_path, vars(args) | {"output_dir": str(args.output_dir), "input": str(args.input)})

    print(f"[run-probe] total={len(records)} done={len(done)} pending={len(pending)}")
    if pending:
        backend = make_backend(args)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            responses = backend.batch_chat_completion(
                [record["messages"] for record in batch],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                use_tqdm=False,
            )
            for record, response in zip(batch, responses):
                parsed_ranking, parsed_ok = parse_ranking(response)
                result = dict(record)
                result.update(
                    {
                        "response_text": response,
                        "parsed_ranking": parsed_ranking,
                        "parsed_ok": parsed_ok,
                    }
                )
                result.update(ranking_metrics(record, parsed_ranking))
                results.append(result)
            write_json(results_path, results)
            write_json(summary_path, summarize(results))
            print(f"[run-probe] wrote {len(results)}/{len(records)}")

    write_json(summary_path, summarize(results))
    print(f"[run-probe] results={results_path}")
    print(f"[run-probe] summary={summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run belief-probe ranking prompts")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=["vllm", "api"], default="vllm")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL", ""))
    parser.add_argument("--api-model-name", default=os.environ.get("API_MODEL_NAME", "deepseek-v3.2"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
