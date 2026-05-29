#!/usr/bin/env python3
"""Run EasySteer/vLLM prefix last-token activation-addition interventions."""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EASYSTEER_ROOT = Path(os.environ.get("EASYSTEER_ROOT", REPO_ROOT / "analysis" / "steering" / "EasySteer"))
for _path in (EASYSTEER_ROOT / "vllm-steer", EASYSTEER_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch

from analysis.steering.common import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    extract_hypothesis_block,
    hypotheses_match,
    normalize_layers,
    parse_hypotheses,
    read_json,
    render_chat_prompt,
    response_hash,
    write_json,
)
from analysis.steering.easysteer_compat import patch_supported_decoder_layers  # noqa: E402


def parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_enable_thinking(value: str) -> Optional[bool]:
    if value == "auto":
        return None
    return value == "true"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unknown"


def load_vectors(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def select_records(dataset: Dict[str, Any], max_records: int) -> List[Dict[str, Any]]:
    records = list(dataset.get("records", {}).get("heldout", []))
    if max_records > 0:
        records = records[:max_records]
    return records


def make_llm(model_path: str, args: argparse.Namespace):
    added_layers = patch_supported_decoder_layers()
    if added_layers:
        print(f"[easysteer] registered decoder layers: {added_layers}", flush=True)

    from vllm import LLM

    kwargs: Dict[str, Any] = {
        "model": str(model_path),
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "max_model_len": int(args.max_length),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enable_steer_vector": True,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "seed": int(args.seed),
    }
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if int(args.max_num_seqs) > 0:
        kwargs["max_num_seqs"] = int(args.max_num_seqs)
    if int(args.max_num_batched_tokens) > 0:
        kwargs["max_num_batched_tokens"] = int(args.max_num_batched_tokens)
    return LLM(**kwargs)


def make_sampling_params(args: argparse.Namespace, seed: int):
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        presence_penalty=float(args.presence_penalty),
        repetition_penalty=float(args.repetition_penalty),
        seed=int(seed),
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )


def make_steer_request(
    *,
    name: str,
    request_id: int,
    vector_path: Path,
    layer: int,
    alpha: float,
    debug: bool,
):
    from vllm.steer_vectors.request import SteerVectorRequest

    return SteerVectorRequest(
        steer_vector_name=name,
        steer_vector_int_id=int(request_id),
        steer_vector_local_path=str(vector_path),
        scale=float(alpha),
        target_layers=[int(layer)],
        prefill_trigger_positions=[-1],
        generate_trigger_tokens=None,
        algorithm="direct",
        normalize=False,
        debug=bool(debug),
    )


def encode_prompt(tokenizer: Any, prompt: str, max_prompt_tokens: int) -> List[int]:
    try:
        token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    except TypeError:
        token_ids = list(tokenizer.encode(prompt))
    if max_prompt_tokens > 0 and len(token_ids) > max_prompt_tokens:
        token_ids = token_ids[-max_prompt_tokens:]
    if not token_ids:
        raise ValueError("empty prompt after tokenization")
    return [int(token_id) for token_id in token_ids]


def render_prompt(record: Dict[str, Any], tokenizer: Any, enable_thinking: Optional[bool], history_mode: str) -> str:
    messages_key = "messages_base_context" if history_mode == "base_context" else "messages_common"
    return render_chat_prompt(tokenizer, record[messages_key], enable_thinking=enable_thinking)


def prepare_prompts(
    *,
    records: Sequence[Dict[str, Any]],
    tokenizer: Any,
    enable_thinking: Optional[bool],
    history_mode: str,
    max_length: int,
    max_new_tokens: int,
) -> Tuple[Dict[str, str], Dict[str, List[int]], Dict[str, int]]:
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"
    max_prompt_tokens = int(max_length) - int(max_new_tokens)
    if max_prompt_tokens < 1:
        raise ValueError(
            f"max-length ({max_length}) must be greater than max-new-tokens ({max_new_tokens})"
        )
    prompt_texts: Dict[str, str] = {}
    prompt_token_ids: Dict[str, List[int]] = {}
    prompt_lengths: Dict[str, int] = {}
    for record in records:
        record_id = str(record["record_id"])
        prompt = render_prompt(record, tokenizer, enable_thinking, history_mode)
        token_ids = encode_prompt(tokenizer, prompt, max_prompt_tokens)
        prompt_texts[record_id] = prompt
        prompt_token_ids[record_id] = token_ids
        prompt_lengths[record_id] = len(token_ids)
    return prompt_texts, prompt_token_ids, prompt_lengths


def expanded_prompt_inputs(
    expanded_items: Sequence[Dict[str, Any]],
    prompt_texts: Dict[str, str],
    prompt_token_ids: Dict[str, List[int]],
) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    for item in expanded_items:
        record = item["record"]
        record_id = str(record["record_id"])
        prompts.append(
            {
                "prompt_token_ids": prompt_token_ids[record_id],
                "prompt": prompt_texts[record_id],
            }
        )
    return prompts


def make_random_vector(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_vec = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
    random_vec = random_vec / max(float(random_vec.norm().item()), 1e-12)
    return random_vec * reference.norm()


def condition_vector(
    vectors: Dict[str, Any],
    condition: str,
    layer: int,
    seed: int,
    vector_source: str,
) -> torch.Tensor:
    source_key = {
        "belief": "belief_vectors",
        "raw": "raw_divergence_vectors",
        "nuisance": "nuisance_vectors",
    }[vector_source]
    by_type_key = f"{source_key}_by_type"
    if by_type_key in vectors and condition in vectors[by_type_key]:
        return vectors[by_type_key][condition][layer].float()
    if condition.startswith("negative_"):
        challenge_type = condition.removeprefix("negative_")
        if by_type_key not in vectors or challenge_type not in vectors[by_type_key]:
            raise ValueError(f"vector file does not contain {by_type_key}[{challenge_type!r}]")
        return -vectors[by_type_key][challenge_type][layer].float()
    source = vectors[source_key][layer].float()
    if condition == "belief":
        return source
    if condition == "negative":
        return -source
    if condition == "nuisance":
        return vectors["nuisance_vectors"][layer].float()
    if condition == "random":
        return make_random_vector(source, seed + layer * 9973)
    raise ValueError(f"unknown condition: {condition}")


def export_condition_vector(
    *,
    vector: torch.Tensor,
    output_dir: Path,
    condition: str,
    vector_source: str,
    layer: int,
) -> Path:
    vector_dir = output_dir / "easysteer_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    path = vector_dir / f"intervention_{safe_name(vector_source)}_{safe_name(condition)}_L{int(layer)}.pt"
    torch.save(vector.detach().float().cpu(), path)
    return path


def existing_results(path: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    results = read_json(path)
    if not isinstance(results, list):
        raise ValueError(f"{path} is not a result list")
    compacted = [compact_result_row(item) for item in results]
    done = {str(item.get("run_id")) for item in compacted}
    return compacted, done


def load_reusable_baseline(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        rows = payload["results"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{path} is neither a result list nor a split result object")

    baseline: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("condition") != "no_steer":
            continue
        normalized = dict(row)
        if not normalized.get("run_id"):
            if "generation_repeat" not in normalized or "record_id" not in normalized:
                raise ValueError(f"baseline row in {path} is missing run_id and cannot be reconstructed")
            normalized["run_id"] = f"no_steer::r{int(normalized['generation_repeat'])}::{normalized['record_id']}"
        normalized["condition"] = "no_steer"
        normalized["layer"] = None
        normalized["alpha"] = 0.0
        normalized["intervention_mode"] = normalized.get("intervention_mode") or "none"
        baseline.append(compact_result_row(normalized))
    if not baseline:
        raise ValueError(f"{path} does not contain reusable no_steer baseline rows")
    return baseline


def merge_reusable_baseline(results: List[Dict[str, Any]], done: set[str], path: Path) -> int:
    added = 0
    for row in load_reusable_baseline(path):
        run_id = str(row["run_id"])
        if run_id in done:
            continue
        results.append(row)
        done.add(run_id)
        added += 1
    return added


COMPACT_RESULT_FIELDS = [
    "run_id",
    "condition",
    "layer",
    "alpha",
    "intervention_mode",
    "vector_source",
    "record_id",
    "case_id",
    "source_tag",
    "source_index",
    "repeat_index",
    "challenge_type",
    "target_turn",
    "num_turns",
    "prompt_token_length",
    "golden_hypotheses",
    "parsed_hypotheses",
    "parse_ok",
    "correct",
    "response_hash",
    "response_text",
    "generation_repeat",
]


def compact_result_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: row[key] for key in COMPACT_RESULT_FIELDS if key in row}


def alpha_file_path(output_dir: Path, condition: str, layer: int, alpha: float) -> Path:
    return output_dir / "interventions" / f"{safe_name(condition)}_L{int(layer)}_a{alpha:g}.json"


def write_baseline_split(output_dir: Path, rows: Sequence[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    baseline_rows = [compact_result_row(row) for row in rows if row.get("condition") == "no_steer"]
    write_json(
        output_dir / "baseline_results.json",
        {
            "metadata": metadata,
            "results": baseline_rows,
        },
    )


def write_alpha_split(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    condition: str,
    layer: int,
    alpha: float,
) -> None:
    group_rows = [
        compact_result_row(row)
        for row in rows
        if row.get("condition") == condition
        and int(row.get("layer")) == int(layer)
        and float(row.get("alpha")) == float(alpha)
    ]
    write_json(
        alpha_file_path(output_dir, condition, layer, alpha),
        {
            "metadata": metadata,
            "results": group_rows,
        },
    )


def build_result_rows(
    *,
    expanded_items: Sequence[Dict[str, Any]],
    outputs: Sequence[Any],
    prompt_lengths: Dict[str, int],
    condition: str,
    layer: Optional[int],
    alpha: float,
    intervention_mode: str,
    vector_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if len(outputs) != len(expanded_items):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(expanded_items)} prompts")
    for item, output in zip(expanded_items, outputs):
        record = item["record"]
        if not getattr(output, "outputs", None):
            response = ""
        else:
            response = str(output.outputs[0].text).strip()
        parsed = parse_hypotheses(response)
        golden = list(record.get("golden_hypotheses") or [])
        record_id = str(record["record_id"])
        row = {
            "run_id": item["run_id"],
            "condition": condition,
            "layer": layer,
            "alpha": float(alpha),
            "intervention_mode": intervention_mode,
            "vector_source": vector_source,
            "record_id": record["record_id"],
            "case_id": record["case_id"],
            "repeat_index": record["repeat_index"],
            "challenge_type": record["challenge_type"],
            "target_turn": record["target_turn"],
            "prompt_token_length": int(prompt_lengths[record_id]),
            "golden_hypotheses": golden,
            "parsed_hypotheses": parsed,
            "parse_ok": parsed is not None,
            "correct": hypotheses_match(parsed, golden),
            "response_hash": response_hash(response),
            "response_text": response,
            "generation_repeat": int(item["generation_repeat"]),
        }
        rows.append(compact_result_row(row))
    return rows


def run_expanded_generate(
    *,
    llm: Any,
    expanded_items: Sequence[Dict[str, Any]],
    sampling_params: Sequence[Any],
    steer_request: Any,
    prompt_texts: Dict[str, str],
    prompt_token_ids: Dict[str, List[int]],
    prompt_lengths: Dict[str, int],
    condition: str,
    layer: Optional[int],
    alpha: float,
    intervention_mode: str,
    vector_source: Optional[str],
    use_tqdm: bool,
) -> List[Dict[str, Any]]:
    if not expanded_items:
        return []
    prompts = expanded_prompt_inputs(expanded_items, prompt_texts, prompt_token_ids)
    outputs = llm.generate(
        prompts,
        sampling_params=list(sampling_params),
        steer_vector_request=steer_request,
        use_tqdm=bool(use_tqdm),
    )
    return build_result_rows(
        expanded_items=expanded_items,
        outputs=outputs,
        prompt_lengths=prompt_lengths,
        condition=condition,
        layer=layer,
        alpha=alpha,
        intervention_mode=intervention_mode,
        vector_source=vector_source,
    )


def failed_isolation_state_run_id(
    *,
    record: Dict[str, Any],
    generation_repeat: int,
    condition: str,
    layer: Optional[int],
    alpha: float,
    intervention_mode: str,
) -> str:
    if condition == "no_steer":
        return f"no_steer::r{generation_repeat}::{record['record_id']}"
    return (
        f"{condition}::{intervention_mode}::L{int(layer)}::a{alpha:g}::"
        f"r{generation_repeat}::{record['record_id']}"
    )


def failed_isolation_build_states(
    records: Sequence[Dict[str, Any]],
    repeats: int,
    done: set[str],
    *,
    condition: str,
    layer: Optional[int],
    alpha: float,
    intervention_mode: str,
) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for generation_repeat in range(repeats):
        for record in records:
            run_id = failed_isolation_state_run_id(
                record=record,
                generation_repeat=generation_repeat,
                condition=condition,
                layer=layer,
                alpha=alpha,
                intervention_mode=intervention_mode,
            )
            if run_id in done:
                continue
            states.append(
                {
                    "record": record,
                    "generation_repeat": generation_repeat,
                    "run_id": run_id,
                    "messages": [{"role": "system", "content": str(record.get("system_prompt") or "")}],
                    "turn_responses": [],
                    "turn_response_hashes": [],
                    "prompt_token_length": 0,
                }
            )
    return states


def failed_isolation_baseline_missing(
    records: Sequence[Dict[str, Any]],
    repeats: int,
    done: set[str],
) -> List[Dict[str, Any]]:
    return failed_isolation_build_states(
        records,
        repeats,
        done,
        condition="no_steer",
        layer=None,
        alpha=0.0,
        intervention_mode="none",
    )


def failed_isolation_prompt_input(
    *,
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    enable_thinking: Optional[bool],
    max_prompt_tokens: int,
) -> Dict[str, Any]:
    prompt = render_chat_prompt(tokenizer, messages, enable_thinking=enable_thinking)
    return {
        "prompt": prompt,
        "prompt_token_ids": encode_prompt(tokenizer, prompt, max_prompt_tokens),
    }


def failed_isolation_seed_for_state(
    args: argparse.Namespace,
    state: Dict[str, Any],
    *,
    layer: int,
    alpha: float,
    turn_index: int,
) -> int:
    return (
        int(args.seed)
        + int(state["generation_repeat"])
        + int(layer) * 1000
        + int(float(alpha) * 100)
        + int(turn_index) * 10000
    )


def failed_isolation_build_result_rows(
    states: Sequence[Dict[str, Any]],
    *,
    condition: str,
    layer: Optional[int],
    alpha: float,
    intervention_mode: str,
    vector_source: Optional[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for state in states:
        record = state["record"]
        response = state["turn_responses"][-1].strip() if state["turn_responses"] else ""
        parsed = parse_hypotheses(response)
        golden = list(record.get("golden_hypotheses") or [])
        row = {
            "run_id": state["run_id"],
            "condition": condition,
            "layer": layer,
            "alpha": float(alpha),
            "intervention_mode": intervention_mode,
            "vector_source": vector_source,
            "record_id": record["record_id"],
            "case_id": record["case_id"],
            "source_tag": record.get("source_tag"),
            "source_index": record.get("source_index"),
            "repeat_index": record["repeat_index"],
            "challenge_type": record["challenge_type"],
            "target_turn": record["target_turn"],
            "num_turns": int(record.get("num_turns") or len(record.get("turns") or [])),
            "prompt_token_length": int(state.get("prompt_token_length") or 0),
            "golden_hypotheses": golden,
            "parsed_hypotheses": parsed,
            "parse_ok": parsed is not None,
            "correct": hypotheses_match(parsed, golden),
            "response_hash": response_hash(response),
            "response_text": response,
            "generation_repeat": int(state["generation_repeat"]),
        }
        rows.append(compact_result_row(row))
    return rows


def failed_isolation_run_live_generate(
    *,
    llm: Any,
    tokenizer: Any,
    states: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    steer_request: Any,
    condition: str,
    layer: Optional[int],
    alpha: float,
    intervention_mode: str,
    vector_source: Optional[str],
    enable_thinking: Optional[bool],
) -> List[Dict[str, Any]]:
    if not states:
        return []
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"
    max_prompt_tokens = int(args.max_length) - int(args.max_new_tokens)
    if max_prompt_tokens < 1:
        raise ValueError(f"max-length ({args.max_length}) must be greater than max-new-tokens ({args.max_new_tokens})")

    max_turns = max(len(state["record"].get("turns") or []) for state in states)
    active_states = list(states)
    for turn_index in range(max_turns):
        turn_states = [
            state
            for state in active_states
            if turn_index < len(state["record"].get("turns") or [])
        ]
        if not turn_states:
            continue
        prompts: List[Dict[str, Any]] = []
        sampling_params: List[Any] = []
        for state in turn_states:
            turn = state["record"]["turns"][turn_index]
            state["messages"].append({"role": "user", "content": str(turn.get("prompt") or "")})
            prompt = failed_isolation_prompt_input(
                tokenizer=tokenizer,
                messages=state["messages"],
                enable_thinking=enable_thinking,
                max_prompt_tokens=max_prompt_tokens,
            )
            state["prompt_token_length"] = len(prompt["prompt_token_ids"])
            prompts.append(prompt)
            sampling_params.append(
                make_sampling_params(
                    args,
                    failed_isolation_seed_for_state(
                        args,
                        state,
                        layer=int(layer or 0),
                        alpha=float(alpha),
                        turn_index=turn_index,
                    ),
                )
            )
        print(
            f"[failed_isolation:easysteer] {condition} L{layer} alpha={alpha:g} "
            f"turn={turn_index} prompts={len(prompts)}",
            flush=True,
        )
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            steer_vector_request=steer_request,
            use_tqdm=bool(args.use_tqdm),
        )
        if len(outputs) != len(turn_states):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(turn_states)} prompts")
        for state, output in zip(turn_states, outputs):
            response = str(output.outputs[0].text).strip() if getattr(output, "outputs", None) else ""
            state["turn_responses"].append(response)
            state["turn_response_hashes"].append(response_hash(response))
            state["messages"].append({"role": "assistant", "content": extract_hypothesis_block(response)})

    return failed_isolation_build_result_rows(
        states,
        condition=condition,
        layer=layer,
        alpha=alpha,
        intervention_mode=intervention_mode,
        vector_source=vector_source,
    )


def baseline_items(records: Sequence[Dict[str, Any]], repeats: int, done: set[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for generation_repeat in range(repeats):
        for record in records:
            run_id = f"no_steer::r{generation_repeat}::{record['record_id']}"
            if run_id in done:
                continue
            items.append(
                {
                    "record": record,
                    "generation_repeat": generation_repeat,
                    "run_id": run_id,
                }
            )
    return items


def intervention_items(
    records: Sequence[Dict[str, Any]],
    repeats: int,
    done: set[str],
    condition: str,
    layer: int,
    alpha: float,
    intervention_mode: str,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for generation_repeat in range(repeats):
        run_prefix = f"{condition}::{intervention_mode}::L{layer}::a{alpha:g}::r{generation_repeat}"
        for record in records:
            run_id = f"{run_prefix}::{record['record_id']}"
            if run_id in done:
                continue
            items.append(
                {
                    "record": record,
                    "generation_repeat": generation_repeat,
                    "run_id": run_id,
                }
            )
    return items


def seeds_for_items(args: argparse.Namespace, items: Sequence[Dict[str, Any]], *, layer: int = 0, alpha: float = 0.0) -> List[Any]:
    return [
        make_sampling_params(
            args,
            args.seed + int(item["generation_repeat"]) + int(layer) * 1000 + int(float(alpha) * 100),
        )
        for item in items
    ]


def write_compact_results(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_json(path, [compact_result_row(row) for row in rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-mode", choices=["standard", "failed_isolation"], default="standard")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--layers", default="", help="Comma-separated layers. Empty means all vector layers.")
    parser.add_argument("--alphas", default="0.05,0.1,0.25")
    parser.add_argument("--conditions", default="failed_stay")
    parser.add_argument("--max-heldout", type=int, default=0)
    parser.add_argument("--history-mode", choices=["base_context", "canonical"], default="base_context")
    parser.add_argument("--vector-source", choices=["belief", "raw", "nuisance"], default="raw")
    parser.add_argument("--batch-size", type=int, default=16, help="Compatibility only; vLLM generation uses expanded records x repeats.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=121400)
    parser.add_argument("--max-new-tokens", type=int, default=30000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-thinking", choices=["true", "false", "auto"], default="true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--intervention-mode", choices=["prefill_last_token"], default="prefill_last_token")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--debug-steer", action="store_true", default=False)
    parser.add_argument("--use-tqdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-baseline-from", type=Path, default=None)
    parser.add_argument("--require-reused-baseline", action="store_true", default=False)
    parser.add_argument("--run-baseline-only", action="store_true", default=False)
    args = parser.parse_args()

    dataset = read_json(args.dataset)
    records = select_records(dataset, args.max_heldout)
    enable_thinking = parse_enable_thinking(args.enable_thinking)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results, done = existing_results(args.output)
    if args.reuse_baseline_from is not None:
        added = merge_reusable_baseline(results, done, args.reuse_baseline_from)
        print(
            f"[intervene:easysteer] reused baseline rows={added} from {args.reuse_baseline_from}",
            flush=True,
        )
        write_compact_results(args.output, results)

    if args.dataset_mode == "failed_isolation":
        alphas = parse_floats(args.alphas)
        conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
        if args.conditions == "failed_stay" and records and all(
            str(record.get("challenge_type")) == "failed_isolation" for record in records
        ):
            conditions = ["failed_isolation"]
        missing_baseline = failed_isolation_baseline_missing(records, args.repeats, done)
        if args.require_reused_baseline and missing_baseline:
            raise RuntimeError(
                f"reused baseline does not cover current dataset/repeats: missing={len(missing_baseline)}. "
                f"source={args.reuse_baseline_from}"
            )

        vector_payload = None
        layers: List[int] = []
        if not args.run_baseline_only:
            if args.vectors is None:
                raise ValueError("--vectors is required unless --run-baseline-only is set")
            vector_payload = load_vectors(args.vectors)
            available_layers = [int(layer) for layer in vector_payload["layers"]]
            requested_layers = available_layers if not args.layers.strip() else parse_ints(args.layers)
            layers = requested_layers
            missing = sorted(set(layers) - set(available_layers))
            if missing:
                raise ValueError(f"requested layers missing from vector file: {missing}")

        run_metadata = {
            "backend": "easysteer_vllm",
            "dataset": str(args.dataset),
            "vectors": str(args.vectors) if args.vectors else None,
            "base_model": str(args.base_model),
            "max_new_tokens": int(args.max_new_tokens),
            "max_length": int(args.max_length),
            "enable_thinking": enable_thinking,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "presence_penalty": float(args.presence_penalty),
            "repetition_penalty": float(args.repetition_penalty),
            "intervention_mode": args.intervention_mode,
            "prefill_trigger_positions": [-1],
            "generate_trigger_tokens": None,
            "expanded_repeats": int(args.repeats),
            "row_format": "compact_live_failed_isolation_v1",
            "live_multiturn": True,
            "assistant_context": "hypothesis_block_only",
            "final_turn_only_scoring": True,
            "use_tqdm": bool(args.use_tqdm),
            "reuse_baseline_from": str(args.reuse_baseline_from) if args.reuse_baseline_from else None,
        }
        write_json(args.output.parent / "intervention_run_config.json", run_metadata)

        print(
            f"[failed_isolation:easysteer] records={len(records)} layers={layers} alphas={alphas} "
            f"conditions={conditions} done={len(done)} baseline_only={args.run_baseline_only}",
            flush=True,
        )

        llm = make_llm(args.base_model, args)
        tokenizer = llm.get_tokenizer()
        try:
            pending_baseline = failed_isolation_baseline_missing(records, args.repeats, done)
            print(
                f"[failed_isolation:easysteer] baseline expanded live runs={len(pending_baseline)}",
                flush=True,
            )
            baseline_rows = failed_isolation_run_live_generate(
                llm=llm,
                tokenizer=tokenizer,
                states=pending_baseline,
                args=args,
                steer_request=None,
                condition="no_steer",
                layer=None,
                alpha=0.0,
                intervention_mode="none",
                vector_source=None,
                enable_thinking=enable_thinking,
            )
            for row in baseline_rows:
                results.append(row)
                done.add(str(row["run_id"]))
            write_compact_results(args.output, results)
            write_baseline_split(
                args.output.parent,
                results,
                {
                    **run_metadata,
                    "condition": "no_steer",
                    "layer": None,
                    "alpha": 0.0,
                    "expanded_live_runs": len([row for row in results if row.get("condition") == "no_steer"]),
                },
            )

            if args.run_baseline_only:
                print(f"[failed_isolation:easysteer] wrote {args.output}")
                return 0

            assert vector_payload is not None
            request_counter = 2000
            for layer in layers:
                for condition_index, condition in enumerate(conditions):
                    vector = condition_vector(vector_payload, condition, layer, args.seed, args.vector_source)
                    vector_path = export_condition_vector(
                        vector=vector,
                        output_dir=args.output.parent,
                        condition=condition,
                        vector_source=args.vector_source,
                        layer=layer,
                    )
                    for alpha_index, alpha in enumerate(alphas):
                        pending = failed_isolation_build_states(
                            records,
                            args.repeats,
                            done,
                            condition=condition,
                            layer=layer,
                            alpha=alpha,
                            intervention_mode=args.intervention_mode,
                        )
                        print(
                            f"[failed_isolation:easysteer] {condition} L{layer} alpha={alpha:g} "
                            f"expanded live runs={len(pending)}",
                            flush=True,
                        )
                        if pending:
                            request_counter += 1
                            request_id = (
                                200000 + int(layer) * 1000 + condition_index * 100 + alpha_index * 10 + request_counter
                            )
                            steer_request = make_steer_request(
                                name=f"failed_isolation_L{layer}_a{alpha:g}",
                                request_id=request_id,
                                vector_path=vector_path,
                                layer=layer,
                                alpha=alpha,
                                debug=args.debug_steer,
                            )
                            rows = failed_isolation_run_live_generate(
                                llm=llm,
                                tokenizer=tokenizer,
                                states=pending,
                                args=args,
                                steer_request=steer_request,
                                condition=condition,
                                layer=layer,
                                alpha=alpha,
                                intervention_mode=args.intervention_mode,
                                vector_source=args.vector_source,
                                enable_thinking=enable_thinking,
                            )
                            for row in rows:
                                results.append(row)
                                done.add(str(row["run_id"]))
                            write_compact_results(args.output, results)
                        write_alpha_split(
                            args.output.parent,
                            results,
                            {
                                **run_metadata,
                                "condition": condition,
                                "layer": int(layer),
                                "alpha": float(alpha),
                                "vector_source": args.vector_source,
                                "vector_path": str(vector_path),
                                "vector_norm": float(vector.norm().item()),
                                "result_split": str(alpha_file_path(args.output.parent, condition, layer, alpha)),
                                "steer_vector_request": {
                                    "algorithm": "direct",
                                    "target_layers": [int(layer)],
                                    "scale": float(alpha),
                                    "prefill_trigger_positions": [-1],
                                    "generate_trigger_tokens": None,
                                    "normalize": False,
                                },
                            },
                            condition,
                            layer,
                            alpha,
                        )
        finally:
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_compact_results(args.output, results)
        print(f"[failed_isolation:easysteer] wrote {args.output}")
        return 0

    if args.vectors is None:
        raise ValueError("--vectors is required for standard intervention runs")

    vector_payload = load_vectors(args.vectors)
    available_layers = [int(layer) for layer in vector_payload["layers"]]
    requested_layers = available_layers if not args.layers.strip() else parse_ints(args.layers)
    layers = normalize_layers(requested_layers, max(available_layers) + 1)
    missing = sorted(set(layers) - set(available_layers))
    if missing:
        raise ValueError(f"requested layers missing from vector file: {missing}")

    alphas = parse_floats(args.alphas)
    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
    missing_baseline = baseline_items(records, args.repeats, done)
    if args.require_reused_baseline and missing_baseline:
        raise RuntimeError(
            f"reused baseline does not cover current dataset/repeats: missing={len(missing_baseline)}. "
            f"source={args.reuse_baseline_from}"
        )
    print(
        f"[intervene:easysteer] records={len(records)} layers={layers} alphas={alphas} "
        f"conditions={conditions} done={len(done)} mode={args.intervention_mode}",
        flush=True,
    )

    llm = make_llm(args.base_model, args)
    tokenizer = llm.get_tokenizer()
    prompt_texts, prompt_token_ids, prompt_lengths = prepare_prompts(
        records=records,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        history_mode=args.history_mode,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    records = sorted(records, key=lambda item: prompt_lengths[str(item["record_id"])], reverse=True)
    output_dir = args.output.parent

    run_metadata = {
        "backend": "easysteer_vllm",
        "dataset": str(args.dataset),
        "vectors": str(args.vectors),
        "base_model": str(args.base_model),
        "max_new_tokens": int(args.max_new_tokens),
        "max_length": int(args.max_length),
        "enable_thinking": enable_thinking,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "presence_penalty": float(args.presence_penalty),
        "repetition_penalty": float(args.repetition_penalty),
        "intervention_mode": args.intervention_mode,
        "prefill_trigger_positions": [-1],
        "generate_trigger_tokens": None,
        "expanded_repeats": int(args.repeats),
        "row_format": "compact_v2",
        "use_tqdm": bool(args.use_tqdm),
        "reuse_baseline_from": str(args.reuse_baseline_from) if args.reuse_baseline_from else None,
    }
    write_json(output_dir / "intervention_run_config.json", run_metadata)

    try:
        pending_baseline = baseline_items(records, args.repeats, done)
        print(f"[intervene:easysteer] baseline expanded prompts={len(pending_baseline)}", flush=True)
        baseline_rows = run_expanded_generate(
            llm=llm,
            expanded_items=pending_baseline,
            sampling_params=seeds_for_items(args, pending_baseline),
            steer_request=None,
            prompt_texts=prompt_texts,
            prompt_token_ids=prompt_token_ids,
            prompt_lengths=prompt_lengths,
            condition="no_steer",
            layer=None,
            alpha=0.0,
            intervention_mode="none",
            vector_source=None,
            use_tqdm=args.use_tqdm,
        )
        for result in baseline_rows:
            results.append(result)
            done.add(str(result["run_id"]))
        write_compact_results(args.output, results)
        write_baseline_split(
            output_dir,
            results,
            {
                **run_metadata,
                "condition": "no_steer",
                "layer": None,
                "alpha": 0.0,
                "expanded_prompts": len([row for row in results if row.get("condition") == "no_steer"]),
            },
        )

        request_counter = 1000
        for layer in layers:
            for condition_index, condition in enumerate(conditions):
                vector = condition_vector(vector_payload, condition, layer, args.seed, args.vector_source)
                vector_path = export_condition_vector(
                    vector=vector,
                    output_dir=output_dir,
                    condition=condition,
                    vector_source=args.vector_source,
                    layer=layer,
                )
                for alpha_index, alpha in enumerate(alphas):
                    pending = intervention_items(
                        records,
                        args.repeats,
                        done,
                        condition,
                        layer,
                        alpha,
                        args.intervention_mode,
                    )
                    print(
                        f"[intervene:easysteer] {condition} L{layer} alpha={alpha:g} expanded prompts={len(pending)}",
                        flush=True,
                    )
                    if not pending:
                        write_alpha_split(
                            output_dir,
                            results,
                            {
                                **run_metadata,
                                "condition": condition,
                                "layer": int(layer),
                                "alpha": float(alpha),
                                "vector_source": args.vector_source,
                                "vector_path": str(vector_path),
                                "vector_norm": float(vector.norm().item()),
                            },
                            condition,
                            layer,
                            alpha,
                        )
                        continue
                    request_counter += 1
                    request_id = 100000 + int(layer) * 1000 + condition_index * 100 + alpha_index * 10 + request_counter
                    steer_request = make_steer_request(
                        name=f"{safe_name(condition)}_L{layer}_a{alpha:g}",
                        request_id=request_id,
                        vector_path=vector_path,
                        layer=layer,
                        alpha=alpha,
                        debug=args.debug_steer,
                    )
                    rows = run_expanded_generate(
                        llm=llm,
                        expanded_items=pending,
                        sampling_params=seeds_for_items(args, pending, layer=layer, alpha=alpha),
                        steer_request=steer_request,
                        prompt_texts=prompt_texts,
                        prompt_token_ids=prompt_token_ids,
                        prompt_lengths=prompt_lengths,
                        condition=condition,
                        layer=layer,
                        alpha=alpha,
                        intervention_mode=args.intervention_mode,
                        vector_source=args.vector_source,
                        use_tqdm=args.use_tqdm,
                    )
                    for result in rows:
                        results.append(result)
                        done.add(str(result["run_id"]))
                    write_compact_results(args.output, results)
                    write_alpha_split(
                        output_dir,
                        results,
                        {
                            **run_metadata,
                            "condition": condition,
                            "layer": int(layer),
                            "alpha": float(alpha),
                            "vector_source": args.vector_source,
                            "vector_path": str(vector_path),
                            "vector_norm": float(vector.norm().item()),
                            "steer_vector_request": {
                                "algorithm": "direct",
                                "target_layers": [int(layer)],
                                "scale": float(alpha),
                                "prefill_trigger_positions": [-1],
                                "generate_trigger_tokens": None,
                                "normalize": False,
                            },
                        },
                        condition,
                        layer,
                        alpha,
                    )
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_compact_results(args.output, results)
    print(f"[intervene:easysteer] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
