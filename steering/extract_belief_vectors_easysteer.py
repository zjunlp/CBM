#!/usr/bin/env python3
"""Extract Base-vs-RL steering vectors with EasySteer/vLLM capture."""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EASYSTEER_ROOT = Path(os.environ.get("EASYSTEER_ROOT", "external/EasySteer"))
for _path in (EASYSTEER_ROOT / "vllm-steer", EASYSTEER_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from steering.common import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_RL_MODEL_520,
    normalize_layers,
    read_json,
    render_chat_prompt,
    write_json,
)
from steering.easysteer_compat import patch_supported_decoder_layers  # noqa: E402


def parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_enable_thinking(value: str) -> Optional[bool]:
    if value == "auto":
        return None
    return value == "true"


def flatten_records(dataset: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    records = dataset.get("records", {}).get(split, [])
    if not isinstance(records, list):
        raise ValueError(f"dataset split {split!r} is not a list")
    return records


def iter_batches(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    size = max(1, int(batch_size))
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unknown"


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


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def prompt_input(tokenizer: Any, prompt: str, max_prompt_tokens: int) -> Dict[str, Any]:
    return {
        "prompt_token_ids": encode_prompt(tokenizer, prompt, max_prompt_tokens),
        "prompt": prompt,
    }


def collect_activations(
    *,
    llm: Any,
    records: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    max_length: int,
    batch_size: int,
    enable_thinking: Optional[bool],
    message_key: str,
    desc: str,
) -> Dict[int, torch.Tensor]:
    from easysteer.hidden_states import get_all_hidden_states_generate

    tokenizer = llm.get_tokenizer()
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"

    max_prompt_tokens = max(1, int(max_length) - 1)
    selected_layers: Optional[List[int]] = None
    collected: Dict[int, List[torch.Tensor]] = {}

    for batch in tqdm(list(iter_batches(records, batch_size)), desc=desc):
        prompts = [
            prompt_input(
                tokenizer,
                render_chat_prompt(tokenizer, record[message_key], enable_thinking=enable_thinking),
                max_prompt_tokens,
            )
            for record in batch
        ]
        hidden_samples, _outputs = get_all_hidden_states_generate(
            llm,
            prompts,
            max_tokens=1,
            split_by_samples=True,
            temperature=0.0,
            seed=0,
        )
        if len(hidden_samples) != len(batch):
            raise RuntimeError(
                f"capture returned {len(hidden_samples)} samples for batch size {len(batch)}. "
                "This usually means EasySteer did not wrap the model's decoder layers; "
                "check EASYSTEER_EXTRA_DECODER_LAYERS and the model layer class names."
            )
        for sample_layers in hidden_samples:
            if selected_layers is None:
                selected_layers = normalize_layers(layers, len(sample_layers))
                collected = {layer: [] for layer in selected_layers}
            for layer in selected_layers:
                if layer >= len(sample_layers):
                    raise RuntimeError(f"missing layer {layer}; captured {len(sample_layers)} layers")
                hidden = sample_layers[layer]
                if hidden.shape[0] < 1:
                    raise RuntimeError(f"empty hidden states at layer {layer}")
                # max_tokens=1 capture has length == prompt length, so -1 is the prefix last token.
                collected[layer].append(hidden[-1].detach().float().cpu())

    if selected_layers is None:
        raise ValueError("no records available for vector extraction")
    return {layer: torch.stack(values, dim=0) for layer, values in collected.items()}


def flatten_turn_messages(
    records: Sequence[Dict[str, Any]],
    message_sequence_key: str,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    flat_records: List[Dict[str, Any]] = []
    parent_indices: List[int] = []
    for parent_index, record in enumerate(records):
        messages_by_turn = record.get(message_sequence_key)
        if not isinstance(messages_by_turn, list) or not messages_by_turn:
            raise ValueError(f"{record.get('record_id')} missing non-empty {message_sequence_key}")
        for turn_index, messages in enumerate(messages_by_turn):
            if not isinstance(messages, list):
                raise ValueError(f"{record.get('record_id')} {message_sequence_key}[{turn_index}] is not a list")
            flat_records.append(
                {
                    "record_id": f"{record.get('record_id')}__turn{turn_index}",
                    "messages_for_capture": messages,
                    "parent_record_index": parent_index,
                    "turn_index": turn_index,
                }
            )
            parent_indices.append(parent_index)
    return flat_records, parent_indices


def average_flat_by_parent(
    acts: Dict[int, torch.Tensor],
    parent_indices: Sequence[int],
    parent_count: int,
) -> Dict[int, torch.Tensor]:
    grouped: List[List[int]] = [[] for _ in range(parent_count)]
    for flat_index, parent_index in enumerate(parent_indices):
        grouped[int(parent_index)].append(flat_index)
    if any(not indices for indices in grouped):
        raise ValueError("at least one parent record has no turn activations")

    averaged: Dict[int, torch.Tensor] = {}
    for layer, tensor in acts.items():
        values: List[torch.Tensor] = []
        for indices in grouped:
            index = torch.tensor(indices, dtype=torch.long)
            values.append(tensor.index_select(0, index).mean(dim=0))
        averaged[layer] = torch.stack(values, dim=0)
    return averaged


def collect_turn_average_activations(
    *,
    llm: Any,
    records: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    max_length: int,
    batch_size: int,
    enable_thinking: Optional[bool],
    message_sequence_key: str,
    desc: str,
) -> Dict[int, torch.Tensor]:
    flat_records, parent_indices = flatten_turn_messages(records, message_sequence_key)
    flat_acts = collect_activations(
        llm=llm,
        records=flat_records,
        layers=layers,
        max_length=max_length,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
        message_key="messages_for_capture",
        desc=desc,
    )
    return average_flat_by_parent(flat_acts, parent_indices, len(records))


def vector_stats(base: Dict[int, torch.Tensor], rl: Dict[int, torch.Tensor], vector: Dict[int, torch.Tensor]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for layer, vec in vector.items():
        deltas = rl[layer] - base[layer]
        norm = float(vec.norm().item())
        if norm > 0 and deltas.numel() > 0:
            cos = F.cosine_similarity(deltas, vec.unsqueeze(0), dim=-1)
            cos_mean = float(cos.mean().item())
            cos_std = float(cos.std(unbiased=False).item())
        else:
            cos_mean = 0.0
            cos_std = 0.0
        stats[str(layer)] = {
            "vector_norm": norm,
            "delta_norm_mean": float(deltas.norm(dim=-1).mean().item()) if deltas.numel() else 0.0,
            "cosine_to_vector_mean": cos_mean,
            "cosine_to_vector_std": cos_std,
        }
    return stats


def select_by_indices(acts: Dict[int, torch.Tensor], indices: Sequence[int]) -> Dict[int, torch.Tensor]:
    index = torch.tensor(list(indices), dtype=torch.long)
    return {layer: tensor.index_select(0, index) for layer, tensor in acts.items()}


def indices_by_challenge(records: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        challenge_type = str(record.get("challenge_type") or "unknown")
        grouped.setdefault(challenge_type, []).append(idx)
    return grouped


def export_direct_vectors(output_dir: Path, payload: Dict[str, Any]) -> Dict[str, str]:
    vector_dir = output_dir / "easysteer_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, str] = {}
    top_level = {
        "belief": "belief_vectors",
        "raw": "raw_divergence_vectors",
        "nuisance": "nuisance_vectors",
    }
    by_type = {
        "belief_by_type": "belief_vectors_by_type",
        "raw_by_type": "raw_divergence_vectors_by_type",
        "nuisance_by_type": "nuisance_vectors_by_type",
    }
    for label, key in top_level.items():
        for layer, vec in payload[key].items():
            path = vector_dir / f"{label}_L{int(layer)}.pt"
            torch.save(vec.detach().float().cpu(), path)
            manifest[f"{label}:L{int(layer)}"] = str(path)
    for label, key in by_type.items():
        for challenge_type, vectors in payload[key].items():
            for layer, vec in vectors.items():
                path = vector_dir / f"{label}_{safe_name(challenge_type)}_L{int(layer)}.pt"
                torch.save(vec.detach().float().cpu(), path)
                manifest[f"{label}:{challenge_type}:L{int(layer)}"] = str(path)
    return manifest


def build_vectors(
    *,
    base_all: Dict[int, torch.Tensor],
    rl_all: Dict[int, torch.Tensor],
    divergence: Sequence[Dict[str, Any]],
    control: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    div_n = len(divergence)
    base_div = {layer: acts[:div_n] for layer, acts in base_all.items()}
    base_ctl = {layer: acts[div_n:] for layer, acts in base_all.items()}
    rl_div = {layer: acts[:div_n] for layer, acts in rl_all.items()}
    rl_ctl = {layer: acts[div_n:] for layer, acts in rl_all.items()}
    div_indices_by_type = indices_by_challenge(divergence)
    ctl_indices_by_type = indices_by_challenge(control)

    belief_vectors: Dict[int, torch.Tensor] = {}
    nuisance_vectors: Dict[int, torch.Tensor] = {}
    raw_vectors: Dict[int, torch.Tensor] = {}
    belief_vectors_by_type: Dict[str, Dict[int, torch.Tensor]] = {}
    nuisance_vectors_by_type: Dict[str, Dict[int, torch.Tensor]] = {}
    raw_vectors_by_type: Dict[str, Dict[int, torch.Tensor]] = {}

    for layer in sorted(base_all):
        raw = (rl_div[layer] - base_div[layer]).mean(dim=0)
        if base_ctl[layer].shape[0] > 0:
            nuisance = (rl_ctl[layer] - base_ctl[layer]).mean(dim=0)
        else:
            nuisance = torch.zeros_like(raw)
        raw_vectors[layer] = raw
        nuisance_vectors[layer] = nuisance
        belief_vectors[layer] = raw - nuisance

    for challenge_type, indices in sorted(div_indices_by_type.items()):
        raw_vectors_by_type[challenge_type] = {}
        nuisance_vectors_by_type[challenge_type] = {}
        belief_vectors_by_type[challenge_type] = {}
        ctl_indices = ctl_indices_by_type.get(challenge_type, [])
        div_index = torch.tensor(indices, dtype=torch.long)
        ctl_index = torch.tensor(ctl_indices, dtype=torch.long)
        for layer in sorted(base_all):
            raw = (rl_div[layer].index_select(0, div_index) - base_div[layer].index_select(0, div_index)).mean(dim=0)
            if ctl_indices:
                nuisance = (rl_ctl[layer].index_select(0, ctl_index) - base_ctl[layer].index_select(0, ctl_index)).mean(dim=0)
            else:
                nuisance = torch.zeros_like(raw)
            raw_vectors_by_type[challenge_type][layer] = raw
            nuisance_vectors_by_type[challenge_type][layer] = nuisance
            belief_vectors_by_type[challenge_type][layer] = raw - nuisance

    return {
        "base_divergence": base_div,
        "rl_divergence": rl_div,
        "base_control": base_ctl,
        "rl_control": rl_ctl,
        "belief_vectors": belief_vectors,
        "raw_divergence_vectors": raw_vectors,
        "nuisance_vectors": nuisance_vectors,
        "belief_vectors_by_type": belief_vectors_by_type,
        "raw_divergence_vectors_by_type": raw_vectors_by_type,
        "nuisance_vectors_by_type": nuisance_vectors_by_type,
        "div_indices_by_type": div_indices_by_type,
        "ctl_indices_by_type": ctl_indices_by_type,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--rl-model", default=DEFAULT_RL_MODEL_520)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=121400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", choices=["true", "false", "auto"], default="true")
    parser.add_argument("--context-mode", choices=["canonical", "trajectory"], default="trajectory")
    parser.add_argument(
        "--turn-average",
        action="store_true",
        default=False,
        help="Average Base/RL deltas across every stored turn within each record before averaging records.",
    )
    parser.add_argument("--save-activations", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    dataset = read_json(args.input)
    divergence = flatten_records(dataset, "extract_divergence")
    control = flatten_records(dataset, "control")
    all_records = divergence + control
    if not divergence:
        raise ValueError("extract_divergence is empty; cannot compute Base-vs-RL vector")

    layers = parse_ints(args.layers)
    enable_thinking = parse_enable_thinking(args.enable_thinking)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_message_key = "messages_common" if args.context_mode == "canonical" else "messages_base_context"
    rl_message_key = "messages_common" if args.context_mode == "canonical" else "messages_rl_context"
    base_turn_message_key = "messages_common_by_turn" if args.context_mode == "canonical" else "messages_base_by_turn"
    rl_turn_message_key = "messages_common_by_turn" if args.context_mode == "canonical" else "messages_rl_by_turn"
    print(
        f"[extract:easysteer] records: divergence={len(divergence)} control={len(control)} "
        f"layers={layers} backend=vllm_capture turn_average={args.turn_average}",
        flush=True,
    )

    base_llm = make_llm(args.base_model, args)
    if args.turn_average:
        if control:
            raise ValueError("--turn-average currently expects --no-control failed_isolation datasets")
        base_all = collect_turn_average_activations(
            llm=base_llm,
            records=divergence,
            layers=layers,
            max_length=args.max_length,
            batch_size=args.batch_size,
            enable_thinking=enable_thinking,
            message_sequence_key=base_turn_message_key,
            desc="base turn capture",
        )
    else:
        base_all = collect_activations(
            llm=base_llm,
            records=all_records,
            layers=layers,
            max_length=args.max_length,
            batch_size=args.batch_size,
            enable_thinking=enable_thinking,
            message_key=base_message_key,
            desc="base capture",
        )
    del base_llm
    clear_cuda()

    rl_llm = make_llm(args.rl_model, args)
    if args.turn_average:
        rl_all = collect_turn_average_activations(
            llm=rl_llm,
            records=divergence,
            layers=layers,
            max_length=args.max_length,
            batch_size=args.batch_size,
            enable_thinking=enable_thinking,
            message_sequence_key=rl_turn_message_key,
            desc="rl turn capture",
        )
        all_records = list(divergence)
    else:
        rl_all = collect_activations(
            llm=rl_llm,
            records=all_records,
            layers=layers,
            max_length=args.max_length,
            batch_size=args.batch_size,
            enable_thinking=enable_thinking,
            message_key=rl_message_key,
            desc="rl capture",
        )
    del rl_llm
    clear_cuda()

    built = build_vectors(base_all=base_all, rl_all=rl_all, divergence=divergence, control=control)
    vector_payload = {
        "layers": sorted(built["belief_vectors"]),
        "belief_vectors": built["belief_vectors"],
        "raw_divergence_vectors": built["raw_divergence_vectors"],
        "nuisance_vectors": built["nuisance_vectors"],
        "belief_vectors_by_type": built["belief_vectors_by_type"],
        "raw_divergence_vectors_by_type": built["raw_divergence_vectors_by_type"],
        "nuisance_vectors_by_type": built["nuisance_vectors_by_type"],
        "config": {
            "backend": "easysteer_vllm",
            "base_model": args.base_model,
            "rl_model": args.rl_model,
            "layers": layers,
            "input": str(args.input),
            "enable_thinking": args.enable_thinking,
            "context_mode": args.context_mode,
            "capture_point": "decoder layer complete hidden state at prefix last token",
            "capture_method": "EasySteer hidden_states capture via vLLM generate(max_tokens=1)",
            "turn_average": bool(args.turn_average),
            "vector_formula": {
                "raw_divergence": "mean(H_RL-H_Base | divergence)",
                "raw_divergence_turn_average": "mean_over_records(mean_over_turns(H_RL_turn-H_Base_turn))",
                "raw_divergence_by_type": "mean(H_RL-H_Base | divergence and challenge_type)",
                "nuisance": "mean(H_RL-H_Base | control)",
                "nuisance_by_type": "mean(H_RL-H_Base | control and challenge_type)",
                "belief": "raw_divergence - nuisance",
                "belief_by_type": "raw_divergence_by_type - nuisance_by_type",
                "recommended_for_steering": "raw_divergence_by_type for failed_stay/failed_update-specific runs",
            },
        },
    }
    direct_manifest = export_direct_vectors(args.output_dir, vector_payload)
    vector_payload["easysteer_direct_pt"] = direct_manifest
    vector_path = args.output_dir / "belief_vectors_easysteer.pt"
    torch.save(vector_payload, vector_path)

    if args.save_activations:
        torch.save(
            {
                "base_divergence": built["base_divergence"],
                "rl_divergence": built["rl_divergence"],
                "base_control": built["base_control"],
                "rl_control": built["rl_control"],
            },
            args.output_dir / "belief_vector_activations_easysteer.pt",
        )

    base_div = built["base_divergence"]
    rl_div = built["rl_divergence"]
    summary = {
        "input": str(args.input),
        "vector_path": str(vector_path),
        "backend": "easysteer_vllm",
        "capture_point": "prefix last-token complete hidden state",
        "easysteer_direct_pt": direct_manifest,
        "counts": {
            "extract_divergence": len(divergence),
            "control": len(control),
            "extract_divergence_by_type": dict(Counter(str(item.get("challenge_type") or "unknown") for item in divergence)),
            "control_by_type": dict(Counter(str(item.get("challenge_type") or "unknown") for item in control)),
        },
        "layers": list(vector_payload["layers"]),
        "belief_vector_stats": vector_stats(base_div, rl_div, built["belief_vectors"]),
        "raw_divergence_vector_stats": vector_stats(base_div, rl_div, built["raw_divergence_vectors"]),
        "nuisance_vector_norms": {
            str(layer): float(vec.norm().item()) for layer, vec in built["nuisance_vectors"].items()
        },
        "by_type": {},
    }
    for challenge_type, indices in sorted(built["div_indices_by_type"].items()):
        div_subset_base = select_by_indices(base_div, indices)
        div_subset_rl = select_by_indices(rl_div, indices)
        summary["by_type"][challenge_type] = {
            "extract_divergence": len(indices),
            "control": len(built["ctl_indices_by_type"].get(challenge_type, [])),
            "belief_vector_stats": vector_stats(div_subset_base, div_subset_rl, built["belief_vectors_by_type"][challenge_type]),
            "raw_divergence_vector_stats": vector_stats(div_subset_base, div_subset_rl, built["raw_divergence_vectors_by_type"][challenge_type]),
            "nuisance_vector_norms": {
                str(layer): float(vec.norm().item()) for layer, vec in built["nuisance_vectors_by_type"][challenge_type].items()
            },
        }
    write_json(args.output_dir / "belief_vectors_summary_easysteer.json", summary)
    print(f"[extract:easysteer] wrote {vector_path}")
    print(f"[extract:easysteer] wrote {args.output_dir / 'belief_vectors_summary_easysteer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
