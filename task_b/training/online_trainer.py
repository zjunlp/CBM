"""Dataset-driven multi-turn GRPO trainer for Scenario B (circuit fault diagnosis)."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate.utils import broadcast_object_list, gather_object
from trl import GRPOTrainer
from trl.extras.profiling import profiling_context

from task_b.runtime.orchestrator import parse_hypotheses


@dataclass
class OnlineConfig:
    cot: bool = False
    t01_temperature: float = 0.3
    t01_top_p: float = 0.9
    t01_top_k: Optional[int] = None
    t01_presence_penalty: Optional[float] = None
    t01_repetition_penalty: Optional[float] = None
    t01_max_tokens: int = 256
    t2_temperature: float = 0.7
    t2_top_p: float = 0.9
    t2_top_k: Optional[int] = None
    t2_presence_penalty: Optional[float] = None
    t2_repetition_penalty: Optional[float] = None
    t2_max_tokens: int = 256


def _set_if_present(obj: Any, attr: str, value: Any) -> Any:
    if hasattr(obj, attr):
        old_value = getattr(obj, attr)
        if value is not None:
            setattr(obj, attr, value)
        return old_value
    return None


def _encode_text(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _encode_text_with_offsets(tokenizer, text: str):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return encoded["input_ids"], encoded["offset_mapping"]


def _build_case_messages(row: Dict[str, Any], t0_text: Optional[str] = None, t1_text: Optional[str] = None):
    messages_t0 = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["turn_prompts"][0]},
    ]
    messages_t1 = messages_t0 + [
        {"role": "assistant", "content": t0_text if t0_text is not None else ""},
        {"role": "user", "content": row["turn_prompts"][1]},
    ]
    messages_t2 = messages_t1 + [
        {"role": "assistant", "content": t1_text if t1_text is not None else ""},
        {"role": "user", "content": row["turn_prompts"][2]},
    ]
    return messages_t0, messages_t1, messages_t2


def _extract_sampled_logprobs(logprobs: List[List[List[float]]]) -> List[List[float]]:
    return [[step[0] for step in seq] for seq in logprobs]


def _suffix_after_prefix(full_ids: List[int], prefix_ids: List[int], label: str) -> List[int]:
    if len(full_ids) < len(prefix_ids):
        raise RuntimeError(f"{label}: full token sequence shorter than prefix.")
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError(f"{label}: token prefix mismatch while stitching multi-turn rollout.")
    return full_ids[len(prefix_ids):]


def _token_index_before_char(offsets: List[tuple[int, int]], char_pos: int) -> int:
    split_idx = 0
    for idx, (start, end) in enumerate(offsets):
        if end <= char_pos:
            split_idx = idx + 1
        elif start < char_pos < end:
            split_idx = idx
            break
        elif start >= char_pos:
            break
    return split_idx


def _build_span_mask(
    offsets: List[tuple[int, int]],
    spans: List[tuple[int, int]],
) -> List[int]:
    mask = [0] * len(offsets)
    for idx, (start, end) in enumerate(offsets):
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                mask[idx] = 1
                break
    return mask


def _span_token_indices(
    offsets: List[tuple[int, int]],
    span: tuple[int, int],
) -> List[int]:
    span_start, span_end = span
    indices: List[int] = []
    for idx, (start, end) in enumerate(offsets):
        if start < span_end and end > span_start:
            indices.append(idx)
    return indices


def _trim_trailing_special_logprob_mismatch(
    *,
    indices: List[int],
    turn_ids: List[int],
    turn_logps: List[float],
    tokenizer,
    label: str,
) -> List[float]:
    if len(indices) == len(turn_logps):
        return turn_logps

    if len(turn_logps) == len(indices) + 1 and turn_ids:
        last_token_id = turn_ids[-1]
        if last_token_id in set(tokenizer.all_special_ids):
            return turn_logps[:len(indices)]
        if tokenizer.decode([last_token_id], skip_special_tokens=True) == "":
            return turn_logps[:len(indices)]

    raise RuntimeError(
        f"{label}: token/logprob length mismatch after conversation retokenization: "
        f"{len(indices)} tokens vs {len(turn_logps)} logprobs."
    )


def _is_exact_match(response: str, gt_survivors: List[str]) -> bool:
    hypotheses = parse_hypotheses(response)
    if hypotheses is None:
        return False
    return set(hypotheses) == set(gt_survivors)


def _pad_1d_tensors(
    tensors: List[torch.Tensor],
    padding_value: float | int,
    padding_side: str,
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad an empty tensor list.")
    max_len = max(t.size(0) for t in tensors)
    if pad_to_multiple_of is not None and max_len % pad_to_multiple_of != 0:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    padded = []
    for tensor in tensors:
        pad_len = max_len - tensor.size(0)
        if pad_len == 0:
            padded.append(tensor)
            continue
        pad_tensor = torch.full(
            (pad_len,),
            padding_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if padding_side == "left":
            padded.append(torch.cat([pad_tensor, tensor], dim=0))
        elif padding_side == "right":
            padded.append(torch.cat([tensor, pad_tensor], dim=0))
        else:
            raise ValueError(f"Unsupported padding_side: {padding_side}")
    return torch.stack(padded, dim=0)


def _nanstd(x: torch.Tensor) -> torch.Tensor:
    valid = x[~torch.isnan(x)]
    if valid.numel() <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return valid.std()


def _build_rollout_batch_from_cases(
    rows: List[Dict[str, Any]],
    tokenizer,
    vllm_generation,
    oc: OnlineConfig,
) -> Tuple[
    List[List[int]],
    List[List[int]],
    List[List[int]],
    List[List[float]],
    Dict[str, List[Any]],
    Dict[str, int],
]:
    prompt_ids_list: List[List[int]] = []
    completion_ids_list: List[List[int]] = []
    env_mask_list: List[List[int]] = []
    sampling_logprobs_list: List[List[float]] = []
    extra_fields: Dict[str, List[Any]] = {
        "turn_responses": [],
        "turn_gts": [],
        "prefix_valid": [],
        "reward_weight": [],
        "challenge_type": [],
        "oracle": [],
        "gt_survivors": [],
    }
    summary = {
        "n_rollouts": len(rows),
        "n_unique": len(rows),
        "prefix_valid_count": 0,
    }

    for row in rows:
        messages_t0, messages_t1, messages_t2 = _build_case_messages(row)
        full_t0 = tokenizer.apply_chat_template(messages_t0, tokenize=False, add_generation_prompt=True)
        full_t1 = tokenizer.apply_chat_template(messages_t1, tokenize=False, add_generation_prompt=True)
        full_t2 = tokenizer.apply_chat_template(messages_t2, tokenize=False, add_generation_prompt=True)

        # Roll out T0/T1/T2 inline so we can keep the final token stream consistent.
        old_temperature = getattr(vllm_generation, "temperature", None)
        old_top_p = getattr(vllm_generation, "top_p", None)
        old_top_k = getattr(vllm_generation, "top_k", None)
        old_presence_penalty = getattr(vllm_generation, "presence_penalty", None)
        old_repetition_penalty = getattr(vllm_generation, "repetition_penalty", None)
        old_max_completion_length = getattr(vllm_generation, "max_completion_length", None)
        try:
            t0_prompt_ids, t0_prompt_offsets = _encode_text_with_offsets(tokenizer, full_t0)
            _set_if_present(vllm_generation, "temperature", oc.t01_temperature)
            _set_if_present(vllm_generation, "top_p", oc.t01_top_p)
            _set_if_present(vllm_generation, "top_k", oc.t01_top_k)
            _set_if_present(vllm_generation, "presence_penalty", oc.t01_presence_penalty)
            _set_if_present(vllm_generation, "repetition_penalty", oc.t01_repetition_penalty)
            _set_if_present(vllm_generation, "max_completion_length", oc.t01_max_tokens)
            _, t0_ids_list, t0_logprobs, _ = vllm_generation.generate(
                prompts=[t0_prompt_ids],
                images=None,
                num_generations=1,
                profiler=None,
            )
            t0_ids = t0_ids_list[0]
            t0_text = tokenizer.decode(t0_ids, skip_special_tokens=True)
            t0_logps = _extract_sampled_logprobs(t0_logprobs)[0]

            full_t1 = tokenizer.apply_chat_template(
                _build_case_messages(row, t0_text=t0_text)[1],
                tokenize=False,
                add_generation_prompt=True,
            )
            _set_if_present(vllm_generation, "temperature", oc.t01_temperature)
            _set_if_present(vllm_generation, "top_p", oc.t01_top_p)
            _set_if_present(vllm_generation, "top_k", oc.t01_top_k)
            _set_if_present(vllm_generation, "presence_penalty", oc.t01_presence_penalty)
            _set_if_present(vllm_generation, "repetition_penalty", oc.t01_repetition_penalty)
            _set_if_present(vllm_generation, "max_completion_length", oc.t01_max_tokens)
            t1_prompt_ids, _ = _encode_text_with_offsets(tokenizer, full_t1)
            _, t1_ids_list, t1_logprobs, _ = vllm_generation.generate(
                prompts=[t1_prompt_ids],
                images=None,
                num_generations=1,
                profiler=None,
            )
            t1_ids = t1_ids_list[0]
            t1_text = tokenizer.decode(t1_ids, skip_special_tokens=True)
            t1_logps = _extract_sampled_logprobs(t1_logprobs)[0]

            full_t2 = tokenizer.apply_chat_template(
                _build_case_messages(row, t0_text=t0_text, t1_text=t1_text)[2],
                tokenize=False,
                add_generation_prompt=True,
            )
            _set_if_present(vllm_generation, "temperature", oc.t2_temperature)
            _set_if_present(vllm_generation, "top_p", oc.t2_top_p)
            _set_if_present(vllm_generation, "top_k", oc.t2_top_k)
            _set_if_present(vllm_generation, "presence_penalty", oc.t2_presence_penalty)
            _set_if_present(vllm_generation, "repetition_penalty", oc.t2_repetition_penalty)
            _set_if_present(vllm_generation, "max_completion_length", oc.t2_max_tokens)
            t2_prompt_ids, _ = _encode_text_with_offsets(tokenizer, full_t2)
            _, t2_ids_list, t2_logprobs, _ = vllm_generation.generate(
                prompts=[t2_prompt_ids],
                images=None,
                num_generations=1,
                profiler=None,
            )
            t2_ids = t2_ids_list[0]
            t2_text = tokenizer.decode(t2_ids, skip_special_tokens=True)
            t2_logps = _extract_sampled_logprobs(t2_logprobs)[0]
        finally:
            if hasattr(vllm_generation, "temperature"):
                vllm_generation.temperature = old_temperature
            if hasattr(vllm_generation, "top_p"):
                vllm_generation.top_p = old_top_p
            if hasattr(vllm_generation, "top_k"):
                vllm_generation.top_k = old_top_k
            if hasattr(vllm_generation, "presence_penalty"):
                vllm_generation.presence_penalty = old_presence_penalty
            if hasattr(vllm_generation, "repetition_penalty"):
                vllm_generation.repetition_penalty = old_repetition_penalty
            if hasattr(vllm_generation, "max_completion_length"):
                vllm_generation.max_completion_length = old_max_completion_length

        gt_t0, gt_t1, gt_t2 = [sorted(x) for x in row["turn_gts"]]
        prefix_valid = _is_exact_match(t0_text, gt_t0) and _is_exact_match(t1_text, gt_t1)
        if prefix_valid:
            summary["prefix_valid_count"] += 1

        final_messages = _build_case_messages(row, t0_text=t0_text, t1_text=t1_text)[2] + [
            {"role": "assistant", "content": t2_text}
        ]
        final_text = tokenizer.apply_chat_template(
            final_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids, full_offsets = _encode_text_with_offsets(tokenizer, final_text)
        prompt_end_char = len(full_t0)
        prompt_split = _token_index_before_char(full_offsets, prompt_end_char)
        prompt_ids = full_ids[:prompt_split]
        completion_ids = full_ids[prompt_split:]

        t0_start = final_text.find(t0_text, prompt_end_char)
        if t0_start < 0:
            raise RuntimeError("Unable to locate T0 response inside rendered conversation.")
        t1_prompt_start = final_text.find(row["turn_prompts"][1], t0_start + len(t0_text))
        if t1_prompt_start < 0:
            raise RuntimeError("Unable to locate T1 evidence inside rendered conversation.")
        t1_start = final_text.find(t1_text, t1_prompt_start + len(row["turn_prompts"][1]))
        if t1_start < 0:
            raise RuntimeError("Unable to locate T1 response inside rendered conversation.")
        t2_prompt_start = final_text.find(row["turn_prompts"][2], t1_start + len(t1_text))
        if t2_prompt_start < 0:
            raise RuntimeError("Unable to locate T2 evidence inside rendered conversation.")
        t2_start = final_text.find(t2_text, t2_prompt_start + len(row["turn_prompts"][2]))
        if t2_start < 0:
            raise RuntimeError("Unable to locate T2 response inside rendered conversation.")

        completion_offsets = full_offsets[prompt_split:]
        env_mask = _build_span_mask(
            completion_offsets,
            [
                (t0_start, t0_start + len(t0_text)),
                (t1_start, t1_start + len(t1_text)),
                (t2_start, t2_start + len(t2_text)),
            ],
        )
        sampling_logprobs = [0.0] * len(completion_ids)
        for span, turn_ids, turn_logps, label in [
            ((t0_start, t0_start + len(t0_text)), t0_ids, t0_logps, "t0"),
            ((t1_start, t1_start + len(t1_text)), t1_ids, t1_logps, "t1"),
            ((t2_start, t2_start + len(t2_text)), t2_ids, t2_logps, "t2"),
        ]:
            indices = _span_token_indices(completion_offsets, span)
            turn_logps = _trim_trailing_special_logprob_mismatch(
                indices=indices,
                turn_ids=turn_ids,
                turn_logps=turn_logps,
                tokenizer=tokenizer,
                label=label,
            )
            for idx, logp in zip(indices, turn_logps, strict=True):
                sampling_logprobs[idx] = logp

        prompt_ids_list.append(prompt_ids)
        completion_ids_list.append(completion_ids)
        env_mask_list.append(env_mask)
        sampling_logprobs_list.append(sampling_logprobs)
        extra_fields["turn_responses"].append([t0_text, t1_text, t2_text])
        extra_fields["turn_gts"].append([gt_t0, gt_t1, gt_t2])
        extra_fields["prefix_valid"].append(prefix_valid)
        extra_fields["reward_weight"].append(1.0 if prefix_valid else 0.0)
        extra_fields["challenge_type"].append(row.get("challenge_type", ""))
        extra_fields["oracle"].append(row.get("oracle", ""))
        extra_fields["gt_survivors"].append(json.dumps(gt_t2))

    return (
        prompt_ids_list,
        completion_ids_list,
        env_mask_list,
        sampling_logprobs_list,
        extra_fields,
        summary,
    )


class DynamicOnlineGRPOTrainer(GRPOTrainer):
    """GRPOTrainer subclass that rolls out fixed multi-turn cases from the dataset."""

    def __init__(
        self,
        *args,
        online_config: Optional[OnlineConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.online_config = online_config or OnlineConfig()
        self._online_step_count = 0
        self._last_online_summary: Optional[Dict[str, int]] = None
        if not self.use_vllm:
            raise ValueError(
                "DynamicOnlineGRPOTrainer requires use_vllm=True."
            )

    def _sync_vllm_weights(self) -> None:
        """Sync trainer weights to vLLM across TRL versions."""
        if self.state.global_step == self._last_loaded_step:
            return

        if hasattr(self, "_move_model_to_vllm"):
            self._move_model_to_vllm()
        elif hasattr(self, "vllm_generation"):
            with profiling_context(self, "sync_weights"):
                self.vllm_generation.sync_weights()
        else:
            raise AttributeError(
                "Unable to sync weights to vLLM: neither `_move_model_to_vllm` "
                "nor `vllm_generation.sync_weights()` is available."
            )
        self._last_loaded_step = self.state.global_step

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        required = {"system_prompt", "turn_prompts", "turn_gts"}
        missing = required - set(inputs[0].keys())
        if missing:
            raise ValueError(
                f"Dataset-driven multi-turn training requires columns {sorted(required)}; missing {sorted(missing)}."
            )

        prompts = [x["prompt"] for x in inputs]
        self._sync_vllm_weights()
        if self.accelerator.is_main_process:
            payload = _build_rollout_batch_from_cases(
                rows=inputs,
                tokenizer=self.processing_class,
                vllm_generation=self.vllm_generation,
                oc=self.online_config,
            )
        else:
            payload = (None, None, None, None, None, None)
        object_payload = broadcast_object_list(list(payload), from_process=0)
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            sampling_per_token_logps_list,
            extra_fields,
            summary,
        ) = object_payload
        self._last_online_summary = summary
        completions = self.processing_class.batch_decode(
            completion_ids_list,
            skip_special_tokens=True,
        )
        completion_lengths = torch.tensor(
            [sum(mask) for mask in tool_mask_list],
            device=self.accelerator.device,
        )
        num_items_in_batch = self.accelerator.gather(completion_lengths).sum()

        prompt_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in prompt_ids_list]
        prompt_mask_tensors = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids_tensors]
        completion_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in completion_ids_list]
        completion_mask_tensors = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids_tensors]

        prompt_ids = _pad_1d_tensors(
            prompt_ids_tensors,
            padding_value=self.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = _pad_1d_tensors(
            prompt_mask_tensors,
            padding_value=0,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_ids = _pad_1d_tensors(
            completion_ids_tensors,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = _pad_1d_tensors(
            completion_mask_tensors,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)

        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = _pad_1d_tensors(
                [torch.tensor(v, dtype=torch.float32) for v in sampling_per_token_logps_list],
                padding_value=0.0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device=device)
        else:
            sampling_per_token_logps = None

        if tool_mask_list is not None:
            tool_mask = _pad_1d_tensors(
                [torch.tensor(v, dtype=torch.long) for v in tool_mask_list],
                padding_value=1,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device=device)
        else:
            tool_mask = None

        if getattr(self, "mask_truncated_completions", False):
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and getattr(self, "vllm_importance_sampling_correction", False)
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None

            if (
                self.use_vllm
                and getattr(self, "vllm_importance_sampling_correction", False)
                and sampling_per_token_logps is not None
            ):
                mask = completion_mask if tool_mask is None else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                else:
                    logps_diff = per_token_logps_diff
                vllm_importance_sampling_ratio = torch.exp(logps_diff)
                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio,
                        max=self.vllm_importance_sampling_cap,
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap,
                        value=0.0,
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling mode: {self.vllm_importance_sampling_mode}"
                    )
            else:
                vllm_importance_sampling_ratio = None
                sequence_level_is = False

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                    )
                else:
                    model = self.accelerator.unwrap_model(self.model)
                    with model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                        )
            else:
                ref_per_token_logps = None

        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        if self.multi_objective_aggregation == "sum_then_normalize":
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            else:
                raise ValueError(f"Invalid value for scale_rewards: {self.scale_rewards}")
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            if num_generations > 1:
                flat_std = []
                for i in range(grouped.size(2)):
                    per_group = []
                    for j in range(grouped.size(0)):
                        per_group.append(_nanstd(grouped[j, :, i]))
                    flat_std.append(torch.stack(per_group))
                std_k = torch.stack(flat_std, dim=1).unsqueeze(1)
            else:
                std_k = torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        else:
            raise ValueError(f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}")

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        grouped_rewards = rewards.view(-1, num_generations)
        group_reward_std = grouped_rewards.std(dim=1) if num_generations > 1 else torch.zeros(
            grouped_rewards.size(0), device=rewards.device, dtype=rewards.dtype
        )

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = _nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item() if rewards.numel() > 1 else 0.0)
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        self._metrics[mode]["rewards/group_std_mean"].append(group_reward_std.mean().item())
        self._metrics[mode]["advantages/abs_mean"].append(all_process_advantages.abs().mean().item())
        self._metrics[mode]["advantages/nonzero_frac"].append(
            (all_process_advantages.abs() > 1e-8).float().mean().item()
        )

        if hasattr(self, "_logs"):
            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))
            for i, name in enumerate(self.reward_func_names):
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._logs["advantages"].extend(all_process_advantages.tolist())

        if (
            self.use_vllm
            and getattr(self, "vllm_importance_sampling_correction", False)
            and sampling_per_token_logps is not None
        ):
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if vllm_importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if tool_mask is not None:
            output["tool_mask"] = tool_mask

        output = output
        self._online_step_count += 1
        if (
            self.accelerator.is_main_process
            and self._last_online_summary is not None
            and (self._online_step_count <= 3 or self._online_step_count % 20 == 0)
        ):
            summary = self._last_online_summary
            print(
                f"[v7 online step {self._online_step_count}] "
                f"n_rollouts={summary['n_rollouts']} "
                f"n_unique={summary['n_unique']} "
                f"prefix_valid={summary['prefix_valid_count']}/{summary['n_rollouts']}",
                flush=True,
            )
        return output
