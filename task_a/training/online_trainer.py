"""Dataset-driven multi-turn GRPO trainer."""

import json
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate.utils import broadcast_object_list, gather, gather_object
from swift.infer_engine import RequestConfig
from swift.infer_engine.protocol import RolloutInferRequest
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from swift.rlhf_trainers import GRPOTrainer
from swift.rlhf_trainers.utils import profiling_context
from swift.utils import unwrap_model_for_generation
from transformers import LogitsProcessor, LogitsProcessorList

from utils.hypotheses_parser import parse_hypotheses


@dataclass
class OnlineConfig:
    cot: bool = False
    t01_temperature: float = 0.3
    t01_top_p: float = 0.9
    t01_top_k: Optional[int] = None
    t01_min_p: Optional[float] = None
    t01_presence_penalty: Optional[float] = None
    t01_repetition_penalty: Optional[float] = None
    t01_max_tokens: int = 256
    t2_temperature: float = 0.7
    t2_top_p: float = 0.9
    t2_top_k: Optional[int] = None
    t2_min_p: Optional[float] = None
    t2_presence_penalty: Optional[float] = None
    t2_repetition_penalty: Optional[float] = None
    t2_max_tokens: int = 256
    use_vllm_fast_inference: bool = False


class _PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Apply an OpenAI/vLLM-style once-per-seen-token presence penalty."""

    def __init__(self, penalty: float, excluded_token_ids: Optional[List[int]] = None):
        self.penalty = float(penalty)
        self.excluded_token_ids = set(excluded_token_ids or [])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        scores = scores.clone()
        for batch_idx in range(input_ids.size(0)):
            seen = torch.unique(input_ids[batch_idx])
            if self.excluded_token_ids:
                keep = torch.ones_like(seen, dtype=torch.bool)
                for token_id in self.excluded_token_ids:
                    keep &= seen != token_id
                seen = seen[keep]
            if seen.numel() > 0:
                scores[batch_idx, seen] -= self.penalty
        return scores


def _encode_text(tokenizer, text: str) -> List[int]:
    input_ids = tokenizer(text=text, add_special_tokens=False)["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def _build_case_messages_for_turn(
    row: Dict[str, Any],
    prior_responses: List[str],
    turn_idx: int,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["turn_prompts"][0]},
    ]
    for idx, response in enumerate(prior_responses[:turn_idx]):
        messages.extend(
            [
                {"role": "assistant", "content": response},
                {"role": "user", "content": row["turn_prompts"][idx + 1]},
            ]
        )
    return messages


def _build_final_case_messages(row: Dict[str, Any], responses: List[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["turn_prompts"][0]},
    ]
    for idx, response in enumerate(responses):
        messages.append({"role": "assistant", "content": response})
        if idx + 1 < len(row["turn_prompts"]):
            messages.append({"role": "user", "content": row["turn_prompts"][idx + 1]})
    return messages


def _build_sampled_case_messages(row: Dict[str, Any], responses: List[str]) -> List[Dict[str, str]]:
    """Build a transcript that ends at the last sampled assistant response."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["turn_prompts"][0]},
    ]
    for idx, response in enumerate(responses):
        messages.append({"role": "assistant", "content": response})
        if idx + 1 < len(responses):
            messages.append({"role": "user", "content": row["turn_prompts"][idx + 1]})
    return messages


def _chat_template_ids(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> List[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return _encode_text(tokenizer, text)


def _build_response_token_spans(
    row: Dict[str, Any],
    responses: List[str],
    tokenizer,
) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    for idx in range(len(responses)):
        before_ids = _chat_template_ids(
            tokenizer,
            _build_case_messages_for_turn(row, responses[:idx], idx),
            add_generation_prompt=True,
        )
        after_ids = _chat_template_ids(
            tokenizer,
            _build_sampled_case_messages(row, responses[: idx + 1]),
            add_generation_prompt=False,
        )
        spans.append((len(before_ids), len(after_ids)))
    return spans


def _build_token_span_mask(
    length: int,
    spans: List[tuple[int, int]],
    offset: int,
) -> List[int]:
    mask = [0] * length
    for abs_start, abs_end in spans:
        start = max(abs_start - offset, 0)
        end = min(abs_end - offset, length)
        for idx in range(start, end):
            mask[idx] = 1
    return mask


def _relative_token_indices(
    length: int,
    span: tuple[int, int],
    offset: int,
) -> List[int]:
    start = max(span[0] - offset, 0)
    end = min(span[1] - offset, length)
    if end <= start:
        return []
    return list(range(start, end))


def _mode_from_row(row: Dict[str, Any]) -> str:
    challenge_type = str(row.get("challenge_type", "")).lower()
    if "failed_stay" in challenge_type:
        return "failed_stay"
    if "failed_update" in challenge_type or "misrecord" in challenge_type or "correction" in challenge_type:
        return "failed_update"
    return challenge_type


def _prefix_turn_count(row: Dict[str, Any]) -> int:
    n_turns = len(row["turn_prompts"])
    mode = _mode_from_row(row)
    if mode == "failed_update":
        return min(3, n_turns)
    if mode == "failed_stay":
        return min(2, n_turns)
    return max(n_turns - 1, 0)


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
        tokenizer = self.processing_class
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.pad_token_id is None and hasattr(tokenizer, "tokenizer"):
            self.pad_token_id = getattr(tokenizer.tokenizer, "pad_token_id", None)
        if self.eos_token_id is None and hasattr(tokenizer, "tokenizer"):
            self.eos_token_id = getattr(tokenizer.tokenizer, "eos_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        self.pad_to_multiple_of = getattr(self.args, "pad_to_multiple_of", None)
        self._online_max_prompt_length = getattr(self.args, "max_prompt_length", None)

    def _sync_vllm_weights_for_online_rollout(self) -> None:
        if not getattr(self, "use_fast_infer", False):
            raise RuntimeError("Online vLLM rollout is enabled, but Swift GRPOConfig.use_vllm is false.")
        if getattr(self, "vllm_mode", None) != "server":
            raise RuntimeError("The custom dataset online rollout requires Swift vLLM server mode.")

        args = self.args
        sleep_level = getattr(args, "sleep_level", 0)
        if self.state.global_step != getattr(self, "_last_loaded_step", -1) or sleep_level == 2:
            with profiling_context(self, "sync_vllm_weights"):
                self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

    def _build_vllm_request_config(
        self,
        sampling_config: Dict[str, Any],
        presence_penalty: Optional[float],
    ) -> RequestConfig:
        return RequestConfig(
            n=1,
            max_tokens=sampling_config["max_new_tokens"],
            temperature=sampling_config["temperature"],
            top_p=sampling_config["top_p"],
            top_k=sampling_config.get("top_k"),
            repetition_penalty=sampling_config.get("repetition_penalty"),
            presence_penalty=presence_penalty or 0.0,
            return_details=True,
            logprobs=False,
        )

    def _generate_turns_vllm_server(
        self,
        prompt_messages: List[List[Dict[str, str]]],
        sampling_config: Dict[str, Any],
        presence_penalty: Optional[float],
    ) -> List[List[int]]:
        """Generate one turn with a dedicated Swift rollout server."""
        request_config = self._build_vllm_request_config(sampling_config, presence_penalty)
        with profiling_context(self, "vllm_server.generate"), torch.no_grad():
            self._sync_vllm_weights_for_online_rollout()
            local_requests = [
                RolloutInferRequest(messages=messages, uuid=f"chatcmpl-{uuid.uuid4().hex}")
                for messages in prompt_messages
            ]
            all_requests = gather_object(local_requests)
            all_lengths = gather_object([len(local_requests)])

            if self.accelerator.is_main_process and all_requests:
                all_outputs = self.vllm_client.infer(
                    all_requests,
                    request_config,
                    use_tqdm=False,
                )
            else:
                all_outputs = [None] * len(all_requests)

            all_outputs = broadcast_object_list(all_outputs, from_process=0)
            start = sum(all_lengths[: self.accelerator.process_index])
            end = start + all_lengths[self.accelerator.process_index]
            outputs = all_outputs[start:end]

        generated_ids = []
        for output in outputs:
            response = getattr(output, "response", output)
            choice = response.choices[0]
            generated_ids.append(list(choice.token_ids or []))
        return generated_ids

    def _generate_turns(
        self,
        prompt_texts: List[str],
        prompt_messages: List[List[Dict[str, str]]],
        sampling_config: Dict[str, Any],
        presence_penalty: Optional[float],
    ) -> List[List[int]]:
        if self.online_config.use_vllm_fast_inference:
            return self._generate_turns_vllm_server(prompt_messages, sampling_config, presence_penalty)
        return self._generate_turns_local(prompt_texts, sampling_config, presence_penalty)

    def _logprob_inputs(
        self,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> Dict[str, torch.Tensor | int]:
        return {
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "logits_to_keep": logits_to_keep,
        }

    def _score_online_rewards(self, inputs: List[Dict[str, Any]], completions: List[str]) -> torch.Tensor:
        device = self.accelerator.device
        local_rewards = torch.zeros((len(inputs), len(self.reward_funcs)), device=device)
        reward_kwargs = {"trainer_state": self.state}
        batched_inputs: Dict[str, List[Any]] = {}
        for row in inputs:
            for key, value in row.items():
                if key == "add_eos":
                    continue
                batched_inputs.setdefault(key, []).append(value)
        reward_kwargs.update(batched_inputs)

        for idx, (reward_func, reward_func_name) in enumerate(zip(self.reward_funcs, self.reward_func_names, strict=True)):
            with profiling_context(self, reward_func_name):
                output = reward_func(completions, **reward_kwargs)
            output = [reward if reward is not None else torch.nan for reward in output]
            local_rewards[:, idx] = torch.tensor(output, dtype=torch.float32, device=device)
        return gather(local_rewards)

    def _generate_turns_local(
        self,
        prompt_texts: List[str],
        sampling_config: Dict[str, Any],
        presence_penalty: Optional[float],
    ) -> List[List[int]]:
        """Generate one turn for a batch of prompts with the active online sampling config."""
        tokenizer = self.processing_class
        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=getattr(self.args, "ds3_gather_for_generation", True),
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False)
            if getattr(self, "is_fsdp_enabled", False)
            else nullcontext(),
        ):
            old_padding_side = getattr(tokenizer, "padding_side", None)
            if old_padding_side is not None:
                tokenizer.padding_side = "left"
            try:
                tokenize_kwargs = {
                    "text": prompt_texts,
                    "return_tensors": "pt",
                    "padding": True,
                    "add_special_tokens": False,
                }
                if self._online_max_prompt_length is not None:
                    tokenize_kwargs["max_length"] = self._online_max_prompt_length
                    tokenize_kwargs["truncation"] = True
                generate_inputs = tokenizer(**tokenize_kwargs)
            finally:
                if old_padding_side is not None:
                    tokenizer.padding_side = old_padding_side
            generate_inputs = {
                key: value.to(self.accelerator.device) if isinstance(value, torch.Tensor) else value
                for key, value in generate_inputs.items()
            }
            logits_processor = None
            if presence_penalty is not None and presence_penalty != 0.0:
                excluded_ids = [self.pad_token_id]
                if self.eos_token_id is not None:
                    excluded_ids.append(self.eos_token_id)
                logits_processor = LogitsProcessorList(
                    [_PresencePenaltyLogitsProcessor(presence_penalty, excluded_ids)]
                )
            generation_kwargs = {
                "max_new_tokens": sampling_config["max_new_tokens"],
                "do_sample": True,
                "temperature": sampling_config["temperature"],
                "top_p": sampling_config["top_p"],
                "top_k": sampling_config.get("top_k"),
                "min_p": sampling_config.get("min_p"),
                "repetition_penalty": sampling_config.get("repetition_penalty"),
                "pad_token_id": self.pad_token_id,
                "eos_token_id": self.eos_token_id,
                "disable_compile": True,
            }
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
            if logits_processor is not None:
                generation_kwargs["logits_processor"] = logits_processor
            was_training = unwrapped_model.training
            unwrapped_model.eval()
            try:
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs,
                    **generation_kwargs,
                )
            finally:
                if was_training:
                    unwrapped_model.train()

        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        return [
            row_ids[row_mask.bool()].tolist()
            for row_ids, row_mask in zip(completion_ids, completion_mask, strict=True)
        ]

    def _build_local_rollout_batch_from_cases(
        self,
        rows: List[Dict[str, Any]],
    ) -> Tuple[
        List[List[int]],
        List[List[int]],
        List[List[int]],
        None,
        Dict[str, List[Any]],
        Dict[str, int],
    ]:
        """Roll out multi-turn cases with the training model's local generate path."""
        tokenizer = self.processing_class
        oc = self.online_config
        prompt_ids_list: List[List[int]] = []
        completion_ids_list: List[List[int]] = []
        env_mask_list: List[List[int]] = []
        extra_fields: Dict[str, List[Any]] = {
            "turn_responses": [],
            "turn_gts": [],
            "prefix_valid": [],
            "turn_reward_mask": [],
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

        def get_sampling_config(prefix_phase: bool) -> tuple[Dict[str, Any], Optional[float]]:
            if prefix_phase:
                return {
                    "temperature": oc.t01_temperature,
                    "top_p": oc.t01_top_p,
                    "top_k": oc.t01_top_k,
                    "min_p": oc.t01_min_p,
                    "repetition_penalty": oc.t01_repetition_penalty,
                    "max_new_tokens": oc.t01_max_tokens,
                }, oc.t01_presence_penalty
            return {
                "temperature": oc.t2_temperature,
                "top_p": oc.t2_top_p,
                "top_k": oc.t2_top_k,
                "min_p": oc.t2_min_p,
                "repetition_penalty": oc.t2_repetition_penalty,
                "max_new_tokens": oc.t2_max_tokens,
            }, oc.t2_presence_penalty

        n_turns_by_row: List[int] = []
        prefix_turns_by_row: List[int] = []
        turn_texts_by_row: List[List[str]] = [[] for _ in rows]
        prefix_valid_by_row: List[Optional[bool]] = []

        for row in rows:
            n_turns = len(row["turn_prompts"])
            if n_turns == 0 or len(row["turn_gts"]) != n_turns:
                raise ValueError(
                    f"Case {row.get('case_id', '<unknown>')} has inconsistent "
                    "turn_prompts/turn_gts lengths."
                )
            n_turns_by_row.append(n_turns)
            prefix_turns = _prefix_turn_count(row)
            prefix_turns_by_row.append(prefix_turns)
            prefix_valid_by_row.append(True if prefix_turns == 0 else None)

        max_turns = max(n_turns_by_row, default=0)
        for turn_idx in range(max_turns):
            active_indices = [
                idx for idx, n_turns in enumerate(n_turns_by_row)
                if turn_idx < n_turns and (turn_idx < prefix_turns_by_row[idx] or prefix_valid_by_row[idx] is True)
            ]
            for prefix_phase in (True, False):
                group_indices = [
                    idx for idx in active_indices
                    if (turn_idx < prefix_turns_by_row[idx]) == prefix_phase
                ]
                if not group_indices:
                    continue
                prompt_texts = [
                    tokenizer.apply_chat_template(
                        _build_case_messages_for_turn(rows[idx], turn_texts_by_row[idx], turn_idx),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for idx in group_indices
                ]
                prompt_messages = [
                    _build_case_messages_for_turn(rows[idx], turn_texts_by_row[idx], turn_idx)
                    for idx in group_indices
                ]
                sampling_config, presence_penalty = get_sampling_config(prefix_phase)
                if (
                    self.accelerator.is_main_process
                    and (self._online_step_count < 3 or self._online_step_count % 20 == 0)
                ):
                    phase = "prefix" if prefix_phase else "post"
                    print(
                        f"[v7 online rollout {self._online_step_count + 1}] "
                        f"turn={turn_idx} phase={phase} batch={len(group_indices)} "
                        f"max_new_tokens={sampling_config['max_new_tokens']}",
                        flush=True,
                    )
                generated_ids_list = self._generate_turns(
                    prompt_texts,
                    prompt_messages,
                    sampling_config,
                    presence_penalty,
                )
                for idx, turn_ids in zip(group_indices, generated_ids_list, strict=True):
                    turn_texts_by_row[idx].append(tokenizer.decode(turn_ids, skip_special_tokens=True))
                    if prefix_phase and turn_idx + 1 == prefix_turns_by_row[idx]:
                        prefix_valid_by_row[idx] = all(
                            _is_exact_match(turn_texts_by_row[idx][prefix_idx], rows[idx]["turn_gts"][prefix_idx])
                            for prefix_idx in range(prefix_turns_by_row[idx])
                        )

        for row, turn_texts, prefix_turns, n_turns, prefix_valid_state in zip(
            rows,
            turn_texts_by_row,
            prefix_turns_by_row,
            n_turns_by_row,
            prefix_valid_by_row,
            strict=True,
        ):
            sampled_turn_count = len(turn_texts)
            turn_gts = [sorted(x) for x in row["turn_gts"][:sampled_turn_count]]
            prefix_valid = bool(prefix_valid_state)
            if prefix_valid:
                summary["prefix_valid_count"] += 1
            turn_reward_mask = [1.0] * sampled_turn_count

            final_messages = _build_sampled_case_messages(row, turn_texts)
            full_ids = _chat_template_ids(tokenizer, final_messages, add_generation_prompt=False)
            first_prompt_ids = _chat_template_ids(
                tokenizer,
                _build_case_messages_for_turn(row, [], 0),
                add_generation_prompt=True,
            )
            prompt_split = len(first_prompt_ids)
            prompt_ids = full_ids[:prompt_split]
            completion_ids = full_ids[prompt_split:]

            response_spans = _build_response_token_spans(row, turn_texts, tokenizer)

            env_mask = _build_token_span_mask(len(completion_ids), response_spans, prompt_split)

            prompt_ids_list.append(prompt_ids)
            completion_ids_list.append(completion_ids)
            env_mask_list.append(env_mask)
            extra_fields["turn_responses"].append(turn_texts)
            extra_fields["turn_gts"].append(turn_gts)
            extra_fields["prefix_valid"].append(prefix_valid)
            extra_fields["turn_reward_mask"].append(turn_reward_mask)
            extra_fields["reward_weight"].append(1.0)
            extra_fields["challenge_type"].append(row.get("challenge_type", ""))
            extra_fields["oracle"].append(row.get("oracle", ""))
            extra_fields["gt_survivors"].append(json.dumps(turn_gts[-1] if turn_gts else []))

        return prompt_ids_list, completion_ids_list, env_mask_list, None, extra_fields, summary

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
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            _,
            extra_fields,
            summary,
        ) = self._build_local_rollout_batch_from_cases(inputs)
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
        gathered_completion_lengths = self.accelerator.gather(completion_lengths)
        prefix_valid_count = torch.tensor(
            float(summary.get("prefix_valid_count", 0)),
            device=self.accelerator.device,
        )
        rollout_count = torch.tensor(
            float(summary.get("n_rollouts", len(inputs))),
            device=self.accelerator.device,
        )
        gathered_prefix_valid = self.accelerator.gather(prefix_valid_count).sum()
        gathered_rollouts = self.accelerator.gather(rollout_count).sum().clamp(min=1.0)
        self._metrics[mode]["online/prefix_valid_rate"].append(
            (gathered_prefix_valid / gathered_rollouts).item()
        )
        self._metrics[mode]["online/rollouts"].append(gathered_rollouts.item())
        self._metrics[mode]["online/completion_tokens_mean"].append(
            gathered_completion_lengths.float().mean().item()
        )
        if (
            self.accelerator.is_main_process
            and (self._online_step_count < 3 or self._online_step_count % 20 == 0)
        ):
            print(
                f"[v7 online rollout {self._online_step_count + 1}] "
                f"sampled_batch_ready local_mean_tokens={completion_lengths.float().mean().item():.1f} "
                f"local_max_tokens={completion_lengths.max().item()} "
                "phase=logprob",
                flush=True,
            )

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
        logprob_inputs = self._logprob_inputs(prompt_completion_ids, attention_mask, logits_to_keep)

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    logprob_inputs,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        logprob_inputs,
                    )
                else:
                    model = self.accelerator.unwrap_model(self.model)
                    with model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            logprob_inputs,
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

        rewards_per_func = self._score_online_rewards(inputs, completions_text)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        multi_objective_aggregation = getattr(self, "multi_objective_aggregation", "sum_then_normalize")
        if multi_objective_aggregation == "sum_then_normalize":
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
        elif multi_objective_aggregation == "normalize_then_sum":
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
            raise ValueError(f"Invalid multi_objective_aggregation: {multi_objective_aggregation}")

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

    def _compute_loss(self, model, inputs):
        """Use full completion tokens as context, but train only on sampled assistant spans."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        loss_mask = completion_mask
        if "tool_mask" in inputs:
            loss_mask = completion_mask * inputs["tool_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        logprob_inputs = self._logprob_inputs(input_ids, attention_mask, logits_to_keep)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            logprob_inputs,
            compute_entropy=True,
        )

        top_entropy_quantile = getattr(self, "top_entropy_quantile", 1.0)
        if top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, loss_mask, 1 - top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        importance_sampling_level = getattr(self, "importance_sampling_level", "token")
        if importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        epsilon_base = getattr(self, "epsilon", 0.2)
        epsilon_low = getattr(self, "epsilon_low", epsilon_base)
        epsilon_high = getattr(self, "epsilon_high", epsilon_base)
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
        if getattr(self.args, "delta", None) is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        current_grad_accum = getattr(self, "current_gradient_accumulation_steps", 1)
        if self.loss_type == "grpo":
            loss = ((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / current_grad_accum
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = loss / current_grad_accum
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * loss_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / current_grad_accum
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * loss_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"
        completion_token_count = loss_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * loss_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())
        self._metrics[mode]["clip_ratio/low_mean"].append(self.accelerator.gather(low_clip).nanmean().item())
        self._metrics[mode]["clip_ratio/high_mean"].append(self.accelerator.gather(high_clip).nanmean().item())
        self._metrics[mode]["clip_ratio/region_mean"].append(self.accelerator.gather(clip_ratio).nanmean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).nanmean().item())

        return loss
