"""v7 GRPO training: dynamic online RL with per-batch multi-turn regeneration.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m task_a.training.train_grpo_v7 \
        --model-path models/Qwen2.5-7B-Instruct \
        --cases-json task_a/training/data/cases/train_cases.json \
        --output-dir task_a/training/checkpoints_v7 \
        --num-train-epochs 1
"""

import argparse
import math
import inspect
import os
from collections import Counter, defaultdict

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from swift.model import get_model_processor
from swift.rlhf_trainers import GRPOConfig
from swift.template import get_template
from swift.utils import get_current_device

try:
    from swanlab.integration.transformers import SwanLabCallback
except Exception:
    SwanLabCallback = None

from task_a.training.cases_dataset import load_cases_dataset
from utils.hypotheses_parser import set_default_hypotheses_parse_mode
from task_a.training.online_trainer import (
    DynamicOnlineGRPOTrainer,
    OnlineConfig,
)
from task_a.training.reward import belief_exact_match_reward


def _case_mode(row) -> str:
    challenge_type = str(row.get("challenge_type", "")).lower()
    if "failed_stay" in challenge_type:
        return "failed_stay"
    if "failed_update" in challenge_type or "misrecord" in challenge_type or "correction" in challenge_type:
        return "failed_update"
    return challenge_type


def _prefix_turn_count(row) -> int:
    n_turns = len(row["turn_prompts"])
    mode = _case_mode(row)
    if mode == "failed_update":
        return min(3, n_turns)
    if mode == "failed_stay":
        return min(2, n_turns)
    return max(n_turns - 1, 0)


def _validate_case_turns(dataset, expected_num_turns: int | None = None) -> None:
    turn_counts = Counter()
    prefix_counts = defaultdict(Counter)
    for row in dataset:
        n_turns = len(row["turn_prompts"])
        turn_counts[n_turns] += 1
        prefix_counts[_case_mode(row)][_prefix_turn_count(row)] += 1

    print(f"Case turn-count distribution: {dict(sorted(turn_counts.items()))}")
    print(
        "Case prefix-turn distribution: "
        f"{ {mode: dict(sorted(counts.items())) for mode, counts in sorted(prefix_counts.items())} }"
    )
    if expected_num_turns is not None and set(turn_counts) != {expected_num_turns}:
        raise ValueError(
            f"Expected every case to have {expected_num_turns} turns, "
            f"but found distribution {dict(sorted(turn_counts.items()))}."
        )


def _build_grpo_config(args) -> GRPOConfig:
    """Construct Swift GRPOConfig while tolerating version differences."""
    config_kwargs = {
        "output_dir": args.output_dir,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "max_prompt_length": args.max_prompt_length,
        "temperature": args.temperature,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "gradient_checkpointing": args.gradient_checkpointing,
        "deepspeed": args.deepspeed,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "bf16": True,
        "log_completions": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "use_vllm": args.use_vllm,
        "vllm_mode": "server" if args.use_vllm else None,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_max_model_len": args.vllm_max_model_len or args.max_model_len,
        "vllm_max_model_length": args.vllm_max_model_len or args.max_model_len,
        "vllm_server_host": [args.vllm_server_host],
        "vllm_server_port": [args.vllm_server_port],
        "vllm_server_group_port": [args.vllm_server_group_port],
        "vllm_server_timeout": args.vllm_server_timeout,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "vllm_enforce_eager": args.vllm_enforce_eager,
        "vllm_enable_prefix_caching": args.vllm_enable_prefix_caching,
        "vllm_enable_lora": args.vllm_enable_lora,
        "lora_rank": args.lora_r,
        "tuner_type": "full" if args.no_lora or args.full_finetuning else "lora",
        "sleep_level": args.vllm_sleep_level,
        "move_model_batches": args.move_model_batches,
    }
    if args.online_top_p is not None:
        config_kwargs["top_p"] = args.online_top_p
    if args.online_top_k is not None:
        config_kwargs["top_k"] = args.online_top_k
    if args.online_min_p is not None:
        config_kwargs["min_p"] = args.online_min_p
    if args.online_repetition_penalty is not None:
        config_kwargs["repetition_penalty"] = args.online_repetition_penalty
    if args.max_steps is not None:
        config_kwargs["max_steps"] = args.max_steps
    valid_params = inspect.signature(GRPOConfig.__init__).parameters
    filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in valid_params}
    skipped = sorted(set(config_kwargs) - set(filtered_kwargs))
    if skipped:
        print(f"Skipping unsupported GRPOConfig args for this Swift version: {', '.join(skipped)}")
    return GRPOConfig(**filtered_kwargs)


def _build_vllm_client(args, config):
    """Create Swift's rollout client when GRPOConfig was built outside Swift's pipeline."""
    if not args.use_vllm:
        return None
    existing_client = getattr(config, "vllm_client", None)
    if existing_client is not None:
        return existing_client

    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return None

    from swift.rlhf_trainers import VLLMClient

    print(
        "Connecting to Swift vLLM rollout server: "
        f"http://{args.vllm_server_host}:{args.vllm_server_port}, "
        f"group_port={args.vllm_server_group_port}"
    )
    client = VLLMClient(
        hosts=[args.vllm_server_host],
        server_ports=[args.vllm_server_port],
        group_ports=[args.vllm_server_group_port],
        connection_timeout=args.vllm_server_timeout,
    )
    client.close_communicator()
    client.init_communicator(device=get_current_device())
    print("Connected to Swift vLLM rollout server.")
    return client


def parse_args():
    p = argparse.ArgumentParser(description="v7 dynamic online GRPO training")
    p.add_argument(
        "--model-path", type=str,
        default=os.environ.get("MODEL_PATH", "models/Qwen2.5-7B-Instruct"),
    )
    p.add_argument("--output-dir", type=str, default="task_a/training/checkpoints_v7")
    p.add_argument(
        "--cases-json",
        type=str,
        default="task_a/training/data/cases/train_cases.json",
    )

    # LoRA
    p.add_argument("--no-lora", action="store_true", default=False)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # GRPO training
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Optional override. If unset, steps are derived from dataset size and epochs.")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--save-num", type=int, default=5,
                   help="Desired number of checkpoints across the whole run. Ignored if --save-steps is set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--curriculum-phase", type=int, default=1,
                   help="Curriculum phase label (1 or 2) for SwanLab tracking")
    p.add_argument("--expected-num-turns", type=int, default=None,
                   help="Optional safety check that every training case has this many turns.")
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="Path to checkpoint dir to resume training from")
    p.add_argument("--deepspeed", type=str, default=None,
                   help="Optional DeepSpeed config for multi-GPU training")
    p.add_argument("--device-map", type=str, default=None,
                   help="Device map passed to Swift model loading. Defaults to Swift's own default.")
    p.add_argument("--attn-impl", type=str, default=None,
                   help="Optional attention implementation passed to Swift, e.g. flash_attn.")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Optional max_model_len passed to Swift model/template loading.")
    p.add_argument("--template", type=str, default=None,
                   help="Optional Swift template type override. Defaults to model metadata.")
    p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=False)
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--load-in-8bit", action="store_true", default=False)
    p.add_argument("--load-in-16bit", action="store_true", default=False)
    p.add_argument("--full-finetuning", action="store_true", default=False)

    # vLLM rollout backend
    p.add_argument("--use-vllm", action="store_true", default=False,
                   help="Use a dedicated Swift vLLM rollout server for online sampling.")
    p.add_argument("--vllm-server-host", type=str, default="127.0.0.1",
                   help="Host of the dedicated Swift rollout server.")
    p.add_argument("--vllm-server-port", type=int, default=8000,
                   help="HTTP port of the dedicated Swift rollout server.")
    p.add_argument("--vllm-server-group-port", type=int, default=51216,
                   help="NCCL communicator port used for training-to-server weight sync.")
    p.add_argument("--vllm-server-timeout", type=float, default=600.0,
                   help="Timeout in seconds when connecting to the Swift rollout server.")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9,
                   help="vLLM GPU memory utilization used by the rollout server launcher.")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1,
                   help="Tensor parallel size used by the rollout server launcher.")
    p.add_argument("--vllm-max-model-len", type=int, default=None,
                   help="vLLM max model length. Defaults to --max-model-len.")
    p.add_argument("--vllm-max-num-seqs", type=int, default=None,
                   help="Optional vLLM max_num_seqs override.")
    p.add_argument("--vllm-enforce-eager", action="store_true", default=False,
                   help="Pass enforce_eager to vLLM.")
    p.add_argument("--vllm-enable-prefix-caching", action="store_true", default=False,
                   help="Enable vLLM prefix caching.")
    p.add_argument("--vllm-enable-lora", action="store_true", default=False,
                   help="Sync LoRA adapters to vLLM instead of merged full weights.")
    p.add_argument("--vllm-sleep-level", type=int, default=0,
                   help="Swift vLLM sleep level after rollout; 0 keeps engine resident.")
    p.add_argument("--move-model-batches", type=int, default=None,
                   help="Optional Swift weight-sync batching override.")

    # Rollout config
    p.add_argument("--online-temperature", type=float, default=0.3,
                   help="Sampling temperature used for all rollout turns")
    p.add_argument("--online-top-p", type=float, default=0.9,
                   help="Top-p used for all rollout turns")
    p.add_argument("--online-top-k", type=int, default=None,
                   help="Optional top-k used for all rollout turns")
    p.add_argument("--online-min-p", type=float, default=None,
                   help="Optional min-p used for all rollout turns")
    p.add_argument("--online-presence-penalty", type=float, default=None,
                   help="Optional presence penalty used for all rollout turns")
    p.add_argument("--online-repetition-penalty", type=float, default=None,
                   help="Optional repetition penalty used for all rollout turns")
    p.add_argument("--online-max-tokens", "--max-output-tokens", dest="online_max_tokens", type=int, default=256,
                   help="Max new tokens used for all rollout turns")
    p.add_argument("--cot", action="store_true", default=False,
                   help="Append CoT instruction to T2 prompt (matches v6)")
    p.add_argument(
        "--hypotheses-parse-mode",
        type=str,
        choices=["tag", "json", "auto"],
        default="auto",
        help="How to parse model hypothesis outputs during training reward computation.",
    )
    return p.parse_args()


def _infer_total_steps(args, dataset_size: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_micro_batch = max(args.per_device_batch_size * world_size, 1)
    steps_per_epoch = math.ceil(dataset_size / (global_micro_batch * args.gradient_accumulation_steps))
    steps_per_epoch = max(steps_per_epoch, 1)
    if args.max_steps is not None:
        return args.max_steps
    return max(math.ceil(args.num_train_epochs * steps_per_epoch), 1)


def _resolve_save_steps(args, dataset_size: int) -> tuple[int, int]:
    total_steps = _infer_total_steps(args, dataset_size)
    if args.save_steps is not None:
        return max(args.save_steps, 1), total_steps
    save_num = max(args.save_num, 1)
    save_steps = max(math.ceil(total_steps / save_num), 1)
    return save_steps, total_steps


def _build_quantization_config(args):
    if args.load_in_16bit:
        return None
    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if args.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def main():
    args = parse_args()
    if args.max_completion_length < args.online_max_tokens:
        print(
            "Warning: max_completion_length is smaller than online_max_tokens. "
            "The online rollout still uses online_max_tokens per turn, while "
            "max_completion_length is used by the GRPO config/loss normalization."
        )
    set_default_hypotheses_parse_mode(args.hypotheses_parse_mode)
    train_dataset = load_cases_dataset(args.cases_json)
    print(f"Train dataset: {len(train_dataset)} cases from {args.cases_json}")
    _validate_case_turns(train_dataset, args.expected_num_turns)
    args.save_steps, inferred_total_steps = _resolve_save_steps(args, len(train_dataset))
    print(
        f"Checkpoint schedule: save_steps={args.save_steps} "
        f"(save_num={args.save_num}, inferred_total_steps={inferred_total_steps})"
    )

    max_model_len = args.max_model_len
    if max_model_len is None:
        max_model_len = args.max_prompt_length + args.online_max_tokens * 3
    if args.vllm_max_model_len is None:
        args.vllm_max_model_len = max_model_len
    print(
        "Loading model with Swift: "
        f"max_model_len={max_model_len}, load_in_4bit={args.load_in_4bit}, "
        f"load_in_8bit={args.load_in_8bit}, load_in_16bit={args.load_in_16bit}, "
        f"device_map={args.device_map}, attn_impl={args.attn_impl}"
    )
    model_for_trainer, tokenizer = get_model_processor(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        quantization_config=_build_quantization_config(args),
        max_model_len=max_model_len,
        attn_impl=args.attn_impl,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    template = get_template(
        tokenizer,
        template_type=args.template,
        max_length=max_model_len,
        truncation_strategy="left",
        padding_side="left",
    )
    template.set_mode("train")
    if template.use_model:
        template.model = model_for_trainer

    config = _build_grpo_config(args)
    vllm_client = _build_vllm_client(args, config)

    callbacks = []
    if SwanLabCallback is not None:
        callbacks.append(
            SwanLabCallback(
                project="belief_training_task_a",
                workspace="12345",
                experiment_name=f"online_training_grpo_{args.num_train_epochs}epochs",
                config={
                    "model": os.path.basename(args.model_path),
                    "version": f"v11_curriculum_p{args.curriculum_phase}",
                    "cases_json": args.cases_json,
                    "train_samples": len(train_dataset),
                    "online_temperature": args.online_temperature,
                    "online_top_p": args.online_top_p,
                    "online_top_k": args.online_top_k,
                    "online_min_p": args.online_min_p,
                    "online_presence_penalty": args.online_presence_penalty,
                    "online_repetition_penalty": args.online_repetition_penalty,
                    "online_max_tokens": args.online_max_tokens,
                    "framework": "swift",
                    "use_vllm": args.use_vllm,
                    "vllm_mode": "server" if args.use_vllm else None,
                    "vllm_server_host": args.vllm_server_host,
                    "vllm_server_port": args.vllm_server_port,
                    "vllm_server_group_port": args.vllm_server_group_port,
                    "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                    "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
                    "cot": args.cot,
                    "num_train_epochs": args.num_train_epochs,
                    "max_steps": args.max_steps,
                    "save_num": args.save_num,
                    "resolved_save_steps": args.save_steps,
                    "num_generations": args.num_generations,
                    "per_device_batch": args.per_device_batch_size,
                    "grad_accum": args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "beta": args.beta,
                },
            )
        )
    else:
        print("SwanLab is not installed; continuing without SwanLabCallback.")

    if not args.no_lora and not args.full_finetuning:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        if args.load_in_4bit or args.load_in_8bit:
            model_for_trainer = prepare_model_for_kbit_training(
                model_for_trainer,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        model_meta = getattr(model_for_trainer, "model_meta", None)
        model_info = getattr(model_for_trainer, "model_info", None)
        model_for_trainer = get_peft_model(model_for_trainer, peft_config)
        if model_meta is not None:
            model_for_trainer.model_meta = model_meta
        if model_info is not None:
            model_for_trainer.model_info = model_info
        if template.use_model:
            template.model = model_for_trainer
        print(f"Using PEFT LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    online_cfg = OnlineConfig(
        cot=args.cot,
        t01_temperature=args.online_temperature,
        t01_top_p=args.online_top_p,
        t01_top_k=args.online_top_k,
        t01_min_p=args.online_min_p,
        t01_presence_penalty=args.online_presence_penalty,
        t01_repetition_penalty=args.online_repetition_penalty,
        t01_max_tokens=args.online_max_tokens,
        t2_temperature=args.online_temperature,
        t2_top_p=args.online_top_p,
        t2_top_k=args.online_top_k,
        t2_min_p=args.online_min_p,
        t2_presence_penalty=args.online_presence_penalty,
        t2_repetition_penalty=args.online_repetition_penalty,
        t2_max_tokens=args.online_max_tokens,
        use_vllm_fast_inference=args.use_vllm,
    )

    print("Initializing DynamicOnlineGRPOTrainer...")
    trainer = DynamicOnlineGRPOTrainer(
        model=model_for_trainer,
        reward_funcs=belief_exact_match_reward,
        args=config,
        train_dataset=train_dataset,
        template=template,
        callbacks=callbacks,
        vllm_client=vllm_client,
        online_config=online_cfg,
    )

    if args.max_steps is None:
        print(f"Starting training: epochs={args.num_train_epochs} over {len(train_dataset)} cases")
    else:
        print(f"Starting training: epochs={args.num_train_epochs}, max_steps={args.max_steps}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving to {args.output_dir}/final")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
