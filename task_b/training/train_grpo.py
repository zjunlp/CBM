"""GRPO training for Scenario B circuit fault diagnosis.

Dynamic online RL: per-batch T0/T1 rollout + T2 reward via vLLM server.

Usage:
    # Terminal 1: launch vLLM server
    CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model /path/to/model \
        --max-model-len 10240

    # Terminal 2: launch training
    CUDA_VISIBLE_DEVICES=0 python -m task_b.training.train_grpo \
        --model-path /path/to/model \
        --cases-json task_b/training/data/cases/train_cases.json \
        --output-dir task_b/training/checkpoints
"""

import argparse
import inspect
import math
import os

from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig
from swanlab.integration.transformers import SwanLabCallback

from task_b.training.cases_dataset import load_cases_dataset
from task_b.training.online_trainer import DynamicOnlineGRPOTrainer, OnlineConfig
from task_b.training.reward import circuit_fault_jaccard_reward


def _build_grpo_config(args) -> GRPOConfig:
    """Construct GRPOConfig while tolerating TRL version differences."""
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
        "use_vllm": True,
        "vllm_server_host": args.vllm_server_host,
        "vllm_server_port": args.vllm_server_port,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "bf16": True,
        "log_completions": True,
        "report_to": "none",
    }
    if args.online_top_p is not None:
        config_kwargs["top_p"] = args.online_top_p
    if args.online_top_k is not None:
        config_kwargs["top_k"] = args.online_top_k
    if args.online_repetition_penalty is not None:
        config_kwargs["repetition_penalty"] = args.online_repetition_penalty
    if args.vllm_max_model_len is not None:
        config_kwargs["vllm_max_model_len"] = args.vllm_max_model_len
    if args.max_steps is not None:
        config_kwargs["max_steps"] = args.max_steps

    valid_params = inspect.signature(GRPOConfig.__init__).parameters
    filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in valid_params}
    skipped = sorted(set(config_kwargs) - set(filtered_kwargs))
    if skipped:
        print(f"Skipping unsupported GRPOConfig args for this TRL version: {', '.join(skipped)}")
    return GRPOConfig(**filtered_kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for Scenario B circuit fault diagnosis")
    p.add_argument(
        "--model-path", type=str,
        default=os.environ.get("MODEL_PATH", "models/Qwen2.5-7B-Instruct"),
    )
    p.add_argument("--output-dir", type=str, default="task_b/training/checkpoints")
    p.add_argument(
        "--cases-json", type=str,
        default="task_b/training/data/cases/train_cases.json",
        help="Path to cases JSON produced by case_conversion.py",
    )

    # LoRA
    p.add_argument("--no-lora", action="store_true", default=False)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # GRPO hyper-parameters
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Optional step cap. If unset, inferred from epochs x dataset size.")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--vllm-server-host", type=str, default="localhost")
    p.add_argument("--vllm-server-port", type=int, default=8000)
    p.add_argument(
        "--vllm-max-model-len", type=int, default=None,
        help="Optional TRL passthrough. The vLLM server config remains authoritative.",
    )
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--save-num", type=int, default=5,
                   help="Desired checkpoint count across the run. Ignored if --save-steps is set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    p.add_argument("--deepspeed", type=str, default=None,
                   help="Optional DeepSpeed config for multi-GPU training")

    # Rollout generation config
    p.add_argument("--online-temperature", type=float, default=0.3,
                   help="Sampling temperature for T0/T1/T2 rollout turns")
    p.add_argument("--online-top-p", type=float, default=0.9)
    p.add_argument("--online-top-k", type=int, default=None)
    p.add_argument("--online-presence-penalty", type=float, default=None)
    p.add_argument("--online-repetition-penalty", type=float, default=None)
    p.add_argument(
        "--online-max-tokens", "--max-output-tokens",
        dest="online_max_tokens", type=int, default=512,
        help="Max new tokens per rollout turn",
    )
    p.add_argument("--cot", action="store_true", default=False,
                   help="Append chain-of-thought instruction to T2 prompt")

    return p.parse_args()


def _infer_total_steps(args, dataset_size: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_micro_batch = max(args.per_device_batch_size * world_size, 1)
    steps_per_epoch = math.ceil(
        dataset_size / (global_micro_batch * args.gradient_accumulation_steps)
    )
    steps_per_epoch = max(steps_per_epoch, 1)
    if args.max_steps is not None:
        return args.max_steps
    return max(math.ceil(args.num_train_epochs * steps_per_epoch), 1)


def _resolve_save_steps(args, dataset_size: int) -> tuple:
    total_steps = _infer_total_steps(args, dataset_size)
    if args.save_steps is not None:
        return max(args.save_steps, 1), total_steps
    save_num = max(args.save_num, 1)
    save_steps = max(math.ceil(total_steps / save_num), 1)
    return save_steps, total_steps


def main() -> None:
    args = parse_args()

    # Align max_completion_length to online_max_tokens
    if args.max_completion_length != args.online_max_tokens:
        print(
            f"Aligning max_completion_length to online_max_tokens: "
            f"{args.max_completion_length} -> {args.online_max_tokens}"
        )
        args.max_completion_length = args.online_max_tokens

    train_dataset = load_cases_dataset(args.cases_json)
    print(f"Train dataset: {len(train_dataset)} cases from {args.cases_json}")

    args.save_steps, inferred_total_steps = _resolve_save_steps(args, len(train_dataset))
    print(
        f"Checkpoint schedule: save_steps={args.save_steps} "
        f"(save_num={args.save_num}, inferred_total_steps={inferred_total_steps})"
    )

    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    config = _build_grpo_config(args)

    swanlab_cb = SwanLabCallback(
        project="belief_training_task_b",
        workspace="experiment_workspace",
        experiment_name=f"grpo_{args.num_train_epochs}ep",
        config={
            "model": os.path.basename(args.model_path),
            "cases_json": args.cases_json,
            "train_samples": len(train_dataset),
            "online_temperature": args.online_temperature,
            "online_top_p": args.online_top_p,
            "online_top_k": args.online_top_k,
            "online_max_tokens": args.online_max_tokens,
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

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        print(f"Using LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    online_cfg = OnlineConfig(
        cot=args.cot,
        t01_temperature=args.online_temperature,
        t01_top_p=args.online_top_p,
        t01_top_k=args.online_top_k,
        t01_presence_penalty=args.online_presence_penalty,
        t01_repetition_penalty=args.online_repetition_penalty,
        t01_max_tokens=args.online_max_tokens,
        t2_temperature=args.online_temperature,
        t2_top_p=args.online_top_p,
        t2_top_k=args.online_top_k,
        t2_presence_penalty=args.online_presence_penalty,
        t2_repetition_penalty=args.online_repetition_penalty,
        t2_max_tokens=args.online_max_tokens,
    )

    print("Initializing DynamicOnlineGRPOTrainer...")
    trainer = DynamicOnlineGRPOTrainer(
        model=args.model_path,
        reward_funcs=circuit_fault_jaccard_reward,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[swanlab_cb],
        online_config=online_cfg,
    )

    if args.max_steps is None:
        print(f"Starting training: epochs={args.num_train_epochs} over {len(train_dataset)} cases")
    else:
        print(f"Starting training: max_steps={args.max_steps}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    out = os.path.join(args.output_dir, "final")
    print(f"Saving to {out}")
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print("Done.")


if __name__ == "__main__":
    main()
