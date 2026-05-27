"""Unified GRPO training for belief management.

Supports two modes:
  - offline: Standard training on pre-built static datasets.
  - online:  Dynamic per-batch T0/T1 regeneration via vLLM (v7).

Usage (offline):
    CUDA_VISIBLE_DEVICES=0 python -m task_a.training.train_grpo \
        --mode offline \
        --model-path /path/to/model \
        --data-dir task_a/training/data

Usage (online, requires separate vLLM server):
    # Terminal 1: launch vLLM server
    CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model /path/to/model

    # Terminal 2: launch training
    CUDA_VISIBLE_DEVICES=0 python -m task_a.training.train_grpo \
        --mode online \
        --model-path /path/to/model \
        --target-steps 600
"""

import argparse
import os

from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from swanlab.integration.transformers import SwanLabCallback

from utils.hypotheses_parser import set_default_hypotheses_parse_mode
from task_a.training.reward import belief_exact_match_reward


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for belief management")

    # ---------- mode ----------
    p.add_argument(
        "--mode", type=str, choices=["offline", "online"], default="online",
        help="offline: train on static dataset; online: dynamic per-batch regeneration via vLLM",
    )

    # ---------- common ----------
    p.add_argument(
        "--model-path", type=str,
        default=os.environ.get("MODEL_PATH", "models/Qwen2.5-7B-Instruct"),
    )
    p.add_argument("--output-dir", type=str, default=None,
                   help="Default: task_a/training/checkpoints_{mode}")

    # LoRA
    p.add_argument("--no-lora", action="store_true", default=False)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # GRPO hyper-parameters
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hypotheses-parse-mode",
        type=str,
        choices=["tag", "json", "auto"],
        default="tag",
        help="How to parse model hypothesis outputs during training reward computation.",
    )

    # ---------- offline-specific ----------
    p.add_argument("--data-dir", type=str, default="task_a/training/data",
                   help="(offline) Path to HuggingFace dataset on disk")
    p.add_argument("--num-train-epochs", type=int, default=3,
                   help="(offline) Number of training epochs")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="(offline) Max training steps; -1 = use epochs")
    p.add_argument("--deepspeed", type=str, default=None,
                   help="(offline) DeepSpeed config JSON path")

    # ---------- online-specific ----------
    p.add_argument("--target-steps", type=int, default=600,
                   help="(online) Total optimization steps")
    p.add_argument("--vllm-server-host", type=str, default="0.0.0.0",
                   help="(online) vLLM server host")
    p.add_argument("--vllm-server-port", type=int, default=8000,
                   help="(online) vLLM server port")
    p.add_argument("--failed_update-ratio", type=float, default=0.3,
                   help="(online) Fraction of failed_update samples vs failed_stay")
    p.add_argument("--cot", action="store_true", default=False,
                   help="(online) Append CoT instruction to T2 prompt")
    p.add_argument("--online-seed", type=int, default=1234,
                   help="(online) RNG seed for challenge generation")

    return p.parse_args()


def _build_lora_config(args):
    if args.no_lora:
        return None
    print(f"Using LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )


# ──────────────────────────────────────────────────────────────
#  Offline mode: static dataset
# ──────────────────────────────────────────────────────────────

def run_offline(args):
    set_default_hypotheses_parse_mode(args.hypotheses_parse_mode)
    print(f"[offline] Loading dataset from {args.data_dir}")
    train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
    print(f"[offline] Train samples: {len(train_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        beta=args.beta,
        use_vllm=False,
        deepspeed=args.deepspeed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=True,
        log_completions=True,
        report_to="none",
    )

    swanlab_cb = SwanLabCallback(
        project="task_a",
        workspace="experiment_workspace",
        experiment_name=f"offline_ep{args.num_train_epochs}",
        config={
            "model": os.path.basename(args.model_path),
            "mode": "offline",
            "lora": not args.no_lora,
            "num_generations": args.num_generations,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "train_samples": len(train_dataset),
        },
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=belief_exact_match_reward,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=_build_lora_config(args),
        callbacks=[swanlab_cb],
    )

    print("[offline] Starting training...")
    trainer.train()

    out = os.path.join(args.output_dir, "final")
    print(f"[offline] Saving to {out}")
    trainer.save_model(out)
    tokenizer.save_pretrained(out)


# ──────────────────────────────────────────────────────────────
#  Online mode: dynamic per-batch T0/T1 regeneration
# ──────────────────────────────────────────────────────────────

def run_online(args):
    from task_a.core.rules import BENCHMARK_RULES
    from task_a.training.online_trainer import DynamicOnlineGRPOTrainer, OnlineConfig

    set_default_hypotheses_parse_mode(args.hypotheses_parse_mode)
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps
    dataset_len = max(args.target_steps * effective_batch, 64)
    placeholder = Dataset.from_dict({
        "prompt": ["__placeholder__"] * dataset_len,
        "gt_survivors": ["[]"] * dataset_len,
    })
    print(f"[online] Placeholder dataset: {dataset_len} rows (replaced per-batch)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1,
        max_steps=args.target_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=True,
        log_completions=True,
        report_to="none",
    )

    swanlab_cb = SwanLabCallback(
        project="task_a",
        workspace="experiment_workspace",
        experiment_name=f"online_steps{args.target_steps}_ir{args.failed_update_ratio}",
        config={
            "model": os.path.basename(args.model_path),
            "mode": "online",
            "failed_update_ratio": args.failed_update_ratio,
            "cot": args.cot,
            "target_steps": args.target_steps,
            "num_generations": args.num_generations,
            "per_device_batch": args.per_device_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
        },
    )

    online_cfg = OnlineConfig(
        failed_update_ratio=args.failed_update_ratio,
        oracle_pool=list(BENCHMARK_RULES),
        cot=args.cot,
    )

    print("[online] Initializing DynamicOnlineGRPOTrainer...")
    trainer = DynamicOnlineGRPOTrainer(
        model=args.model_path,
        reward_funcs=belief_exact_match_reward,
        args=config,
        train_dataset=placeholder,
        processing_class=tokenizer,
        peft_config=_build_lora_config(args),
        callbacks=[swanlab_cb],
        online_config=online_cfg,
        online_seed=args.online_seed,
    )

    print(f"[online] Starting training: {args.target_steps} steps")
    trainer.train()

    out = os.path.join(args.output_dir, "final")
    print(f"[online] Saving to {out}")
    trainer.save_model(out)
    tokenizer.save_pretrained(out)


# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Default output dir per mode
    if args.output_dir is None:
        args.output_dir = f"task_a/training/checkpoints_{args.mode}"

    print(f"Mode: {args.mode} | Model: {args.model_path} | Output: {args.output_dir}")

    if args.mode == "offline":
        run_offline(args)
    else:
        run_online(args)

    print("Done.")


if __name__ == "__main__":
    main()
