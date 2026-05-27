#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_PREFIX="${CONDA_PREFIX:-.conda/envs/train_swift}"
if [[ -f "$CONDA_PREFIX/bin/activate" ]]; then
    source "$CONDA_PREFIX/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MAX_PIXELS=1003520
export MASTER_PORT=29500

# Model and dataset configuration
MODEL="${MODEL:-models/Qwen3-VL-8B-Thinking}"
DATASET="${DATASET:-task_a/training/refer/dataset/orientation_dataset_400_999_grpo.json}"
OUTPUT_DIR="${OUTPUT_DIR:-task_a/training/refer/GRPO_Output}"

# Batch configuration
# DP size = world_size // (CP * TP * PP) = 4 // (1 * 1 * 1) = 4
# global_batch_size = micro_batch_size * DP * gradient_accumulation_steps
# generation_batch_size = global_batch_size * steps_per_generation
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=4  # 1 * 4 * 1 = 4
STEPS_PER_GENERATION=4
NUM_GENERATIONS=4

# Learning rate configuration
LR=5e-6
NUM_EPOCHS=3

echo "=================================="
echo "GRPO Training - Global Orientation"
echo "=================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "=================================="
echo ""

megatron rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --dataset "$DATASET" \
    --num_train_epochs $NUM_EPOCHS \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --steps_per_generation $STEPS_PER_GENERATION \
    --num_generations $NUM_GENERATIONS \
    --external_plugins task_a/training/refer/grpo_plugin.py \
    --reward_funcs orientation_reward \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 8192 \
    --max_length 8192 \
    --max_completion_length 8192 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --lr $LR \
    --bf16 true \
    --beta 0.001 \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --epsilon_high 0.2 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --logging_steps 10 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --temperature 0.7 \
    --padding_free true \
    --log_completions true \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 100 \
    --eval_steps 50

echo ""
echo "Training complete."
