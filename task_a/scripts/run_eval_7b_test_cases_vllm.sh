#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDA_PREFIX="${CONDA_PREFIX:-.conda/envs/swift}"
PYTHON_BIN=$CONDA_PREFIX/bin/python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

CUDA_VISIBLE_DEVICES=1
TEST_DATA=data/task_a_7B_test_cases
EVAL_TYPES=(failed_stay failed_update failed_isolation)

BASE_MODEL_PATH=${BASE_MODEL_PATH:-models/Qwen2.5-7B-Instruct}
LORA_SOURCE_TYPE=merged

EVAL_TARGETS=(
  base
)

LORA_PATHS=(
  ""
)

OUTPUT_DIRS=(
  task_a/outputs/test_7B_prompt_enhanced_refined_v2
)

BASE_RESULT_DIR=

TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=32000
MODEL_DTYPE=bfloat16
DISABLE_CUSTOM_ALL_REDUCE=true

AGENT_TEMPERATURE=0.3
AGENT_MAX_TOKENS=1024
SAMPLING_TOP_P=
SAMPLING_TOP_K=
SAMPLING_PRESENCE_PENALTY=
SAMPLING_REPETITION_PENALTY=
REPEATS=3
SEED=
PROMPT_ENHANCEMENT=true

if [[ ${#EVAL_TARGETS[@]} -ne ${#LORA_PATHS[@]} || ${#EVAL_TARGETS[@]} -ne ${#OUTPUT_DIRS[@]} ]]; then
  echo "[eval-a-7b-test-cases] EVAL_TARGETS, LORA_PATHS and OUTPUT_DIRS must have the same length" >&2
  exit 1
fi

has_reusable_base_result() {
  local root="$1"
  local base_label="$2"
  [[ -n "$root" ]] || return 1
  [[ -d "$root/$base_label" ]] || return 1
  [[ -f "$root/$base_label/stats_report.json" ]] || return 1
  [[ -f "$root/$base_label/failed_stay/stats_report.json" ]] || return 1
  [[ -f "$root/$base_label/failed_update/stats_report.json" ]] || return 1
  [[ -f "$root/$base_label/failed_isolation/stats_report.json" ]] || return 1
  return 0
}

copy_reusable_base_result() {
  local src_root="$1"
  local dst_root="$2"
  local base_label="$3"
  mkdir -p "$dst_root"
  cp -a "$src_root/$base_label" "$dst_root/"
}

for idx in "${!EVAL_TARGETS[@]}"; do
  eval_target="${EVAL_TARGETS[$idx]}"
  lora_path="${LORA_PATHS[$idx]}"
  output_dir="${OUTPUT_DIRS[$idx]}"
  effective_target="$eval_target"
  base_label=base
  if [[ "$PROMPT_ENHANCEMENT" == "true" ]]; then
    base_label=base_prompt_enhanced
  fi

  if [[ "$eval_target" != "base" && -z "$lora_path" ]]; then
    echo "[eval-a-7b-test-cases] non-base run at index $idx requires a lora path" >&2
    exit 1
  fi

  mkdir -p "$output_dir"

  if [[ "$eval_target" == "base" || "$eval_target" == "both" ]]; then
    if has_reusable_base_result "$BASE_RESULT_DIR" "$base_label"; then
      echo "[eval-a-7b-test-cases] reusing cached base result from $BASE_RESULT_DIR"
      copy_reusable_base_result "$BASE_RESULT_DIR" "$output_dir" "$base_label"
      if [[ "$eval_target" == "base" ]]; then
        echo "[eval-a-7b-test-cases] base run skipped because cached base result was copied"
        continue
      fi
      effective_target="lora"
    fi
  fi

  echo "[eval-a-7b-test-cases] index=$idx"
  echo "[eval-a-7b-test-cases] test_data=$TEST_DATA"
  echo "[eval-a-7b-test-cases] output_dir=$output_dir"
  echo "[eval-a-7b-test-cases] eval_target=$eval_target effective_target=$effective_target repeats=$REPEATS"
  echo "[eval-a-7b-test-cases] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[eval-a-7b-test-cases] base_model_path=$BASE_MODEL_PATH"
  echo "[eval-a-7b-test-cases] lora_source_type=$LORA_SOURCE_TYPE"
  echo "[eval-a-7b-test-cases] lora_path=$lora_path"
  echo "[eval-a-7b-test-cases] base_result_dir=$BASE_RESULT_DIR"
  echo "[eval-a-7b-test-cases] prompt_enhancement=$PROMPT_ENHANCEMENT base_label=$base_label"
  echo "[eval-a-7b-test-cases] tp=$TENSOR_PARALLEL_SIZE max_model_len=$MAX_MODEL_LEN model_dtype=$MODEL_DTYPE"
  echo "[eval-a-7b-test-cases] disable_custom_all_reduce=$DISABLE_CUSTOM_ALL_REDUCE"

  ARGS=(
    task_a/training/eval_7b_test_cases_vllm.py
    --test-data "$TEST_DATA"
    --output-dir "$output_dir"
    --eval-target "$effective_target"
    --base-model-path "$BASE_MODEL_PATH"
    --lora-source-type "$LORA_SOURCE_TYPE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --model-dtype "$MODEL_DTYPE"
    --agent-temperature "$AGENT_TEMPERATURE"
    --agent-max-tokens "$AGENT_MAX_TOKENS"
    --repeats "$REPEATS"
    --eval-types "${EVAL_TYPES[@]}"
  )

  if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "true" ]]; then
    ARGS+=(--disable-custom-all-reduce)
  fi

  if [[ -n "$SAMPLING_TOP_P" ]]; then
    ARGS+=(--sampling-top-p "$SAMPLING_TOP_P")
  fi

  if [[ -n "$SAMPLING_TOP_K" ]]; then
    ARGS+=(--sampling-top-k "$SAMPLING_TOP_K")
  fi

  if [[ -n "$SAMPLING_PRESENCE_PENALTY" ]]; then
    ARGS+=(--sampling-presence-penalty "$SAMPLING_PRESENCE_PENALTY")
  fi

  if [[ -n "$SAMPLING_REPETITION_PENALTY" ]]; then
    ARGS+=(--sampling-repetition-penalty "$SAMPLING_REPETITION_PENALTY")
  fi

  if [[ "$LORA_SOURCE_TYPE" == "merged" && -n "$lora_path" ]]; then
    ARGS+=(--lora-model-path "$lora_path")
  fi

  if [[ "$LORA_SOURCE_TYPE" == "adapter" && -n "$lora_path" ]]; then
    ARGS+=(--lora-adapter-path "$lora_path")
  fi

  if [[ -n "$SEED" ]]; then
    ARGS+=(--seed "$SEED")
  fi

  if [[ "$PROMPT_ENHANCEMENT" == "true" ]]; then
    ARGS+=(--enable-prompt-enhancement)
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "${ARGS[@]}"
done
