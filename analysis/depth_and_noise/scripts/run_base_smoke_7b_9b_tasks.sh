#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
CONDA_PREFIX=/disk4/xuweihong/envs/miniconda3/envs/swift
PYTHON_BIN="${PYTHON_BIN:-/disk4/xuweihong/envs/miniconda3/envs/swift/bin/python}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
MODEL_7B_PATH="${MODEL_7B_PATH:-/disk1/xuhaoming/models/Qwen2.5-7B-Instruct}"
MODEL_9B_PATH="${MODEL_9B_PATH:-/disk1/xuhaoming/models/Qwen3.5-9B}"

GPU_7B="${GPU_7B:-1}"
GPU_9B="${GPU_9B:-2}"

SOURCE_CASES="${SOURCE_CASES:-1}"
EVAL_CASES="${EVAL_CASES:-1}"
REPEATS="${REPEATS:-1}"

run_eval() {
  local task="$1"
  local model="$2"
  local pipeline="$3"
  local eval_type="$4"
  local gen_script="$5"
  local model_path="$6"
  local gpu="$7"
  local max_model_len="$8"
  local max_tokens="$9"
  local temperature="${10}"
  local enable_thinking="${11}"

  local output_root="analysis/depth_and_noise/outputs/${task}/${model}/${pipeline}"
  local eval_out="${output_root}/eval/base_smoke"

  echo "[base-smoke] generate task=${task} model=${model} pipeline=${pipeline}"
  CUDA_VISIBLE_DEVICES="$gpu" TASK="$task" MODEL="$model" MAX_SOURCE_CASES="$SOURCE_CASES" \
    bash "$gen_script"

  local eval_script
  if [[ "$model" == "7B" ]]; then
    eval_script="analysis/depth_and_noise/eval/${task}_eval_7b_test_cases_vllm.py"
  else
    eval_script="analysis/depth_and_noise/eval/${task}_eval_9b_test_cases_vllm.py"
  fi

  echo "[base-smoke] eval task=${task} model=${model} eval_type=${eval_type} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$eval_script" \
    --test-data "$output_root" \
    --output-dir "$eval_out" \
    --eval-target base \
    --base-model-path "$model_path" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$max_model_len" \
    --agent-max-tokens "$max_tokens" \
    --agent-temperature "$temperature" \
    --repeats "$REPEATS" \
    --max-cases "$EVAL_CASES" \
    --eval-types "$eval_type" \
    --enable-thinking "$enable_thinking"
}

run_model() {
  local model="$1"
  local model_path="$2"
  local gpu="$3"
  local max_model_len="$4"
  local max_tokens="$5"
  local temperature="$6"
  local enable_thinking="$7"

  for task in task_a task_b; do
    run_eval "$task" "$model" failed_stay_depth failed_stay \
      analysis/depth_and_noise/scripts/run_failed_stay_depth.sh \
      "$model_path" "$gpu" "$max_model_len" "$max_tokens" "$temperature" "$enable_thinking"

    run_eval "$task" "$model" failed_update_depth failed_update \
      analysis/depth_and_noise/scripts/run_failed_update_depth.sh \
      "$model_path" "$gpu" "$max_model_len" "$max_tokens" "$temperature" "$enable_thinking"

    run_eval "$task" "$model" noise_typology failed_isolation \
      analysis/depth_and_noise/scripts/run_noise_typology.sh \
      "$model_path" "$gpu" "$max_model_len" "$max_tokens" "$temperature" "$enable_thinking"
  done
}

run_model 7B "$MODEL_7B_PATH" "$GPU_7B" 30000 1024 0.3 false
run_model 9B "$MODEL_9B_PATH" "$GPU_9B" 200000 50000 1.0 true

echo "[base-smoke] done"
