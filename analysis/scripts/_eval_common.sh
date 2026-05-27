#!/bin/bash
# Shared helpers for analysis vLLM evaluation scripts.

BELIEF_TRAINING_ENV="${BELIEF_TRAINING_ENV:-.conda/envs/swift}"

# OpenAI-compatible API backend defaults. Override these with environment variables
# when launching EVAL_TARGET=api.
export API_MAX_WORKERS="${API_MAX_WORKERS:-64}"
export API_CASE_MODEL="${API_CASE_MODEL:-9B}"
export API_CONTAMINATION_CASE_MODEL="${API_CONTAMINATION_CASE_MODEL:-api}"
export API_ENABLE_THINKING="${API_ENABLE_THINKING:-true}"
export API_AGENT_TEMPERATURE="${API_AGENT_TEMPERATURE:-}"
export API_AGENT_MAX_TOKENS="${API_AGENT_MAX_TOKENS:-30000}"
export API_SAMPLING_TOP_P="${API_SAMPLING_TOP_P:-}"
export API_SAMPLING_TOP_K="${API_SAMPLING_TOP_K:-}"
export API_SAMPLING_PRESENCE_PENALTY="${API_SAMPLING_PRESENCE_PENALTY:-}"
export API_SAMPLING_REPETITION_PENALTY="${API_SAMPLING_REPETITION_PENALTY:-}"
export API_SAMPLING_FREQUENCY_PENALTY="${API_SAMPLING_FREQUENCY_PENALTY:-}"
export API_SAMPLING_MIN_P="${API_SAMPLING_MIN_P:-}"

analysis_repo_root() {
  cd "$(dirname "${BASH_SOURCE[1]}")/../.." && pwd
}

analysis_pipeline_eval_type() {
  case "$1" in
    failed_stay_depth) echo failed_stay ;;
    failed_update_depth) echo failed_update ;;
    noise_typology) echo failed_isolation ;;
    *)
      echo "[analysis-eval] unknown PIPELINE=$1" >&2
      return 1
      ;;
  esac
}

analysis_apply_model_preset() {
  local model="$1"
  case "$model" in
    7B)
      PYTHON_BIN="$BELIEF_TRAINING_ENV/bin/python"
      BASE_MODEL_PATH="${BASE_MODEL_7B_PATH:-models/Qwen2.5-7B-Instruct}"
      MAX_MODEL_LEN="30000"
      AGENT_TEMPERATURE="0.3"
      AGENT_MAX_TOKENS="1024"
      PROMPT_ENHANCEMENT="${PROMPT_ENHANCEMENT:-false}"
      ENABLE_THINKING="false"
      SAMPLING_TOP_P=""
      SAMPLING_TOP_K=""
      SAMPLING_PRESENCE_PENALTY=""
      SAMPLING_REPETITION_PENALTY=""
      ;;
    9B)
      PYTHON_BIN="$BELIEF_TRAINING_ENV/bin/python"
      export LD_LIBRARY_PATH="$BELIEF_TRAINING_ENV/lib"
      BASE_MODEL_PATH="${BASE_MODEL_9B_PATH:-models/Qwen3.5-9B}"
      MAX_MODEL_LEN="200000"
      AGENT_TEMPERATURE="1.0"
      AGENT_MAX_TOKENS="50000"
      PROMPT_ENHANCEMENT="${PROMPT_ENHANCEMENT:-false}"
      ENABLE_THINKING="true"
      SAMPLING_TOP_P="1.0"
      SAMPLING_TOP_K="20"
      SAMPLING_PRESENCE_PENALTY="1.5"
      SAMPLING_REPETITION_PENALTY="1.0"
      ;;
    *)
      echo "[analysis-eval] unsupported MODEL=$model (use 7B or 9B)" >&2
      return 1
      ;;
  esac
}

analysis_apply_api_preset() {
  PYTHON_BIN="${API_PYTHON_BIN:-$BELIEF_TRAINING_ENV/bin/python}"
  # These vLLM-only values are still passed because the reused parser accepts them,
  # but the API evaluation path does not instantiate vLLM or use them.
  BASE_MODEL_PATH="${API_BASE_MODEL_PATH:-api-backend}"
  MAX_MODEL_LEN="${API_MAX_MODEL_LEN:-49152}"
  MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
  PROMPT_ENHANCEMENT="${PROMPT_ENHANCEMENT:-false}"
  ENABLE_THINKING="$API_ENABLE_THINKING"
  AGENT_TEMPERATURE="$API_AGENT_TEMPERATURE"
  AGENT_MAX_TOKENS="$API_AGENT_MAX_TOKENS"
  SAMPLING_TOP_P="$API_SAMPLING_TOP_P"
  SAMPLING_TOP_K="$API_SAMPLING_TOP_K"
  SAMPLING_PRESENCE_PENALTY="$API_SAMPLING_PRESENCE_PENALTY"
  SAMPLING_REPETITION_PENALTY="$API_SAMPLING_REPETITION_PENALTY"
}

analysis_resolve_eval_script() {
  local scenario="$1"
  local model="$2"
  if [[ "$model" == "7B" ]]; then
    echo "analysis/eval/scenario_${scenario}_eval_7b_test_cases_vllm.py"
  else
    echo "analysis/eval/scenario_${scenario}_eval_test_cases_vllm.py"
  fi
}

analysis_build_eval_args() {
  local root="$1"
  local eval_script="$2"
  EVAL_ARGS=(
    "$root/$eval_script"
    --test-data "$TEST_DATA"
    --output-dir "$OUTPUT_DIR"
    --eval-target "$EVAL_TARGET"
    --repeats "$REPEATS"
    --eval-types "$EVAL_TYPE"
    --enable-thinking "$ENABLE_THINKING"
  )

  if [[ "$EVAL_TARGET" != "api" ]]; then
    EVAL_ARGS+=(
      --base-model-path "$BASE_MODEL_PATH"
      --lora-source-type "$LORA_SOURCE_TYPE"
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
      --max-model-len "$MAX_MODEL_LEN"
      --model-dtype "$MODEL_DTYPE"
    )
  fi

  if [[ -n "$AGENT_TEMPERATURE" ]]; then
    EVAL_ARGS+=(--agent-temperature "$AGENT_TEMPERATURE")
  fi
  if [[ -n "$AGENT_MAX_TOKENS" ]]; then
    EVAL_ARGS+=(--agent-max-tokens "$AGENT_MAX_TOKENS")
  fi

  if [[ "$MAX_CASES" != "0" ]]; then
    EVAL_ARGS+=(--max-cases "$MAX_CASES")
  fi

  if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "true" ]]; then
    EVAL_ARGS+=(--disable-custom-all-reduce)
  fi
  if [[ -n "$SAMPLING_TOP_P" ]]; then
    EVAL_ARGS+=(--sampling-top-p "$SAMPLING_TOP_P")
  fi
  if [[ -n "$SAMPLING_TOP_K" ]]; then
    EVAL_ARGS+=(--sampling-top-k "$SAMPLING_TOP_K")
  fi
  if [[ -n "$SAMPLING_PRESENCE_PENALTY" ]]; then
    EVAL_ARGS+=(--sampling-presence-penalty "$SAMPLING_PRESENCE_PENALTY")
  fi
  if [[ -n "$SAMPLING_REPETITION_PENALTY" ]]; then
    EVAL_ARGS+=(--sampling-repetition-penalty "$SAMPLING_REPETITION_PENALTY")
  fi
  if [[ "$LORA_SOURCE_TYPE" == "merged" && -n "$LORA_PATH" ]]; then
    EVAL_ARGS+=(--lora-model-path "$LORA_PATH")
  fi
  if [[ "$LORA_SOURCE_TYPE" == "adapter" && -n "$LORA_PATH" ]]; then
    EVAL_ARGS+=(--lora-adapter-path "$LORA_PATH")
  fi
  if [[ -n "$SEED" ]]; then
    EVAL_ARGS+=(--seed "$SEED")
  fi
  if [[ "$PROMPT_ENHANCEMENT" == "true" ]]; then
    EVAL_ARGS+=(--enable-prompt-enhancement)
  fi
}

analysis_run_curve_summary() {
  local pipeline="$1"
  local cases_dir="$2"
  local curve_name
  case "$pipeline" in
    failed_stay_depth) curve_name=failed_stay_depth_curve_summary.json ;;
    failed_update_depth) curve_name=failed_update_depth_curve_summary.json ;;
    *) return 0 ;;
  esac
  "$PYTHON_BIN" -m analysis.eval.analyze_turn_curves \
    --eval-dir "$OUTPUT_DIR" \
    --augmented-cases-dir "$cases_dir" \
    --output "$OUTPUT_DIR/$curve_name"
  echo "[analysis-eval] curve summary -> $OUTPUT_DIR/$curve_name"
}

analysis_run_category_breakdown() {
  local cases_dir="$1"
  "$PYTHON_BIN" -m analysis.eval.summarize_augmented_categories \
    --eval-dir "$OUTPUT_DIR" \
    --augmented-cases-dir "$cases_dir" \
    --output "$OUTPUT_DIR/category_breakdown_summary.json"
  echo "[analysis-eval] category breakdown -> $OUTPUT_DIR/category_breakdown_summary.json"
}
