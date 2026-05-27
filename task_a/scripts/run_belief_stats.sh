#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MODES=(
  failed_stay
  # failed_update
)

BACKEND="${BACKEND:-vllm}"
API_BASE_URL="${API_BASE_URL:-}"
API_MODEL="${API_MODEL:-qwen3.5-plus}"
API_KEY="${API_KEY:-}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"

AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-models/Qwen3.5-9B}"
GPUS="${GPUS:-1}"
VLLM_DTYPE="${VLLM_DTYPE:-bf16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"

# Determine model type for prompt/template selection.
MODEL_TYPE="local"
if [[ "$BACKEND" == "api" ]] && [[ "$API_MODEL" == "qwen3.5-plus" ]]; then
  MODEL_TYPE="api_qwen35"
elif [[ "$BACKEND" == "vllm" ]]; then
  MODEL_TYPE="local"
fi

# Use the full benchmark rule set to enlarge the candidate space and make
# post-convergence bookkeeping harder once post-turn predictions are hidden.
RULES=(
  exactly_two_positive
  at_least_two_numbers_gt_3
  median_positive
  sum_abs_lte_12
  sum_of_squares_gt_50
)
# post-4    mountain_or_valley endpoints_ascending_xor_middle_between small failed_stay ratio
TARGETS=(
  # mountain_or_valley
  # one_negative_xor_ascending
  # distinct_xor_same_parity
  # endpoints_ascending_xor_middle_between
  sum_even_xnor_product_negative
)
HELDOUT_RULES=(
  mountain_or_valley
  one_negative_xor_ascending
  distinct_xor_same_parity
  endpoints_ascending_xor_middle_between
  sum_even_xnor_product_negative
)

INCLUDE_HELDOUT="${INCLUDE_HELDOUT:-1}"
HELDOUT_SET="${HELDOUT_SET:-hard}"
POST_CONVERGENCE_INTERFERENCE_ROUNDS="${POST_CONVERGENCE_INTERFERENCE_ROUNDS:-2}"
EVIDENCE_PER_ROUND="${EVIDENCE_PER_ROUND:-2}"
PERTURB_ORACLE_RULE_PREDICTION_IN_POST="${PERTURB_ORACLE_RULE_PREDICTION_IN_POST:-0}"
HIDE_RULE_PREDICTIONS_IN_DRIFT_POST="${HIDE_RULE_PREDICTIONS_IN_DRIFT_POST:-1}"

SEED="${SEED:-42}"
NUM_RUNS="${NUM_RUNS:-50}"
REPEATS="${REPEATS:-3}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-16}"
MAX_ATTEMPTS_MULTIPLIER="${MAX_ATTEMPTS_MULTIPLIER:-20}"

TEMP="${TEMP:-1.0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-10240}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

if [[ "$BACKEND" == "api" ]]; then
  MODEL_TAG="${API_MODEL}"
else
  MODEL_TAG="$(basename "$AGENT_MODEL_PATH")"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-task_a/outputs/${MODEL_TAG}_belief_stats_backend=${BACKEND}_model=${MODEL_TYPE}_hard_heldout_${EVIDENCE_PER_ROUND}_evidence_post_${POST_CONVERGENCE_INTERFERENCE_ROUNDS}_5bench_5heldout_hide_post_preds_seed${SEED}_target_${TARGETS[0]}_gpu${GPUS}}"
mkdir -p "$OUTPUT_ROOT"

echo "=================================================================="
echo "task_a belief_stats sampling"
echo "  Backend:     $BACKEND"
echo "  Model type:  $MODEL_TYPE"
echo "  Model:       $MODEL_TAG"
echo "  Modes:       ${MODES[*]}"
echo "  GPUs:        $GPUS"
echo "  Num runs:    $NUM_RUNS per target"
echo "  Repeats:     $REPEATS"
echo "  Include heldout: $INCLUDE_HELDOUT"
echo "  Heldout set: $HELDOUT_SET"
echo "  Evidence per round: $EVIDENCE_PER_ROUND"
echo "  Post rounds: $POST_CONVERGENCE_INTERFERENCE_ROUNDS"
echo "  Hide post predictions: $HIDE_RULE_PREDICTIONS_IN_DRIFT_POST"
echo "  Temperature: $TEMP"
echo "  Seed:        $SEED"
echo "  Output root: $OUTPUT_ROOT"
echo "=================================================================="

for MODE in "${MODES[@]}"; do
  if [[ "$MODE" != "failed_update" && "$MODE" != "failed_stay" ]]; then
    echo "MODE must be failed_update or failed_stay; got: $MODE" >&2
    exit 1
  fi

  OUT_DIR="$OUTPUT_ROOT/$MODE"
  LOG_FILE="$OUTPUT_ROOT/${MODE}.log"
  mkdir -p "$OUT_DIR"

  CMD=(
    python -m task_a.experiments.belief_stats
    --mode "$MODE"
    --backend "$BACKEND"
    --model-type "$MODEL_TYPE"
    --rules "${RULES[@]}"
    --seed "$SEED"
    --num-runs "$NUM_RUNS"
    --repeats "$REPEATS"
    --preprocess-workers "$PREPROCESS_WORKERS"
    --max-attempts-multiplier "$MAX_ATTEMPTS_MULTIPLIER"
    --temperature "$TEMP"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --presence-penalty "$PRESENCE_PENALTY"
    --repetition-penalty "$REPETITION_PENALTY"
    --output-dir "$OUT_DIR"
    --heldout-set "$HELDOUT_SET"
    --evidence-per-round "$EVIDENCE_PER_ROUND"
    --post-convergence-interference-rounds "$POST_CONVERGENCE_INTERFERENCE_ROUNDS"
  )

  if [[ "$BACKEND" == "api" ]]; then
    CMD+=(--api-base-url "$API_BASE_URL" --api-key-env "$API_KEY_ENV")
    if [[ -n "$API_MODEL" ]]; then
      CMD+=(--api-model "$API_MODEL")
    fi
    if [[ -n "$API_KEY" ]]; then
      CMD+=(--api-key "$API_KEY")
    fi
  else
    CMD+=(
      --agent-model-path "$AGENT_MODEL_PATH"
      --gpus "$GPUS"
      --vllm-dtype "$VLLM_DTYPE"
      --max-model-len "$VLLM_MAX_MODEL_LEN"
      --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
    )
  fi

  if [[ "$INCLUDE_HELDOUT" == "1" ]]; then
    CMD+=(--include-heldout)
    if [[ "${#HELDOUT_RULES[@]}" -gt 0 ]]; then
      CMD+=(--heldout-rules "${HELDOUT_RULES[@]}")
    fi
  fi

  if [[ "${#TARGETS[@]}" -gt 0 ]]; then
    CMD+=(--targets "${TARGETS[@]}")
  fi

  if [[ "$PERTURB_ORACLE_RULE_PREDICTION_IN_POST" == "1" ]]; then
    CMD+=(--perturb-oracle-rule-prediction-in-post)
  else
    CMD+=(--no-perturb-oracle-rule-prediction-in-post)
  fi

  if [[ "$HIDE_RULE_PREDICTIONS_IN_DRIFT_POST" == "1" ]]; then
    CMD+=(--hide-rule-predictions-in-failed_stay-post)
  fi

  echo
  echo "[run][$MODE] ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
done

echo
echo "=================================================================="
echo "All modes complete. Per-mode summaries are saved as stats_report.json and summary.json."
echo "=================================================================="
