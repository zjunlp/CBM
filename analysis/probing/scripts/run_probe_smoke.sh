#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONDA_PREFIX=/disk4/xuweihong/envs/miniconda3/envs/swift
PYTHON_BIN="${PYTHON_BIN:-/disk4/xuweihong/envs/miniconda3/envs/swift/bin/python}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
GPU="${GPU:-2}"
MODEL_PATH="${MODEL_PATH:-/disk1/xuhaoming/models/Qwen3.5-9B}"

SOURCE_DIR="${SOURCE_DIR:-analysis/depth_and_noise/outputs/task_a/9B/failed_stay_depth/eval/base_smoke/base/failed_stay}"
OUT_ROOT="${OUT_ROOT:-analysis/probing/outputs/smoke}"
DATASET_PATH="${DATASET_PATH:-$OUT_ROOT/belief_probe_dataset.json}"
SUMMARY_PATH="${SUMMARY_PATH:-$OUT_ROOT/belief_probe_dataset.summary.json}"
RESULT_DIR="${RESULT_DIR:-$OUT_ROOT/ranking}"
SIMPLE_CASES_PATH="${SIMPLE_CASES_PATH:-$RESULT_DIR/probe_ranking_simple_cases.json}"

PROBE_LIMIT="${PROBE_LIMIT:-12}"
RUN_RANKING="${RUN_RANKING:-1}"

echo "[probe-smoke] source=${SOURCE_DIR}"
echo "[probe-smoke] dataset=${DATASET_PATH}"

"$PYTHON_BIN" analysis/probing/scripts/build_belief_probe_dataset.py \
  "$SOURCE_DIR" \
  --scenario all \
  --output "$DATASET_PATH" \
  --summary "$SUMMARY_PATH"

if [[ "$RUN_RANKING" != "1" ]]; then
  echo "[probe-smoke] RUN_RANKING=${RUN_RANKING}; skip ranking"
  exit 0
fi

echo "[probe-smoke] ranking limit=${PROBE_LIMIT} gpu=${GPU} model=${MODEL_PATH}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" analysis/probing/scripts/run_belief_probe_ranking.py \
  --input "$DATASET_PATH" \
  --output-dir "$RESULT_DIR" \
  --backend vllm \
  --model-path "$MODEL_PATH" \
  --limit "$PROBE_LIMIT" \
  --batch-size 1 \
  --max-tokens 256 \
  --max-model-len 30000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --overwrite

"$PYTHON_BIN" analysis/probing/scripts/export_simple_belief_probe_cases.py \
  --input "$RESULT_DIR/probe_ranking_results.json" \
  --output "$SIMPLE_CASES_PATH"

echo "[probe-smoke] done"
