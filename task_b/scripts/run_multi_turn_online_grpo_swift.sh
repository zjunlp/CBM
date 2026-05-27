#!/bin/bash
# Task B 9B multi-turn GRPO launcher using native Swift RLHF training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_PREFIX="${CONDA_PREFIX:-/disk4/xuweihong/envs/miniconda3/envs/swift}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_COMPILE_DISABLE=1
export MASTER_ADDR=127.0.0.1
export NCCL_DEBUG=WARN

SWIFT_BIN=$CONDA_PREFIX/bin/swift
PYTHON_BIN=$CONDA_PREFIX/bin/python

MODEL=${MODEL:-/disk1/xuhaoming/models/Qwen3.5-9B}
DATASET=${DATASET:-data/task_b_9B_train_cases/train_cases_9B_thinking.json}
OUTPUT_DIR=${OUTPUT_DIR:-task_b/training/checkpoints_multi_turn_online_swift_grpo_9B_thinking_rollout_8}
REWARD_PLUGIN=${REWARD_PLUGIN:-task_b/training/reward.py}
REWARD_FUNC=${REWARD_FUNC:-task_b_belief_reward}
USE_TEMPLATE_PATCH=${USE_TEMPLATE_PATCH:-1}
TEMPLATE_TYPE_PATCHED=${TEMPLATE_TYPE_PATCHED:-qwen3_5_keep_history_think}
TEMPLATE_TYPE_DEFAULT=${TEMPLATE_TYPE_DEFAULT:-qwen3_5}
CUSTOM_REGISTER_PATH=${CUSTOM_REGISTER_PATH:-task_a/training/template_patch.py}

if [[ "$USE_TEMPLATE_PATCH" == "1" ]]; then
    TEMPLATE_TYPE="$TEMPLATE_TYPE_PATCHED"
    TEMPLATE_ARGS=(--template "$TEMPLATE_TYPE" --custom_register_path "$CUSTOM_REGISTER_PATH")
else
    TEMPLATE_TYPE="$TEMPLATE_TYPE_DEFAULT"
    TEMPLATE_ARGS=()
fi

TRAIN_GPUS=${TRAIN_GPUS:-4,5,6,7}
VLLM_GPU=${VLLM_GPU:-3}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}

NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
STEPS_PER_GENERATION=${STEPS_PER_GENERATION:-4}
NUM_ITERATIONS=${NUM_ITERATIONS:-1}
MAX_LENGTH=${MAX_LENGTH:-50000}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-25000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
BETA=${BETA:-0.04}

TUNER_TYPE=${TUNER_TYPE:-lora}
TARGET_MODULES=${TARGET_MODULES:-all-linear}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

USE_VLLM=${USE_VLLM:-1}
VLLM_MODE=${VLLM_MODE:-server}
VLLM_SERVER_HOST=${VLLM_SERVER_HOST:-127.0.0.1}
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8012}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING:-false}
VLLM_SERVER_TIMEOUT=${VLLM_SERVER_TIMEOUT:-600}

DEEPSPEED=${DEEPSPEED:-zero2}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-20}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}

SAVE_NUM=${SAVE_NUM:-12}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-12}
LOGGING_STEPS=${LOGGING_STEPS:-1}
SEED=${SEED:-42}

REPORT_TO=(${REPORT_TO:-swanlab})
SWANLAB_PROJECT=${SWANLAB_PROJECT:-belief_training_task_b}
SWANLAB_EXP_NAME=${SWANLAB_EXP_NAME:-swift_grpo_9B_zero2}

LOGDIR=${LOGDIR:-task_b/outputs/multi_turn_online_swift_grpo_logs}
TRAIN_LOG=${TRAIN_LOG:-$LOGDIR/train.log}
VLLM_LOG=${VLLM_LOG:-$LOGDIR/vllm_server.log}

mkdir -p "$LOGDIR"

CHECK_PATHS=("$SWIFT_BIN" "$PYTHON_BIN" "$MODEL" "$DATASET" "$REWARD_PLUGIN")
if [[ "$USE_TEMPLATE_PATCH" == "1" ]]; then
    CHECK_PATHS+=("$CUSTOM_REGISTER_PATH")
fi
for path in "${CHECK_PATHS[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "[swift-grpo] ERROR: path not found: $path"
        exit 1
    fi
done

port_is_listening() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
}

IFS=',' read -r -a TRAIN_GPU_ARR <<< "$TRAIN_GPUS"
NUM_TRAIN_PROCS=${#TRAIN_GPU_ARR[@]}
if (( NUM_TRAIN_PROCS == 0 )); then
    echo "[swift-grpo] ERROR: TRAIN_GPUS is empty"
    exit 1
fi
if [[ "$USE_VLLM" == "1" ]]; then
    for gpu in "${TRAIN_GPU_ARR[@]}"; do
        if [[ "$gpu" == "$VLLM_GPU" ]]; then
            echo "[swift-grpo] ERROR: VLLM_GPU ($VLLM_GPU) overlaps with TRAIN_GPUS ($TRAIN_GPUS)"
            exit 1
        fi
    done
fi

echo "[swift-grpo] model=$MODEL"
echo "[swift-grpo] dataset=$DATASET"
echo "[swift-grpo] output_dir=$OUTPUT_DIR"
echo "[swift-grpo] train_gpus=$TRAIN_GPUS nproc=$NUM_TRAIN_PROCS master_port=$MAIN_PROCESS_PORT"
echo "[swift-grpo] use_vllm=$USE_VLLM vllm_mode=$VLLM_MODE vllm_gpu=$VLLM_GPU server=$VLLM_SERVER_HOST:$VLLM_SERVER_PORT"
if [[ "$USE_TEMPLATE_PATCH" == "1" ]]; then
    echo "[swift-grpo] template=$TEMPLATE_TYPE custom_register_path=$CUSTOM_REGISTER_PATH reward_func=$REWARD_FUNC"
else
    echo "[swift-grpo] template=$TEMPLATE_TYPE custom_register_path=disabled reward_func=$REWARD_FUNC"
fi
echo "[swift-grpo] epochs=$NUM_TRAIN_EPOCHS bs=$PER_DEVICE_TRAIN_BATCH_SIZE grad_accum=$GRADIENT_ACCUMULATION_STEPS num_generations=$NUM_GENERATIONS steps_per_generation=$STEPS_PER_GENERATION num_iterations=$NUM_ITERATIONS"
echo "[swift-grpo] max_length=$MAX_LENGTH max_completion_length=$MAX_COMPLETION_LENGTH"
echo "[swift-grpo] tuner_type=$TUNER_TYPE target_modules=$TARGET_MODULES lora_r=$LORA_R"
echo "[swift-grpo] deepspeed=${DEEPSPEED:-disabled} gradient_checkpointing=$GRADIENT_CHECKPOINTING"
echo "[swift-grpo] keep_history_think_template=$USE_TEMPLATE_PATCH"

DATASET_SIZE=$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
path = Path("$DATASET")
data = json.loads(path.read_text())
print(len(data))
PY
)
GLOBAL_MICRO_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_TRAIN_PROCS))
STEPS_PER_EPOCH=$((DATASET_SIZE / GLOBAL_MICRO_BATCH_SIZE))
if (( STEPS_PER_EPOCH < 1 )); then
    STEPS_PER_EPOCH=1
fi
UPDATE_STEPS_PER_EPOCH=$((STEPS_PER_EPOCH / GRADIENT_ACCUMULATION_STEPS))
if (( UPDATE_STEPS_PER_EPOCH < 1 )); then
    UPDATE_STEPS_PER_EPOCH=1
fi
ROLLOUT_REUSE_FACTOR=$((STEPS_PER_GENERATION * NUM_ITERATIONS))
if (( ROLLOUT_REUSE_FACTOR < 1 )); then
    ROLLOUT_REUSE_FACTOR=1
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
    TOTAL_TRAIN_STEPS=$MAX_STEPS
else
    TOTAL_TRAIN_STEPS=$((NUM_TRAIN_EPOCHS * UPDATE_STEPS_PER_EPOCH * ROLLOUT_REUSE_FACTOR))
fi
if (( TOTAL_TRAIN_STEPS < 1 )); then
    TOTAL_TRAIN_STEPS=1
fi
SAVE_STEPS=$(((TOTAL_TRAIN_STEPS + SAVE_NUM - 1) / SAVE_NUM))
if (( SAVE_STEPS < 1 )); then
    SAVE_STEPS=1
fi
echo "[swift-grpo] dataset_size=$DATASET_SIZE update_steps_per_epoch=$UPDATE_STEPS_PER_EPOCH rollout_reuse_factor=$ROLLOUT_REUSE_FACTOR total_train_steps=$TOTAL_TRAIN_STEPS save_num=$SAVE_NUM save_steps=$SAVE_STEPS save_total_limit=$SAVE_TOTAL_LIMIT"

VLLM_SERVER_PID=
TRAIN_PID=
CLEANED_UP=0
force_kill_port() {
    local port="$1"
    local label="$2"

    if [[ -z "$port" ]]; then
        return
    fi

    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | xargs -r kill -TERM 2>/dev/null || true
        sleep 1
        lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
    fi

    "$PYTHON_BIN" - "$port" <<'PY' 2>/dev/null || true
import signal
import sys
import time

port = int(sys.argv[1])

try:
    import psutil
except Exception:
    sys.exit(0)

pids = set()
for conn in psutil.net_connections(kind='tcp'):
    try:
        if conn.laddr and conn.laddr.port == port and conn.pid is not None:
            pids.add(conn.pid)
    except Exception:
        continue

for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in sorted(pids):
        try:
            proc = psutil.Process(pid)
        except psutil.Error:
            continue
        procs = proc.children(recursive=True)
        procs.append(proc)
        for p in reversed(procs):
            try:
                p.send_signal(sig)
            except psutil.Error:
                pass
    time.sleep(1)
PY

    if port_is_listening "$VLLM_SERVER_HOST" "$port"; then
        echo "[swift-grpo] WARNING: $label port still in use after cleanup: $VLLM_SERVER_HOST:$port"
    fi
}

stop_process_tree() {
    local pid="$1"
    local label="$2"
    local pattern="${3:-}"
    local port="${4:-}"
    local pgid
    local pid_alive=0

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        pid_alive=1
    fi

    if [[ "$pid_alive" == "1" ]]; then
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        echo "[swift-grpo] Stopping $label pid=$pid pgid=${pgid:-unknown}"

        if [[ -n "$pgid" ]]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true

        sleep 3

        if [[ -n "$pgid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- "-$pgid" 2>/dev/null || true
        fi
        pkill -KILL -P "$pid" 2>/dev/null || true
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi

    if [[ -n "$pattern" ]] || [[ -n "$port" ]]; then
        echo "[swift-grpo] Recursive fallback cleanup for $label pattern=${pattern:-none} port=${port:-none}"
        "$PYTHON_BIN" - "${pattern:-}" "${port:-}" <<'PY' 2>/dev/null || true
import os
import signal
import sys
import time

pattern = sys.argv[1]
port = sys.argv[2]
port = int(port) if port else None

try:
    import psutil
except Exception:
    sys.exit(0)


def kill_tree(root_pid: int) -> None:
    try:
        proc = psutil.Process(root_pid)
    except psutil.Error:
        return
    procs = proc.children(recursive=True)
    procs.append(proc)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in reversed(procs):
            try:
                p.send_signal(sig)
            except psutil.Error:
                pass
        gone, alive = psutil.wait_procs(procs, timeout=2)
        if not alive:
            return
        procs = alive


matched_pids = set()

if pattern:
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if cmdline and pattern in cmdline:
            matched_pids.add(proc.info['pid'])

if port is not None:
    for conn in psutil.net_connections(kind='tcp'):
        try:
            if not conn.laddr or conn.laddr.port != port or conn.pid is None:
                continue
        except Exception:
            continue
        matched_pids.add(conn.pid)

for pid in sorted(matched_pids):
    kill_tree(pid)
PY
    fi
}
cleanup() {
    if [[ "$CLEANED_UP" == "1" ]]; then
        return
    fi
    CLEANED_UP=1
    stop_process_tree "$TRAIN_PID" "training process group" "$SWIFT_BIN rlhf"
    stop_process_tree "$VLLM_SERVER_PID" "vLLM rollout server" "$SWIFT_BIN rollout --host $VLLM_SERVER_HOST --port $VLLM_SERVER_PORT" "$VLLM_SERVER_PORT"
    if [[ "$USE_VLLM" == "1" ]]; then
        force_kill_port "$VLLM_SERVER_PORT" "vLLM"
    fi
}
trap cleanup EXIT ERR INT TERM HUP QUIT

if [[ "$USE_VLLM" == "1" ]]; then
    if port_is_listening "$VLLM_SERVER_HOST" "$VLLM_SERVER_PORT"; then
        echo "[swift-grpo] ERROR: vLLM port already in use: $VLLM_SERVER_HOST:$VLLM_SERVER_PORT"
        exit 1
    fi

    echo "[swift-grpo] Starting vLLM rollout server on GPU $VLLM_GPU -> $VLLM_LOG"
    CUDA_VISIBLE_DEVICES=$VLLM_GPU setsid "$SWIFT_BIN" rollout \
        --model "$MODEL" \
        "${TEMPLATE_ARGS[@]}" \
        --infer_backend vllm \
        --host "$VLLM_SERVER_HOST" \
        --port "$VLLM_SERVER_PORT" \
        --torch_dtype bfloat16 \
        --max_model_len "$MAX_LENGTH" \
        --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
        --vllm_enable_prefix_caching "$VLLM_ENABLE_PREFIX_CACHING" \
        > "$VLLM_LOG" 2>&1 &
    VLLM_SERVER_PID=$!

    echo "[swift-grpo] Waiting for vLLM rollout server health check..."
    for attempt in $(seq 1 "$VLLM_SERVER_TIMEOUT"); do
        if "$PYTHON_BIN" -c "import urllib.request; urllib.request.urlopen('http://$VLLM_SERVER_HOST:$VLLM_SERVER_PORT/health/', timeout=2).read()" >/dev/null 2>&1; then
            echo "[swift-grpo] vLLM rollout server is ready after ${attempt}s"
            break
        fi
        if ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
            echo "[swift-grpo] ERROR: vLLM rollout server exited early. Last log lines:"
            tail -n 80 "$VLLM_LOG" || true
            exit 1
        fi
        if [[ "$attempt" == "$VLLM_SERVER_TIMEOUT" ]]; then
            echo "[swift-grpo] ERROR: timed out waiting for vLLM rollout server. Last log lines:"
            tail -n 80 "$VLLM_LOG" || true
            exit 1
        fi
        sleep 1
    done
fi

TRAIN_ARGS=(
    rlhf
    --rlhf_type grpo
    --model "$MODEL"
    "${TEMPLATE_ARGS[@]}"
    --dataset "$DATASET"
    --external_plugins "$REWARD_PLUGIN"
    --reward_funcs "$REWARD_FUNC"
    --output_dir "$OUTPUT_DIR"
    --torch_dtype bfloat16
    --attn_impl flash_attn
    --use_logits_to_keep true
    --dataset_shuffle true
    --tuner_type "$TUNER_TYPE"
    --target_modules "$TARGET_MODULES"
    --lora_rank "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --learning_rate "$LEARNING_RATE"
    --beta "$BETA"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --max_steps "${MAX_STEPS:--1}"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --num_generations "$NUM_GENERATIONS"
    --steps_per_generation "$STEPS_PER_GENERATION"
    --num_iterations "$NUM_ITERATIONS"
    --max_length "$MAX_LENGTH"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
    --repetition_penalty "$REPETITION_PENALTY"
    --logging_steps "$LOGGING_STEPS"
    --save_strategy steps
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --eval_strategy no
    --seed "$SEED"
    --log_completions true
    --report_to "${REPORT_TO[@]}"
    --swanlab_project "$SWANLAB_PROJECT"
    --swanlab_exp_name "$SWANLAB_EXP_NAME"
)

if [[ "$GRADIENT_CHECKPOINTING" == "true" ]]; then
    TRAIN_ARGS+=(--gradient_checkpointing true)
fi
if [[ -n "${DEEPSPEED:-}" ]]; then
    TRAIN_ARGS+=(--deepspeed "$DEEPSPEED")
fi
if [[ "$USE_VLLM" == "1" ]]; then
    TRAIN_ARGS+=(
        --use_vllm true
        --vllm_mode "$VLLM_MODE"
        --vllm_server_host "$VLLM_SERVER_HOST"
        --vllm_server_port "$VLLM_SERVER_PORT"
        --vllm_server_timeout "$VLLM_SERVER_TIMEOUT"
        --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
        --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS"
        --vllm_enable_prefix_caching "$VLLM_ENABLE_PREFIX_CACHING"
    )
fi

echo "[swift-grpo] Launching native Swift GRPO -> $TRAIN_LOG"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
NPROC_PER_NODE=$NUM_TRAIN_PROCS \
MASTER_PORT=$MAIN_PROCESS_PORT \
setsid "$SWIFT_BIN" "${TRAIN_ARGS[@]}" > >(tee "$TRAIN_LOG") 2>&1 &
TRAIN_PID=$!

TRAIN_STATUS=0
wait "$TRAIN_PID" || TRAIN_STATUS=$?
TRAIN_PID=

cleanup

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    echo "[swift-grpo] Training failed with exit code $TRAIN_STATUS"
    exit "$TRAIN_STATUS"
fi

echo "[swift-grpo] Training finished. Output: $OUTPUT_DIR"
