#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TASK="${TASK:-task_a}"
MODEL="${MODEL:-7B}"
SEED="${SEED:-42}"
SOURCE_LIMIT="${MAX_SOURCE_CASES:-${MAX_CASES:-0}}"
EXTRA_ARGS=()
if [[ "$SOURCE_LIMIT" != "0" ]]; then
  EXTRA_ARGS+=(--max-source-cases "$SOURCE_LIMIT")
fi
python -m analysis.augment.cli \
  --config analysis/configs/failed_stay_depth.yaml \
  --pipeline failed_stay_depth \
  --task "$TASK" \
  --model "$MODEL" \
  --challenge-type failed_stay \
  --seed "$SEED" \
  --overwrite \
  "${EXTRA_ARGS[@]}" \
  "$@"
