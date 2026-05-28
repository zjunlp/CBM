#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
TASK="${TASK:-task_a}"
MODEL="${MODEL:-7B}"
SEED="${SEED:-42}"
SOURCE_LIMIT="${MAX_SOURCE_CASES:-${MAX_CASES:-0}}"
EXTRA_ARGS=()
if [[ "$SOURCE_LIMIT" != "0" ]]; then
  EXTRA_ARGS+=(--max-source-cases "$SOURCE_LIMIT")
fi
python -m analysis.depth_and_noise.augment.cli \
  --config analysis/depth_and_noise/configs/failed_update_depth.yaml \
  --pipeline failed_update_depth \
  --task "$TASK" \
  --model "$MODEL" \
  --challenge-type failed_update \
  --seed "$SEED" \
  --overwrite \
  "${EXTRA_ARGS[@]}" \
  "$@"
