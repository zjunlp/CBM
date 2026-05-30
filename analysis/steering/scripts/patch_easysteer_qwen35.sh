#!/bin/bash
# Patch local EasySteer/vllm-steer for Qwen3.5 smoke runs.
#
# Fixes:
# 1. vLLM worker processes recognize Qwen3.5 decoder layers.
# 2. Steer-vector injection returns hidden states in the model's original dtype.

set -euo pipefail

EASYSTEER_ROOT="${EASYSTEER_ROOT:-/Path/To/EasySteer}"
CONFIG_PATH="$EASYSTEER_ROOT/vllm-steer/vllm/steer_vectors/config.py"
LAYERS_PATH="$EASYSTEER_ROOT/vllm-steer/vllm/steer_vectors/layers.py"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[patch-easysteer] ERROR: config not found: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$LAYERS_PATH" ]]; then
  echo "[patch-easysteer] ERROR: layers not found: $LAYERS_PATH" >&2
  exit 1
fi

python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

layers = [
    "Qwen3_5DecoderLayer",
]

changed = False
for layer in layers:
    if f'"{layer}"' in text:
        print(f"[patch-easysteer] already present: {layer}")
        continue
    marker = '    "Qwen3NextDecoderLayer",\n'
    if marker not in text:
        raise SystemExit(f"[patch-easysteer] insertion marker not found in {path}: {marker!r}")
    text = text.replace(marker, marker + f'    "{layer}",\n')
    changed = True
    print(f"[patch-easysteer] added: {layer}")

if changed:
    path.write_text(text, encoding="utf-8")
print(f"[patch-easysteer] patched: {path}")
PY

python - "$LAYERS_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

needle = "        # Apply algorithm transformation\n        modified_complete_hidden_states = active_algo.apply_intervention(complete_hidden_states)\n"
replacement = (
    "        # Apply algorithm transformation\n"
    "        original_dtype = hidden_states.dtype\n"
    "        modified_complete_hidden_states = active_algo.apply_intervention(complete_hidden_states)\n"
    "        modified_complete_hidden_states = modified_complete_hidden_states.to(original_dtype)\n"
)

if "modified_complete_hidden_states = modified_complete_hidden_states.to(original_dtype)" in text:
    print("[patch-easysteer] already present: steer output dtype cast")
elif needle not in text:
    raise SystemExit(f"[patch-easysteer] insertion marker not found in {path}")
else:
    text = text.replace(needle, replacement)
    path.write_text(text, encoding="utf-8")
    print("[patch-easysteer] added: steer output dtype cast")

print(f"[patch-easysteer] patched: {path}")
PY
