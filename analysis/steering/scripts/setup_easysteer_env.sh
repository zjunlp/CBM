#!/bin/bash
# Set up an EasySteer conda environment and clone/install EasySteer.

set -euo pipefail

# User configuration. Set this to the directory where EasySteer should be cloned.
REPO_PARENT_DIR="${REPO_PARENT_DIR:-/Path/To/repos}"

ENV_NAME="${ENV_NAME:-easysteer_test}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VLLM_VERSION="${VLLM_VERSION:-0.17.1}"
TRANSFORMERS_PACKAGE="${TRANSFORMERS_PACKAGE:-transformers}"
EASYSTEER_REPO_URL="${EASYSTEER_REPO_URL:-https://github.com/ZJU-REAL/EasySteer.git}"
VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-95c0f928cdeeaa21c4906e73cee6a156e1b3b995}"

usage() {
  cat <<EOF
Usage: $0

Edit REPO_PARENT_DIR near the top of this script, or override variables at runtime:

  REPO_PARENT_DIR=/path/to/repos ENV_NAME=easysteer_test $0

Environment overrides:
  REPO_PARENT_DIR                  default: /Path/To/repos
  ENV_NAME                         default: easysteer_test
  PYTHON_VERSION                   default: 3.10
  VLLM_VERSION                     default: 0.17.1
  TRANSFORMERS_PACKAGE             default: transformers
  EASYSTEER_REPO_URL               default: https://github.com/ZJU-REAL/EasySteer.git
  VLLM_PRECOMPILED_WHEEL_COMMIT    default: 95c0f928cdeeaa21c4906e73cee6a156e1b3b995
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 0 ]]; then
  usage >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[easysteer-setup] ERROR: conda command not found." >&2
  exit 1
fi

REPO_PARENT_DIR="$(realpath -m "$REPO_PARENT_DIR")"
EASYSTEER_DIR="$REPO_PARENT_DIR/EasySteer"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[easysteer-setup] conda env exists: $ENV_NAME"
else
  echo "[easysteer-setup] creating conda env: $ENV_NAME python=$PYTHON_VERSION"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

echo "[easysteer-setup] python=$(command -v python)"
echo "[easysteer-setup] installing vllm==$VLLM_VERSION"
python -m pip install "vllm==$VLLM_VERSION"

echo "[easysteer-setup] installing cuda-toolkit=12.8 from nvidia channel"
conda install -c nvidia cuda-toolkit=12.8 -y

echo "[easysteer-setup] upgrading transformers package: $TRANSFORMERS_PACKAGE"
python -m pip install --upgrade "$TRANSFORMERS_PACKAGE"
python - <<'PY'
import transformers
print(f"[easysteer-setup] transformers={transformers.__version__}")
PY

if [[ -d "$EASYSTEER_DIR/.git" ]]; then
  echo "[easysteer-setup] EasySteer repo exists: $EASYSTEER_DIR"
else
  if [[ -e "$EASYSTEER_DIR" ]]; then
    echo "[easysteer-setup] ERROR: target exists but is not a git repo: $EASYSTEER_DIR" >&2
    exit 1
  fi
  mkdir -p "$REPO_PARENT_DIR"
  echo "[easysteer-setup] cloning EasySteer into: $EASYSTEER_DIR"
  git clone --recurse-submodules "$EASYSTEER_REPO_URL" "$EASYSTEER_DIR"
fi

echo "[easysteer-setup] installing EasySteer/vllm-steer"
cd "$EASYSTEER_DIR/vllm-steer"
export VLLM_PRECOMPILED_WHEEL_COMMIT
VLLM_USE_PRECOMPILED=1 python -m pip install --editable .

echo "[easysteer-setup] patching EasySteer for Qwen3.5 decoder layer"
cd "$OLDPWD"
EASYSTEER_ROOT="$EASYSTEER_DIR" bash analysis/steering/scripts/patch_easysteer_qwen35.sh

echo "[easysteer-setup] installing EasySteer"
cd "$EASYSTEER_DIR"
python -m pip install --editable .

echo "[easysteer-setup] writing steering EasySteer config"
cd "$OLDPWD"
python - "$EASYSTEER_DIR" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path("analysis/steering/easysteer_config.json")
config_path.write_text(
    json.dumps({"easysteer_root": sys.argv[1]}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"[easysteer-setup] wrote {config_path}")
PY

echo "[easysteer-setup] final versions"
python - <<'PY'
import transformers
print(f"transformers={transformers.__version__}")
try:
    import vllm
    print(f"vllm={getattr(vllm, '__version__', 'unknown')} {getattr(vllm, '__file__', '')}")
except Exception as exc:
    print(f"vllm import failed: {exc!r}")
PY

echo "[easysteer-setup] done"
echo "[easysteer-setup] env: $ENV_NAME"
echo "[easysteer-setup] repo: $EASYSTEER_DIR"
