from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.depth_and_noise.eval.unified_classification_vllm import main
from task_a.training import eval_9b_test_cases_vllm as base


if __name__ == "__main__":
    main(base, __file__)
