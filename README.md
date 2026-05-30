<h1 align="center">Contextual Belief Management</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.30219">📄arXiv</a> •
  <a href="https://huggingface.co/papers/2605.30219">🤗HFPaper</a> •
  <a href="https://huggingface.co/collections/zjunlp/contextualbeliefmanagement">🤗HF Collection</a>
</p>

This repository provides the official implementation of our paper:

> **When Should Models Change Their Minds? Contextual Belief Management in Large Language Models**
>
> Haoming Xu, Weihong Xu, Zongrui Li, Mengru Wang, Yunzhi Yao, Chiyu Wu, Jin Shang, Yu Gong, Shumin Deng

---

## Repository Layout

```text
.
├── task_a/                         # Rule Discovery environment
│   ├── core/                       # Rules, environment, orchestrator
│   ├── experiments/                # Case generation and metrics
│   ├── scripts/                    # Training/evaluation launchers
│   └── training/                   # GRPO reward and vLLM evaluation code
├── task_b/                         # Circuit Diagnosis environment
│   ├── domain/                     # Faults, topologies, rule engine
│   ├── runtime/                    # Agent/environment/orchestrator
│   ├── templates/                  # Template banks and validation
│   ├── experiments/                # Case generation and metrics
│   ├── scripts/                    # Training/evaluation launchers
│   └── training/                   # GRPO reward and vLLM evaluation code
├── analysis/
│   ├── depth_and_noise/            # Positional-depth and noise-typology diagnostics
│   ├── probing/                    # Post-answer belief probing
│   └── steering/                   # Activation steering workflows
├── data/                           # Train/test cases and experiment outputs
├── utils/                          # Shared I/O and model backend utilities
├── environment.yml                 # Conda environment snapshot
└── requirements.txt                # Reserved for lightweight installs
```

## Installation

Clone the repository:

```bash
git clone https://github.com/zjunlp/CBM.git
cd CBM
```

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate belief_training
```

## Data

```bash
hf download zjunlp/BeliefTrackDataset \
  --repo-type=dataset \
  --local-dir ./data/
```

The released train/test cases are organized under `data/`:

Each test split is evaluated with a strict repeat protocol over:

- `failed_stay`
- `failed_update`
- `failed_isolation`

## Training

### Task A: Rule Discovery

```bash
MODEL=/path/to/Qwen3.5-9B \
DATASET=data/Task_A/9B/train/train_cases_9B_thinking.json \
OUTPUT_DIR=task_a/training/checkpoints_multi_turn_online_swift_grpo \
TRAIN_GPUS=2,3 \
VLLM_GPU=1 \
bash task_a/scripts/run_multi_turn_online_grpo_swift.sh
```

### Task B: Circuit Diagnosis

```bash
MODEL=/path/to/Qwen3.5-9B \
DATASET=data/Task_B/9B/train/train_cases_9B_thinking.json \
OUTPUT_DIR=task_b/training/checkpoints_multi_turn_online_swift_grpo \
TRAIN_GPUS=2,3 \
VLLM_GPU=1 \
bash task_b/scripts/run_multi_turn_online_grpo_swift.sh
```

## Evaluation

Evaluation runs each model over the three CBM splits and reports failure rates. The default repeat count is `REPEATS=3`.

### Task A

```bash
BASE_MODEL_PATH=/path/to/Qwen3.5-9B \
bash task_a/scripts/run_eval_9b_test_cases_vllm.sh
```

### Task B

```bash
BASE_MODEL_PATH=/path/to/Qwen3.5-9B \
bash task_b/scripts/run_eval_9b_test_cases_vllm.sh
```

Common output files include per-split trajectories, `stats_report.json`, and aggregate failure statistics under the script-configured `OUTPUT_DIRS`. Edit the corresponding script if you need to change `TEST_DATA`, `EVAL_TYPES`, `REPEATS`, LoRA paths, or output directories.

## Analysis

The analysis code is split into three independent workflows.

### Depth and Noise

`analysis/depth_and_noise/` contains positional-depth augmentation and noise-typology diagnostics.

```bash
TASK=task_a MODEL=9B MAX_SOURCE_CASES=1 \
bash analysis/depth_and_noise/scripts/run_failed_stay_depth.sh

TASK=task_a MODEL=9B MAX_SOURCE_CASES=1 \
bash analysis/depth_and_noise/scripts/run_failed_update_depth.sh

TASK=task_a MODEL=9B MAX_SOURCE_CASES=1 \
bash analysis/depth_and_noise/scripts/run_noise_typology.sh
```

For a small end-to-end smoke test:

```bash
SOURCE_CASES=1 EVAL_CASES=1 REPEATS=1 \
bash analysis/depth_and_noise/scripts/run_base_smoke_7b_9b_tasks.sh
```

More details are in `analysis/depth_and_noise/README.md`.

### Probing

`analysis/probing/` builds post-answer belief-probe datasets and runs ranking-style probes.

```bash
python analysis/probing/scripts/build_belief_probe_dataset.py \
  data/Task_A/9B/test/failed_stay \
  --scenario a

python analysis/probing/scripts/run_belief_probe_ranking.py \
  --input analysis/probing/outputs/belief_probe_dataset_task_a.json \
  --output-dir analysis/probing/outputs/task_a/9B/probing/base
```

More details are in `analysis/probing/README.md`.

### Steering

`analysis/steering/` contains activation-steering workflows. EasySteer is not stored as a Git submodule in this repository. Install it as a separate local checkout and point the steering scripts to that checkout.

To create an EasySteer environment and clone/install EasySteer:

```bash
REPO_PARENT_DIR=/path/to/repos \
ENV_NAME=easysteer_test \
bash analysis/steering/scripts/setup_easysteer_env.sh
```

The setup script creates or reuses the conda environment, installs `vllm==0.17.1`, installs `cuda-toolkit=12.8`, upgrades `transformers`, clones EasySteer with submodules into `$REPO_PARENT_DIR/EasySteer`, installs EasySteer in editable mode, applies the Qwen3.5 compatibility patch, and writes the local path config.

The EasySteer path is configured by `analysis/steering/easysteer_config.json`:

```json
{
  "easysteer_root": "/path/to/repos/EasySteer"
}
```

You can also override the configured path for a single run:

```bash
export EASYSTEER_ROOT=/path/to/repos/EasySteer
```

## 🙏 Acknowledgements
We would like to express our heartfelt gratitude for the contribution of [ms-swift](https://github.com/modelscope/ms-swift),  [VLLM](https://github.com/vllm-project/vllm), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [EasySteer](https://github.com/ZJU-REAL/EasySteer) to our project, as we have utilized portions of their source code in our project.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{xu2026whenshouldmodelschange,
  title={When Should Models Change Their Minds? Contextual Belief Management in Large Language Models},
  author={Haoming Xu and Weihong Xu and Zongrui Li and Mengru Wang and Yunzhi Yao and Chiyu Wu and Jin Shang and Yu Gong and Shumin Deng},
  journal={arXiv preprint arXiv:2605.30219},
  year={2026}
}
```
