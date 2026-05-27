# analysis: Positional Depth and Noise Typology

Offline data augmentation and curve analysis for BELIEFSHIFT diagnostic experiments. This directory is independent from `task_a/` / `task_b/` training code.

## Layout

```text
analysis/
├── augment/          # Data transformation pipelines
├── templates/noise/  # L3 noise template pools (A/B/C)
├── eval/             # FSR/FUR depth curve aggregation
├── configs/          # YAML experiment configs
└── outputs/          # Generated augmented cases (default)
```

## Pipelines

| Pipeline | Input `challenge_type` | Effect |
|----------|------------------------|--------|
| `rd_failed_stay_depth` | `failed_stay` | After lock, insert max-depth redundant consistent triples; summaries project configured depths |
| `rd_failed_update_depth` | `failed_update` | Delay CORRECTION by K oracle-neutral triples (survivor set unchanged) |
| `noise_typology` | `failed_isolation` | Replace/inject host comment by noise type |

Phase-2 CD depth (`analysis/augment/cd_depth.py`) is stubbed.

## Quick start

```bash
# Step 1: generate all augmented data
bash analysis/scripts/run_generate_all_analysis_data.sh

# Step 2a: eval one pipeline
PIPELINE=failed_stay_depth SCENARIO=a MODEL=9B CUDA_VISIBLE_DEVICES=6 \
  bash analysis/scripts/run_eval_analysis.sh

# Step 2b: eval all 12 datasets for one model
MODEL=9B CUDA_VISIBLE_DEVICES=6 bash analysis/scripts/run_eval_all_analysis.sh

# LoRA / merged checkpoint
MODEL=7B EVAL_TARGET=lora RUN_LABEL=ckpt403 \
  LORA_PATH=task_b/training/checkpoints/.../checkpoint-403_merged \
  CUDA_VISIBLE_DEVICES=2 bash analysis/scripts/run_eval_all_analysis.sh
```

Cases live under `analysis/outputs/task_{a,b}/{7B,9B}/{pipeline}/{failed_stay|failed_update|failed_isolation}/`.

### 7B vs 9B eval presets (`run_eval_analysis.sh`)

| Setting | 7B | 9B |
|---------|----|----|
| Eval script | `analysis/eval/task_*_eval_7b_test_cases_vllm.py` | `analysis/eval/task_*_eval_test_cases_vllm.py` |
| Base model | `Qwen2.5-7B-Instruct` | `Qwen3.5-9B` |
| temperature | 0.3 | 1.0 |
| max_tokens | 1024 | 30000 |
| max_model_len | 32000 | 81920 |
| prompt_enhancement | false | true |
| enable_thinking | false | auto |

Hyperparameters align with `task_a/scripts/run_eval_*_test_cases_vllm.sh`.

Eval outputs: `.../{pipeline}/eval/{run_label}/`  
Depth curves: `.../eval/{run_label}/failed_stay_depth_curve_summary.json` or `failed_update_depth_curve_summary.json`

## Data generation (per pipeline)

```bash
bash analysis/scripts/run_failed_stay_depth.sh      # SCENARIO=a MODEL=7B
bash analysis/scripts/run_failed_update_depth.sh
bash analysis/scripts/run_noise_typology.sh
```

Edit `analysis/configs/augment.yaml`, `failed_stay_depth.yaml`, or `failed_update_depth.yaml` for `n_redundant` / `delay_turns` grids.

## Eval and curves (manual)

```bash
python -m analysis.eval.analyze_turn_curves \
  --eval-dir analysis/outputs/task_a/9B/failed_stay_depth/eval/base \
  --augmented-cases-dir analysis/outputs/task_a/9B/failed_stay_depth/failed_stay \
  --output analysis/outputs/task_a/9B/failed_stay_depth/eval/base/failed_stay_depth_curve_summary.json
```

## Output contract

Augmented cases keep original fields and add:

```json
"augmentation": {
  "pipeline": "rd_failed_stay_depth",
  "version": "0.1.0",
  "params": { "n_redundant": 10, "seed": 42 },
  "source_case_id": "...",
  "lock_idx": 1
}
```

Each run also writes `manifest.jsonl` and `augment_summary.json` under the output directory.

## Data source

Input defaults to read-only `data/task_{a,b}/{7B|9B}/test/{failed_stay|failed_update|failed_isolation}/`.
