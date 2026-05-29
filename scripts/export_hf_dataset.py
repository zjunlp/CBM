"""Export BeliefTrack cases into a Hugging Face Dataset-friendly layout."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


DATASETS = [
    {
        "config": "task_a_7b",
        "task": "task_a",
        "task_name": "Rule Discovery",
        "model_setting": "7B",
        "train": "Task_A/7B/train/train_cases_7B.json",
        "test": "Task_A/7B/test/test_cases_7B.json",
    },
    {
        "config": "task_a_9b",
        "task": "task_a",
        "task_name": "Rule Discovery",
        "model_setting": "9B",
        "train": "Task_A/9B/train/train_cases_9B_thinking.json",
        "test": "Task_A/9B/test/test_cases_9B_thinking.json",
    },
    {
        "config": "task_b_7b",
        "task": "task_b",
        "task_name": "Circuit Diagnosis",
        "model_setting": "7B",
        "train": "Task_B/7B/train/train_cases_7B.json",
        "test": "Task_B/7B/test/test_cases_7B.json",
    },
    {
        "config": "task_b_9b",
        "task": "task_b",
        "task_name": "Circuit Diagnosis",
        "model_setting": "9B",
        "train": "Task_B/9B/train/train_cases_9B_thinking.json",
        "test": "Task_B/9B/test/test_cases_9B_thinking.json",
    },
]

TOP_LEVEL_FIELDS = [
    "case_id",
    "task",
    "task_name",
    "model_setting",
    "split",
    "cbm_challenge_type",
    "oracle",
    "target_set_json",
    "system_prompt",
    "turns_json",
    "messages_json",
    "gt_survivors_json",
    "selected_prompt",
    "selected_turn",
    "train_kind",
    "source_category",
    "source_file",
    "source_repeat_index",
    "source_repeat_count",
]

STRING_FIELDS = {
    "case_id",
    "task",
    "task_name",
    "model_setting",
    "split",
    "cbm_challenge_type",
    "oracle",
    "system_prompt",
    "selected_prompt",
    "train_kind",
    "source_category",
    "source_file",
}
INTEGER_FIELDS = {
    "selected_turn",
    "source_repeat_index",
    "source_repeat_count",
}
JSON_STRING_FIELDS = {
    "target_set_json",
    "turns_json",
    "messages_json",
    "gt_survivors_json",
}
JSON_SOURCE_FIELDS = {
    "target_set_json": "target_set",
    "turns_json": "turns",
    "messages_json": "messages",
    "gt_survivors_json": "gt_survivors",
}


def load_cases(path: Path) -> List[Mapping[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list, got {type(data).__name__}")
    return data


def normalize_source_file(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if text.startswith("data/"):
        return text
    return text


def normalize_case(
    case: Mapping[str, Any],
    *,
    task: str,
    task_name: str,
    model_setting: str,
    split: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for field in TOP_LEVEL_FIELDS:
        if field in STRING_FIELDS:
            row[field] = ""
        elif field in INTEGER_FIELDS:
            row[field] = -1
        elif field in JSON_STRING_FIELDS:
            row[field] = "[]"
        else:
            row[field] = None
    row.update({
        "task": task,
        "task_name": task_name,
        "model_setting": model_setting,
        "split": split,
    })
    for field in TOP_LEVEL_FIELDS:
        if field in case:
            value = case[field]
            if value is None:
                continue
            row[field] = value
    for out_field, source_field in JSON_SOURCE_FIELDS.items():
        row[out_field] = json.dumps(case.get(source_field, []), ensure_ascii=False)
    row["source_file"] = normalize_source_file(row.get("source_file"))
    return row


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            f.write("\n")
            count += 1
    return count


def summarize_rows(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    by_type = Counter(str(row.get("cbm_challenge_type")) for row in rows)
    return {
        "num_rows": len(rows),
        "counts_by_cbm_challenge_type": dict(sorted(by_type.items())),
    }


def write_dataset_card(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(
        """---
language:
- en
license: apache-2.0
task_categories:
- text-generation
pretty_name: BeliefTrack
configs:
- config_name: task_a_7b
  data_files:
  - split: train
    path: task_a_7b/train.jsonl
  - split: test
    path: task_a_7b/test.jsonl
- config_name: task_a_9b
  data_files:
  - split: train
    path: task_a_9b/train.jsonl
  - split: test
    path: task_a_9b/test.jsonl
- config_name: task_b_7b
  data_files:
  - split: train
    path: task_b_7b/train.jsonl
  - split: test
    path: task_b_7b/test.jsonl
- config_name: task_b_9b
  data_files:
  - split: train
    path: task_b_9b/train.jsonl
  - split: test
    path: task_b_9b/test.jsonl
---

# BeliefTrack

[📄 arXiv](https://arxiv.org/abs/2605.30219) • [🤗 HFPaper](https://huggingface.co/papers/2605.30219) • [🤗 HF Collection](https://huggingface.co/collections/zjunlp/contextualbeliefmanagement)

BeliefTrack is a closed-world benchmark for **Contextual Belief Management (CBM)** in multi-turn language model interactions. Each example asks a model to maintain the set of hypotheses that remain consistent with formal evidence.

## Configurations

| Config | Task | Model Setting | Splits |
|---|---|---|---|
| `task_a_7b` | Rule Discovery | 7B experiment setting | train/test |
| `task_a_9b` | Rule Discovery | 9B thinking experiment setting | train/test |
| `task_b_7b` | Circuit Diagnosis | 7B experiment setting | train/test |
| `task_b_9b` | Circuit Diagnosis | 9B thinking experiment setting | train/test |

The `7B` and `9B` names identify the experimental data-generation/evaluation setting used in the project. They do not restrict the dataset to those model sizes.

## Failure Modes

- `failed_stay`: redundant evidence should not change the belief state.
- `failed_update`: a correction should replace earlier formal evidence and trigger belief recomputation.
- `failed_isolation`: task-irrelevant noise should not affect the formal-evidence belief state.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("YOUR_ORG/BeliefTrack", "task_a_7b")
print(ds["train"][0])
```

You can also load directly from local files:

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files={
        "train": "task_a_7b/train.jsonl",
        "test": "task_a_7b/test.jsonl",
    },
)
```

## Schema

Each row contains:

- `case_id`: unique example id.
- `task`: `task_a` or `task_b`.
- `task_name`: human-readable task name.
- `model_setting`: `7B` or `9B`.
- `split`: `train` or `test`.
- `cbm_challenge_type`: `failed_stay`, `failed_update`, or `failed_isolation`.
- `system_prompt`: system instruction shown to the model.
- `turns_json`: JSON-encoded multi-turn formal-evidence trajectory with per-turn gold belief states when available.
- `messages_json`: JSON-encoded training-format messages when available.
- `target_set_json`, `oracle`, `gt_survivors_json`: symbolic verifier targets and labels when available.

## Dataset Summary

```json
"""
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + """
```

## Citation

```bibtex
@article{xu2026whenshouldmodelschange,
  title={When Should Models Change Their Minds? Contextual Belief Management in Large Language Models},
  author={Xu, Haoming and Xu, Weihong and Li, Zongrui and Wang, Mengru and Yao, Yunzhi and Wu, Chiyu and Shang, Jin and Gong, Yu and Deng, Shumin},
  journal={arXiv preprint arXiv:2605.30219},
  year={2026}
}
```
""",
        encoding="utf-8",
    )


def export_dataset(source_root: Path, output_root: Path, include_raw: bool) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    summary: Dict[str, Any] = {"configs": {}}
    for spec in DATASETS:
        config_dir = output_root / spec["config"]
        config_dir.mkdir(parents=True)
        summary["configs"][spec["config"]] = {}

        for split in ("train", "test"):
            source_path = source_root / spec[split]
            cases = load_cases(source_path)
            rows = [
                normalize_case(
                    case,
                    task=spec["task"],
                    task_name=spec["task_name"],
                    model_setting=spec["model_setting"],
                    split=split,
                )
                for case in cases
            ]
            out_path = config_dir / f"{split}.jsonl"
            write_jsonl(out_path, rows)
            summary["configs"][spec["config"]][split] = {
                "source": str(source_path),
                **summarize_rows(rows),
            }

    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_dataset_card(output_root / "README.md", summary)

    if include_raw:
        raw_dir = output_root / "raw"
        shutil.copytree(source_root, raw_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/belief_training_task_dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/belieftrack_hf"),
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also copy the original directory into output_root/raw.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        include_raw=args.include_raw,
    )


if __name__ == "__main__":
    main()
