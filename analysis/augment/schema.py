"""Augmented case schema, naming, and manifest helpers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

AUGMENTATION_VERSION = "0.1.0"


def stable_case_hash(case: Dict[str, Any]) -> str:
    payload = json.dumps(case, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_augmented_case_id(source_case_id: str, pipeline: str, suffix: str) -> str:
    return f"{source_case_id}_{pipeline}_{suffix}"


def attach_augmentation(
    case: Dict[str, Any],
    *,
    pipeline: str,
    params: Dict[str, Any],
    source_case_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = deepcopy(case)
    out["case_id"] = params.get("case_id", out.get("case_id", source_case_id))
    augmentation: Dict[str, Any] = {
        "pipeline": pipeline,
        "version": AUGMENTATION_VERSION,
        "params": dict(params),
        "source_case_id": source_case_id,
    }
    if extra:
        augmentation.update(extra)
    out["augmentation"] = augmentation
    return out


def manifest_record(
    *,
    case_id: str,
    source_path: str,
    output_path: str,
    pipeline: str,
    params: Dict[str, Any],
    case_hash: str,
    status: str = "ok",
    skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "case_id": case_id,
        "source_path": source_path,
        "output_path": output_path,
        "pipeline": pipeline,
        "params": params,
        "sha256": case_hash,
        "status": status,
    }
    if skip_reason:
        record["skip_reason"] = skip_reason
    return record


def write_manifest_line(manifest_path: Path, record: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_case(path: Path, case: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stable_case_hash(case)


def iter_input_cases(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*.json")):
        if path.name in {"summary.json", "augment_summary.json", "stats_report.json"}:
            continue
        yield path


def load_case(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_augment_summary(output_dir: Path, records: List[Dict[str, Any]]) -> None:
    ok = [r for r in records if r.get("status") == "ok"]
    skipped = [r for r in records if r.get("status") != "ok"]
    by_pipeline: Dict[str, int] = {}
    skip_reasons: Dict[str, int] = {}
    for record in records:
        pipeline = str(record.get("pipeline", "unknown"))
        by_pipeline[pipeline] = by_pipeline.get(pipeline, 0) + 1
        if record.get("skip_reason"):
            reason = str(record["skip_reason"])
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    summary = {
        "total_records": len(records),
        "ok_count": len(ok),
        "skipped_count": len(skipped),
        "by_pipeline": by_pipeline,
        "skip_reasons": skip_reasons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "augment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
