"""Split scenario A 7B valid trajectories into train and test cases."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from task_a.training.case_conversion import (
    _assistant_messages,
    _conversation_from_trajectory,
    _extract_system_prompt,
    _extract_turn_gts,
    _oracle,
    _turn_matches,
    _user_messages,
)


VALID_SUBDIR = "belief_failure"
SKIP_FILENAMES = {"all_results.json", "summary.json", "stats_report.json", "cases_eval_report.json"}


def _load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _iter_json_files(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix == ".json" and path.name not in SKIP_FILENAMES:
        yield path
        return
    for item in sorted(path.rglob("*.json")):
        if item.name not in SKIP_FILENAMES:
            yield item


def _target_name(target_dir: Path) -> str:
    name = target_dir.name
    return name[len("target_") :] if name.startswith("target_") else name


def _case_sort_key(case: Dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(case.get("challenge_type", "")),
        str(case.get("oracle", "")),
        str(case.get("case_id", "")),
    )


def _load_repeats(path: Path) -> List[Dict[str, Any]]:
    payload = _load_payload(path)
    repeats = payload.get("repeat_trajectories")
    if not isinstance(repeats, list) or not repeats:
        raise ValueError(f"{path} missing repeat_trajectories")
    return [repeat for repeat in repeats if isinstance(repeat, dict) and isinstance(repeat.get("trajectory"), dict)]


def _turns_from_trajectory(path: Path, payload: Dict[str, Any], trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversation = _conversation_from_trajectory(trajectory)
    users = _user_messages(conversation)
    turn_gts = _extract_turn_gts(
        path=path,
        trajectory=trajectory,
        conversation=conversation,
        sequence=payload.get("challenge_sequence") or trajectory.get("challenge_sequence") or {},
    )
    if len(users) != len(turn_gts):
        raise ValueError(f"{path} has {len(users)} user prompts but {len(turn_gts)} golden labels")
    return [{"prompt": str(users[idx]["content"]), "golden": turn_gts[idx]} for idx in range(len(users))]


def _has_correct_prefix(matches: Sequence[bool], selected_turn: int) -> bool:
    return all(prefix_idx < len(matches) and matches[prefix_idx] for prefix_idx in range(selected_turn))


def _build_training_messages(conversation: List[Dict[str, Any]], selected_turn: int) -> List[Dict[str, str]]:
    users = _user_messages(conversation)
    assistants = _assistant_messages(conversation)
    if selected_turn >= len(users):
        raise ValueError(f"selected_turn {selected_turn} out of range for {len(users)} user turns")
    if selected_turn > len(assistants):
        raise ValueError(f"selected_turn {selected_turn} needs {selected_turn} assistant turns, got {len(assistants)}")

    messages: List[Dict[str, str]] = [{"role": "system", "content": _extract_system_prompt(conversation)}]
    for turn_idx in range(selected_turn):
        messages.append({"role": "user", "content": str(users[turn_idx]["content"])})
        messages.append({"role": "assistant", "content": str(assistants[turn_idx]["content"])})
    messages.append({"role": "user", "content": str(users[selected_turn]["content"])})
    return messages


def _convert_entry(path: Path, *, target: str, mode: str) -> Dict[str, Any]:
    payload = _load_payload(path)
    repeats = _load_repeats(path)
    first_trajectory = repeats[0]["trajectory"]
    conversation = _conversation_from_trajectory(first_trajectory)
    turns = _turns_from_trajectory(path, payload, first_trajectory)
    oracle = _oracle(payload, first_trajectory) or target
    return {
        "target": target,
        "mode": mode,
        "path": path,
        "payload": payload,
        "repeats": repeats,
        "eval_case": {
            "case_id": f"{mode}_{path.stem}",
            "challenge_type": mode,
            "oracle": str(oracle),
            "target_set": [target],
            "system_prompt": _extract_system_prompt(conversation),
            "turns": turns,
            "source_file": str(path),
            "source_category": VALID_SUBDIR,
            "source_repeat_count": len(repeats),
        },
    }


def _collect_entries(input_dir: str | Path) -> List[Dict[str, Any]]:
    root = Path(input_dir)
    entries: List[Dict[str, Any]] = []
    for target_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        target = _target_name(target_dir)
        for mode in ("failed_stay", "failed_update"):
            mode_dir = target_dir / mode
            if not mode_dir.exists():
                continue
            for path in _iter_json_files(mode_dir):
                try:
                    entries.append(_convert_entry(path, target=target, mode=mode))
                except ValueError as exc:
                    print(f"[filter-a-7b-cases] skip {path}: {exc}")
    entries.sort(key=lambda item: (item["target"], item["mode"], str(item["path"])))
    return entries


def _sample(items: Sequence[Any], cap: int | None, rng: random.Random) -> List[Any]:
    values = list(items)
    if cap is None or len(values) <= cap:
        return values
    return rng.sample(values, cap)


def _choose_repeat(entry: Dict[str, Any], selected_turn: int, rng: random.Random) -> Dict[str, Any]:
    eligible = []
    for repeat in entry["repeats"]:
        trajectory = repeat["trajectory"]
        if _has_correct_prefix(_turn_matches(trajectory), selected_turn):
            eligible.append(repeat)
    if not eligible:
        raise ValueError(f"{entry['path']} has no repeat with correct prefix before turn {selected_turn}")
    return rng.choice(eligible)


def _train_case(entry: Dict[str, Any], *, selected_turn: int, train_kind: str, rng: random.Random) -> Dict[str, Any]:
    turns = entry["eval_case"]["turns"]
    if selected_turn >= len(turns):
        raise ValueError(f"{entry['path']} selected turn {selected_turn} out of range")
    repeat = _choose_repeat(entry, selected_turn, rng)
    conversation = _conversation_from_trajectory(repeat["trajectory"])
    eval_case = entry["eval_case"]
    return {
        "case_id": f"{eval_case['case_id']}_{train_kind}_turn{selected_turn}",
        "challenge_type": eval_case["challenge_type"],
        "oracle": eval_case["oracle"],
        "target_set": list(eval_case.get("target_set", [])),
        "system_prompt": eval_case["system_prompt"],
        "messages": _build_training_messages(conversation, selected_turn),
        "gt_survivors": turns[selected_turn]["golden"],
        "selected_turn": selected_turn,
        "selected_prompt": turns[selected_turn]["prompt"],
        "turns": turns,
        "source_file": eval_case["source_file"],
        "source_category": VALID_SUBDIR,
        "source_repeat_index": repeat.get("repeat_index"),
        "train_kind": train_kind,
    }


def _candidate_train_case(entry: Dict[str, Any], *, train_kind: str, rng: random.Random) -> Dict[str, Any] | None:
    n_turns = len(entry["eval_case"]["turns"])
    if train_kind == "capability":
        selectable = [idx for idx in (0, 1) if idx < n_turns]
        if not selectable:
            return None
        selected_turn = rng.choice(selectable)
    elif train_kind == "belief":
        selected_turn = 2
        if selected_turn >= n_turns:
            return None
    else:
        raise ValueError(f"unsupported train kind: {train_kind}")
    try:
        return _train_case(entry, selected_turn=selected_turn, train_kind=train_kind, rng=rng)
    except ValueError as exc:
        print(f"[filter-a-7b-cases] skip {train_kind} train case {entry['path']}: {exc}")
        return None


def _build_train_cases(
    entries: Sequence[Dict[str, Any]],
    *,
    max_failed_stay_train_capability_cases_per_target: int | None,
    max_failed_stay_train_belief_cases_per_target: int | None,
    max_failed_update_train_capability_cases_per_target: int | None,
    max_failed_update_train_belief_cases_per_target: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    train_cases: List[Dict[str, Any]] = []
    targets = sorted({entry["target"] for entry in entries})
    for target in targets:
        for mode in ("failed_stay", "failed_update"):
            group = [entry for entry in entries if entry["target"] == target and entry["mode"] == mode]
            capability_rng = random.Random(f"{seed}:train:{target}:{mode}:capability")
            belief_rng = random.Random(f"{seed}:train:{target}:{mode}:belief")
            capability_cap = (
                max_failed_stay_train_capability_cases_per_target
                if mode == "failed_stay"
                else max_failed_update_train_capability_cases_per_target
            )
            belief_cap = (
                max_failed_stay_train_belief_cases_per_target
                if mode == "failed_stay"
                else max_failed_update_train_belief_cases_per_target
            )

            capability_candidates = [
                case
                for entry in group
                if (case := _candidate_train_case(entry, train_kind="capability", rng=capability_rng)) is not None
            ]
            capability_cases = _sample(capability_candidates, capability_cap, capability_rng)
            used_sources = {str(case["source_file"]) for case in capability_cases}

            belief_candidates = [
                case
                for entry in group
                if str(entry["path"]) not in used_sources
                if (case := _candidate_train_case(entry, train_kind="belief", rng=belief_rng)) is not None
            ]
            belief_cases = _sample(belief_candidates, belief_cap, belief_rng)
            train_cases.extend(capability_cases)
            train_cases.extend(belief_cases)
    train_cases.sort(key=_case_sort_key)
    return train_cases


def _build_test_cases(
    entries: Sequence[Dict[str, Any]],
    *,
    test_target: str,
    max_failed_stay_test_cases: int | None,
    max_failed_update_test_cases: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for mode in ("failed_stay", "failed_update"):
        mode_cases = [
            entry["eval_case"]
            for entry in entries
            if entry["target"] == test_target and entry["mode"] == mode
        ]
        mode_cases.sort(key=_case_sort_key)
        cap = max_failed_stay_test_cases if mode == "failed_stay" else max_failed_update_test_cases
        cases.extend(_sample(mode_cases, cap, random.Random(f"{seed}:test:{test_target}:{mode}")))
    cases.sort(key=_case_sort_key)
    return cases


def build_7b_splits(
    *,
    input_dir: str,
    max_failed_stay_test_cases: int | None = None,
    max_failed_update_test_cases: int | None = None,
    max_failed_stay_train_capability_cases_per_target: int | None = None,
    max_failed_stay_train_belief_cases_per_target: int | None = None,
    max_failed_update_train_capability_cases_per_target: int | None = None,
    max_failed_update_train_belief_cases_per_target: int | None = None,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entries = _collect_entries(input_dir)
    if not entries:
        raise ValueError("No valid scenario A 7B source cases found.")
    targets = sorted({entry["target"] for entry in entries})
    test_target = random.Random(f"{seed}:test-target").choice(targets)

    test_cases = _build_test_cases(
        entries,
        test_target=test_target,
        max_failed_stay_test_cases=max_failed_stay_test_cases,
        max_failed_update_test_cases=max_failed_update_test_cases,
        seed=seed,
    )
    if not test_cases:
        raise ValueError(f"No test cases found for target {test_target}")

    train_entries = [entry for entry in entries if entry["target"] != test_target]
    train_cases = _build_train_cases(
        train_entries,
        max_failed_stay_train_capability_cases_per_target=max_failed_stay_train_capability_cases_per_target,
        max_failed_stay_train_belief_cases_per_target=max_failed_stay_train_belief_cases_per_target,
        max_failed_update_train_capability_cases_per_target=max_failed_update_train_capability_cases_per_target,
        max_failed_update_train_belief_cases_per_target=max_failed_update_train_belief_cases_per_target,
        seed=seed,
    )
    return train_cases, test_cases


def _write_cases(cases: Sequence[Dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(cases), ensure_ascii=False, indent=2), encoding="utf-8")


def _assert_target_disjoint(train_cases: Iterable[Dict[str, Any]], test_cases: Iterable[Dict[str, Any]]) -> None:
    train_targets = {tuple(case.get("target_set", [])) for case in train_cases}
    test_targets = {tuple(case.get("target_set", [])) for case in test_cases}
    overlap = train_targets & test_targets
    if overlap:
        raise ValueError(f"test target overlaps with train targets: {sorted(overlap)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split scenario A 7B valid cases into train/test JSON files.")
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--train-output-path", required=True, type=str)
    parser.add_argument("--test-output-path", required=True, type=str)
    parser.add_argument("--max-failed_stay-test-cases", type=int, default=None)
    parser.add_argument("--max-failed_update-test-cases", type=int, default=None)
    parser.add_argument("--max-failed_stay-train-capability-cases-per-target", type=int, default=None)
    parser.add_argument("--max-failed_stay-train-belief-cases-per-target", type=int, default=None)
    parser.add_argument("--max-failed_update-train-capability-cases-per-target", type=int, default=None)
    parser.add_argument("--max-failed_update-train-belief-cases-per-target", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_cases, test_cases = build_7b_splits(
        input_dir=args.input_dir,
        max_failed_stay_test_cases=args.max_failed_stay_test_cases,
        max_failed_update_test_cases=args.max_failed_update_test_cases,
        max_failed_stay_train_capability_cases_per_target=args.max_failed_stay_train_capability_cases_per_target,
        max_failed_stay_train_belief_cases_per_target=args.max_failed_stay_train_belief_cases_per_target,
        max_failed_update_train_capability_cases_per_target=args.max_failed_update_train_capability_cases_per_target,
        max_failed_update_train_belief_cases_per_target=args.max_failed_update_train_belief_cases_per_target,
        seed=args.seed,
    )
    _assert_target_disjoint(train_cases, test_cases)
    _write_cases(train_cases, args.train_output_path)
    _write_cases(test_cases, args.test_output_path)
    test_targets = sorted({",".join(case.get("target_set", [])) for case in test_cases})
    print(f"Saved {len(train_cases)} train cases to {args.train_output_path}")
    print(f"Saved {len(test_cases)} test cases to {args.test_output_path}; test_targets={test_targets}")


if __name__ == "__main__":
    main()
