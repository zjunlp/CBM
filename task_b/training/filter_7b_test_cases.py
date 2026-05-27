"""Build 7B scenario B train/test cases from belief_failure trajectories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from task_b.training.case_conversion import (
    VALID_SUBDIR,
    _assistant_messages,
    _case_sort_key,
    _collect_category_paths,
    _conversation_from_repeat,
    _extract_system_prompt,
    _load_payload,
    _oracle_from_target_set,
    _repeat_records,
    _turns_from_repeat,
    _user_messages,
)


def _sorted_labels(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return sorted(str(item) for item in value)


def _turns_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]] | None:
    raw_turns = payload.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        return None

    turns: List[Dict[str, Any]] = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            return None
        prompt = turn.get("prompt")
        if prompt is None:
            return None
        golden = turn.get("golden", turn.get("golden_hypotheses", turn.get("gt_survivors")))
        turns.append({"prompt": str(prompt), "golden": _sorted_labels(golden)})
    return turns


def _system_prompt_from_payload(payload: Dict[str, Any], repeat: Dict[str, Any] | None, path: Path) -> str:
    system_prompt = payload.get("system_prompt")
    if system_prompt is not None:
        return str(system_prompt)
    if repeat is None:
        raise ValueError(f"{path} missing system_prompt")
    return _extract_system_prompt(_conversation_from_repeat(repeat, path))


def _repeat_records_or_empty(payload: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    try:
        return _repeat_records(payload, path)
    except ValueError:
        return []


def _matches_from_turns(raw_turns: Any) -> List[bool]:
    matches: List[bool] = []
    if not isinstance(raw_turns, list):
        return matches
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        if "model_matches_golden" in turn:
            matches.append(bool(turn["model_matches_golden"]))
            continue
        golden = turn.get("golden_survivors", turn.get("golden_hypotheses", turn.get("gt_survivors")))
        sampled = turn.get("sampled_survivors", turn.get("model_hypotheses", turn.get("hypotheses")))
        matches.append(golden is not None and sampled is not None and set(golden) == set(sampled))
    return matches


def _repeat_infos(payload: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []

    legacy_repeats = payload.get("repeat_trajectories")
    if isinstance(legacy_repeats, list):
        for index, repeat in enumerate(legacy_repeats):
            if not isinstance(repeat, dict):
                continue
            trajectory = repeat.get("trajectory")
            if not isinstance(trajectory, dict):
                continue
            conversation = trajectory.get("conversation") or trajectory.get("messages")
            if not isinstance(conversation, list):
                continue
            infos.append(
                {
                    "repeat_index": repeat.get("repeat_index", index),
                    "conversation": conversation,
                    "matches": _matches_from_turns(trajectory.get("turns")),
                }
            )

    if infos:
        return infos

    for index, repeat in enumerate(_repeat_records_or_empty(payload, path)):
        try:
            conversation = _conversation_from_repeat(repeat, path)
        except ValueError:
            continue
        matches = _matches_from_turns(repeat.get("turn_survivors"))
        infos.append(
            {
                "repeat_index": repeat.get("repeat_index", index),
                "conversation": conversation,
                "matches": matches,
            }
        )
    return infos


def _convert_7b_eval_case(path: Path, mode: str) -> Dict[str, Any]:
    payload = _load_payload(path)
    repeats = _repeat_records_or_empty(payload, path)
    repeat = repeats[0] if repeats else None

    turns = _turns_from_payload(payload)
    if turns is None:
        if repeat is None:
            raise ValueError(f"{path} missing turns or repeated conversation records")
        conversation = _conversation_from_repeat(repeat, path)
        turns = _turns_from_repeat(repeat, conversation, path)

    target_set = tuple(turns[-1]["golden"])
    if not target_set and isinstance(payload.get("target_set"), list):
        target_set = tuple(str(item) for item in payload["target_set"])
    if not target_set and payload.get("oracle"):
        target_set = (str(payload["oracle"]),)

    return {
        "case_id": f"{mode}_{path.stem}",
        "challenge_type": mode,
        "oracle": _oracle_from_target_set(target_set),
        "target_set": list(target_set),
        "system_prompt": _system_prompt_from_payload(payload, repeat, path),
        "turns": turns,
        "source_file": str(path),
        "source_category": VALID_SUBDIR,
        "source_repeat_index": 0,
        "source_repeat_count": len(repeats),
    }


def _target_key(case: Dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in case.get("target_set", []))


def _mode_target_key(case: Dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (str(case.get("challenge_type", "")), _target_key(case))


def _source_id(entry: Dict[str, Any]) -> str:
    return str(entry["eval_case"]["source_file"])


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


def _choose_repeat(entry: Dict[str, Any], selected_turn: int, rng: random.Random) -> Dict[str, Any]:
    eligible = [
        repeat
        for repeat in entry["repeat_infos"]
        if _has_correct_prefix(repeat.get("matches", []), selected_turn)
    ]
    if not eligible:
        raise ValueError(
            f"{entry['eval_case']['source_file']} has no repeat with correct prefix before turn {selected_turn}"
        )
    return rng.choice(eligible)


def _convert_train_case(
    entry: Dict[str, Any],
    *,
    selected_turn: int,
    train_kind: str,
    rng: random.Random,
) -> Dict[str, Any]:
    eval_case = entry["eval_case"]
    turns = eval_case["turns"]
    if selected_turn >= len(turns):
        raise ValueError(f"{eval_case['source_file']} selected turn {selected_turn} out of range")

    repeat = _choose_repeat(entry, selected_turn, rng)
    messages = _build_training_messages(repeat["conversation"], selected_turn)
    return {
        "case_id": f"{eval_case['case_id']}_{train_kind}_turn{selected_turn}",
        "challenge_type": eval_case["challenge_type"],
        "oracle": eval_case["oracle"],
        "target_set": list(eval_case.get("target_set", [])),
        "system_prompt": eval_case["system_prompt"],
        "messages": messages,
        "gt_survivors": turns[selected_turn]["golden"],
        "selected_turn": selected_turn,
        "selected_prompt": turns[selected_turn]["prompt"],
        "turns": turns,
        "source_file": eval_case["source_file"],
        "source_category": VALID_SUBDIR,
        "source_repeat_index": repeat.get("repeat_index"),
        "train_kind": train_kind,
    }


def _sample_without_replacement(items: Sequence[Any], max_items: int | None, rng: random.Random) -> List[Any]:
    values = list(items)
    if max_items is None or len(values) <= max_items:
        return values
    return rng.sample(values, max_items)


def _collect_entries(input_dir: str, *, mode: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in _collect_category_paths([input_dir], VALID_SUBDIR):
        try:
            payload = _load_payload(path)
            eval_case = _convert_7b_eval_case(path, mode)
            repeat_infos = _repeat_infos(payload, path)
            if not repeat_infos:
                raise ValueError("missing usable repeat conversations")
        except ValueError as exc:
            print(f"[filter-7b-cases] skip {mode} case {path}: {exc}")
            continue
        entries.append({"path": path, "mode": mode, "eval_case": eval_case, "repeat_infos": repeat_infos})
    entries.sort(key=lambda item: _case_sort_key(item["eval_case"]))
    return entries


def _candidate_train_case(
    entry: Dict[str, Any],
    *,
    train_kind: str,
    rng: random.Random,
) -> Dict[str, Any] | None:
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
        return _convert_train_case(entry, selected_turn=selected_turn, train_kind=train_kind, rng=rng)
    except ValueError as exc:
        print(f"[filter-7b-cases] skip {train_kind} train case {entry['path']}: {exc}")
        return None


def _build_train_split(
    entries: Sequence[Dict[str, Any]],
    *,
    max_failed_stay_train_capability_cases: int | None,
    max_failed_stay_train_belief_cases: int | None,
    max_failed_update_train_capability_cases: int | None,
    max_failed_update_train_belief_cases: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    train_cases: List[Dict[str, Any]] = []
    for mode in ("failed_stay", "failed_update"):
        mode_entries = [entry for entry in entries if entry["mode"] == mode]
        capability_rng = random.Random(f"{seed}:train:{mode}:capability")
        belief_rng = random.Random(f"{seed}:train:{mode}:belief")
        capability_cap = (
            max_failed_stay_train_capability_cases
            if mode == "failed_stay"
            else max_failed_update_train_capability_cases
        )
        belief_cap = (
            max_failed_stay_train_belief_cases
            if mode == "failed_stay"
            else max_failed_update_train_belief_cases
        )

        capability_candidates = [
            case
            for entry in mode_entries
            if (case := _candidate_train_case(entry, train_kind="capability", rng=capability_rng)) is not None
        ]
        capability_train = _sample_without_replacement(
            capability_candidates,
            capability_cap,
            capability_rng,
        )
        used_sources = {str(case["source_file"]) for case in capability_train}

        belief_candidates = [
            case
            for entry in mode_entries
            if _source_id(entry) not in used_sources
            if (case := _candidate_train_case(entry, train_kind="belief", rng=belief_rng)) is not None
        ]
        belief_train = _sample_without_replacement(
            belief_candidates,
            belief_cap,
            belief_rng,
        )

        train_cases.extend(capability_train)
        train_cases.extend(belief_train)

    train_cases.sort(key=_case_sort_key)
    return train_cases


def _filter_test_target_sets(
    mode_cases: Sequence[Dict[str, Any]],
    *,
    max_target_sets: int | None,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if max_target_sets is None:
        return list(mode_cases)

    target_sets = sorted({_target_key(case) for case in mode_cases})
    if len(target_sets) <= max_target_sets:
        selected_targets = set(target_sets)
    else:
        selected_targets = set(rng.sample(target_sets, max_target_sets))
    return [case for case in mode_cases if _target_key(case) in selected_targets]


def _build_test_split(
    entries: Sequence[Dict[str, Any]],
    *,
    max_failed_stay_cases: int | None,
    max_failed_update_cases: int | None,
    max_failed_stay_target_sets: int | None,
    max_failed_update_target_sets: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for mode in ("failed_stay", "failed_update"):
        mode_cases = [
            entry["eval_case"]
            for entry in entries
            if entry["mode"] == mode
        ]
        mode_cases.sort(key=_case_sort_key)
        target_cap = max_failed_stay_target_sets if mode == "failed_stay" else max_failed_update_target_sets
        mode_cases = _filter_test_target_sets(
            mode_cases,
            max_target_sets=target_cap,
            rng=random.Random(f"{seed}:test:{mode}:targets"),
        )
        cap = max_failed_stay_cases if mode == "failed_stay" else max_failed_update_cases
        mode_cases = _sample_without_replacement(mode_cases, cap, random.Random(f"{seed}:test:{mode}"))
        cases.extend(mode_cases)
    cases.sort(key=_case_sort_key)
    return cases


def build_7b_splits(
    *,
    failed_stay_dir: str,
    failed_update_dir: str,
    max_failed_stay_test_cases: int | None = None,
    max_failed_update_test_cases: int | None = None,
    max_failed_stay_test_target_sets: int | None = None,
    max_failed_update_test_target_sets: int | None = None,
    max_failed_stay_train_capability_cases: int | None = None,
    max_failed_stay_train_belief_cases: int | None = None,
    max_failed_update_train_capability_cases: int | None = None,
    max_failed_update_train_belief_cases: int | None = None,
    max_train_capability_cases: int | None = None,
    max_train_belief_cases: int | None = None,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entries = _collect_entries(failed_stay_dir, mode="failed_stay") + _collect_entries(failed_update_dir, mode="failed_update")
    if not entries:
        raise ValueError("No valid 7B source cases found.")

    test_cases = _build_test_split(
        entries,
        max_failed_stay_cases=max_failed_stay_test_cases,
        max_failed_update_cases=max_failed_update_test_cases,
        max_failed_stay_target_sets=max_failed_stay_test_target_sets,
        max_failed_update_target_sets=max_failed_update_test_target_sets,
        seed=seed,
    )
    if not test_cases:
        raise ValueError("No valid 7B test cases found.")
    test_target_sets = {_mode_target_key(case) for case in test_cases}
    train_entries = [
        entry
        for entry in entries
        if _mode_target_key(entry["eval_case"]) not in test_target_sets
    ]
    train_cases = _build_train_split(
        train_entries,
        max_failed_stay_train_capability_cases=(
            max_train_capability_cases
            if max_failed_stay_train_capability_cases is None
            else max_failed_stay_train_capability_cases
        ),
        max_failed_stay_train_belief_cases=(
            max_train_belief_cases
            if max_failed_stay_train_belief_cases is None
            else max_failed_stay_train_belief_cases
        ),
        max_failed_update_train_capability_cases=(
            max_train_capability_cases
            if max_failed_update_train_capability_cases is None
            else max_failed_update_train_capability_cases
        ),
        max_failed_update_train_belief_cases=(
            max_train_belief_cases
            if max_failed_update_train_belief_cases is None
            else max_failed_update_train_belief_cases
        ),
        seed=seed,
    )
    return train_cases, test_cases


def build_7b_test_cases(
    *,
    failed_stay_dir: str,
    failed_update_dir: str,
    max_failed_stay_cases: int | None = None,
    max_failed_update_cases: int | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    _, test_cases = build_7b_splits(
        failed_stay_dir=failed_stay_dir,
        failed_update_dir=failed_update_dir,
        max_failed_stay_test_cases=max_failed_stay_cases,
        max_failed_update_test_cases=max_failed_update_cases,
        max_failed_stay_train_capability_cases=0,
        max_failed_stay_train_belief_cases=0,
        max_failed_update_train_capability_cases=0,
        max_failed_update_train_belief_cases=0,
        seed=seed,
    )
    return test_cases


def _write_cases(cases: Sequence[Dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(cases), ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split scenario B 7B belief_failure trajectories into train and test cases."
    )
    parser.add_argument("--failed_stay-dir", required=True, type=str)
    parser.add_argument("--failed_update-dir", required=True, type=str)
    parser.add_argument("--train-output-path", type=str, default=None)
    parser.add_argument("--test-output-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None, help="Backward-compatible alias for --test-output-path.")
    parser.add_argument("--max-failed_stay-test-cases", "--max-failed_stay-cases", dest="max_failed_stay_test_cases", type=int, default=None)
    parser.add_argument("--max-failed_update-test-cases", "--max-failed_update-cases", dest="max_failed_update_test_cases", type=int, default=None)
    parser.add_argument("--max-failed_stay-test-target-sets", type=int, default=None)
    parser.add_argument("--max-failed_update-test-target-sets", type=int, default=None)
    parser.add_argument("--max-train-capability-cases", type=int, default=0)
    parser.add_argument("--max-train-belief-cases", type=int, default=0)
    parser.add_argument("--max-failed_stay-train-capability-cases", type=int, default=None)
    parser.add_argument("--max-failed_stay-train-belief-cases", type=int, default=None)
    parser.add_argument("--max-failed_update-train-capability-cases", type=int, default=None)
    parser.add_argument("--max-failed_update-train-belief-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _assert_disjoint_target_sets(train_cases: Iterable[Dict[str, Any]], test_cases: Iterable[Dict[str, Any]]) -> None:
    train_targets = {_mode_target_key(case) for case in train_cases}
    test_targets = {_mode_target_key(case) for case in test_cases}
    overlap = train_targets & test_targets
    if overlap:
        raise ValueError(f"test target sets overlap with train target sets: {sorted(overlap)}")


def main() -> None:
    args = _parse_args()
    test_output_path = args.test_output_path or args.output_path
    if not args.train_output_path and not test_output_path:
        raise ValueError("at least one of --train-output-path or --test-output-path is required")

    train_cases, test_cases = build_7b_splits(
        failed_stay_dir=args.failed_stay_dir,
        failed_update_dir=args.failed_update_dir,
        max_failed_stay_test_cases=args.max_failed_stay_test_cases,
        max_failed_update_test_cases=args.max_failed_update_test_cases,
        max_failed_stay_test_target_sets=args.max_failed_stay_test_target_sets,
        max_failed_update_test_target_sets=args.max_failed_update_test_target_sets,
        max_failed_stay_train_capability_cases=args.max_failed_stay_train_capability_cases,
        max_failed_stay_train_belief_cases=args.max_failed_stay_train_belief_cases,
        max_failed_update_train_capability_cases=args.max_failed_update_train_capability_cases,
        max_failed_update_train_belief_cases=args.max_failed_update_train_belief_cases,
        max_train_capability_cases=args.max_train_capability_cases,
        max_train_belief_cases=args.max_train_belief_cases,
        seed=args.seed,
    )
    _assert_disjoint_target_sets(train_cases, test_cases)

    if args.train_output_path:
        _write_cases(train_cases, args.train_output_path)
        print(f"Saved {len(train_cases)} 7B train cases to {args.train_output_path}")
    if test_output_path:
        _write_cases(test_cases, test_output_path)
        print(f"Saved {len(test_cases)} 7B test cases to {test_output_path}")


if __name__ == "__main__":
    main()
