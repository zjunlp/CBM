"""Convert scenario B repeated trajectory exports into train and test case JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from utils.belieftrack_constants import normalize_cbm_challenge_type


VALID_SUBDIR = "belief_failure"
INSUFFICIENT_SUBDIR = "insufficient_capability"
TRAIN_CATEGORY_SUBDIRS = (VALID_SUBDIR, INSUFFICIENT_SUBDIR)
SKIP_FILENAMES = {"all_results.json", "summary.json", "stats_report.json", "cases_eval_report.json"}


def _iter_json_files(input_dir: str | Path) -> Iterable[Path]:
    base = Path(input_dir)
    if base.is_file() and base.suffix == ".json":
        yield base
        return
    for path in sorted(base.rglob("*.json")):
        if path.name in SKIP_FILENAMES:
            continue
        yield path


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _repeat_records(payload: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    repeats = payload.get("conversation")
    if isinstance(repeats, list) and repeats:
        return [repeat for repeat in repeats if isinstance(repeat, dict)]

    legacy_repeats = payload.get("repeat_trajectories")
    if isinstance(legacy_repeats, list) and legacy_repeats:
        records: List[Dict[str, Any]] = []
        for repeat in legacy_repeats:
            trajectory = repeat.get("trajectory") if isinstance(repeat, dict) else None
            if isinstance(trajectory, dict):
                records.append(
                    {
                        "conversation": trajectory.get("conversation") or trajectory.get("messages"),
                        "turn_survivors": [
                            {
                                "turn": turn.get("turn", index),
                                "golden_survivors": turn.get("gt_survivors"),
                                "sampled_survivors": turn.get("hypotheses"),
                            }
                            for index, turn in enumerate(trajectory.get("turns", []))
                        ],
                    }
                )
        if records:
            return records

    raise ValueError(f"{path} missing repeated conversation records")


def _conversation_from_repeat(repeat: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    conversation = repeat.get("conversation")
    if not isinstance(conversation, list):
        raise ValueError(f"{path} has a repeat without conversation")
    return conversation


def _extract_system_prompt(conversation: List[Dict[str, Any]]) -> str:
    for msg in conversation:
        if msg.get("role") == "system":
            return str(msg["content"])
    raise ValueError("missing system prompt")


def _user_messages(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    users = [msg for msg in conversation if msg.get("role") == "user"]
    if not users:
        raise ValueError("expected at least one user prompt")
    return users


def _assistant_messages(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [msg for msg in conversation if msg.get("role") == "assistant"]


def _turn_survivors(repeat: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    turns = repeat.get("turn_survivors")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{path} missing turn_survivors")
    return turns


def _sorted_survivors(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return sorted(str(item) for item in value)


def _turn_gts(repeat: Dict[str, Any], path: Path) -> List[List[str]]:
    return [_sorted_survivors(turn.get("golden_survivors")) for turn in _turn_survivors(repeat, path)]


def _turn_matches(repeat: Dict[str, Any], path: Path) -> List[bool]:
    matches: List[bool] = []
    for turn in _turn_survivors(repeat, path):
        golden = turn.get("golden_survivors")
        sampled = turn.get("sampled_survivors")
        matches.append(golden is not None and sampled is not None and set(golden) == set(sampled))
    return matches


def _has_correct_prefix(matches: Sequence[bool], selected_turn: int) -> bool:
    return all(prefix_idx < len(matches) and matches[prefix_idx] for prefix_idx in range(selected_turn))


def _target_set_from_path(path: Path) -> Tuple[str, ...]:
    payload = _load_payload(path)
    repeats = _repeat_records(payload, path)
    gts = _turn_gts(repeats[0], path)
    if not gts:
        raise ValueError(f"{path} missing golden survivor sets")
    return tuple(gts[-1])


def _category_dirs(input_dir: str | Path, category_kind: str) -> List[Path]:
    base = Path(input_dir)
    if base.name == category_kind:
        return [base]
    candidate = base / category_kind
    if candidate.exists():
        return [candidate]
    return [path for path in sorted(base.rglob(category_kind)) if path.is_dir()]


def _collect_category_paths(input_dirs: Sequence[str], category_kind: str) -> List[Path]:
    paths: List[Path] = []
    for input_dir in input_dirs:
        for category_dir in _category_dirs(input_dir, category_kind):
            paths.extend(_iter_json_files(category_dir))
    return sorted(set(paths))


def _cap_paths_per_target_set(paths: Sequence[Path], cap: int | None) -> List[Path]:
    if cap is None:
        return list(paths)

    grouped: Dict[Tuple[str, ...], List[Path]] = defaultdict(list)
    for path in paths:
        try:
            grouped[_target_set_from_path(path)].append(path)
        except ValueError as exc:
            print(f"[case-conversion] skip target grouping {path}: {exc}")

    selected: List[Path] = []
    for target_set in sorted(grouped):
        selected.extend(sorted(grouped[target_set])[:cap])
    return sorted(selected)


def _mode_turn_windows(mode: str, category_kind: str, n_turns: int) -> List[int]:
    if mode == "failed_stay":
        window = [2, 3] if category_kind == VALID_SUBDIR else [0, 1]
    elif mode == "failed_update":
        window = [n_turns - 1] if category_kind == VALID_SUBDIR else [0, 1]
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return [idx for idx in window if 0 <= idx < n_turns]


def _select_turn_by_error_count(
    repeats: Sequence[Dict[str, Any]],
    *,
    path: Path,
    mode: str,
    category_kind: str,
    seed: int,
) -> int:
    first_conversation = _conversation_from_repeat(repeats[0], path)
    n_turns = len(_user_messages(first_conversation))
    window = _mode_turn_windows(mode, category_kind, n_turns)
    if not window:
        raise ValueError(f"{path} has no selectable turns for mode={mode} category={category_kind}")

    if mode == "failed_update" and category_kind == VALID_SUBDIR:
        return window[-1]

    counter: Counter[int] = Counter()
    for repeat in repeats:
        matches = _turn_matches(repeat, path)
        for turn_idx in window:
            if (
                _has_correct_prefix(matches, turn_idx)
                and turn_idx < len(matches)
                and not matches[turn_idx]
            ):
                counter[turn_idx] += 1

    if not counter:
        return window[0]
    max_count = max(counter.values())
    candidates = sorted(turn for turn, count in counter.items() if count == max_count)
    rng = random.Random(_stable_seed(seed, path, mode, category_kind, "turn"))
    return rng.choice(candidates)


def _choose_repeat_for_turn(
    repeats: Sequence[Dict[str, Any]],
    *,
    selected_turn: int,
    path: Path,
    seed: int,
) -> Tuple[int, Dict[str, Any]]:
    eligible: List[Tuple[int, Dict[str, Any]]] = []
    all_repeats: List[Tuple[int, Dict[str, Any]]] = []
    for index, repeat in enumerate(repeats):
        matches = _turn_matches(repeat, path)
        if not _has_correct_prefix(matches, selected_turn):
            continue
        all_repeats.append((index, repeat))
        if selected_turn < len(matches) and not matches[selected_turn]:
            eligible.append((index, repeat))
    candidates = eligible or all_repeats
    if not candidates:
        raise ValueError(f"{path} has no repeat with correct prefix before turn {selected_turn}")
    rng = random.Random(_stable_seed(seed, path, selected_turn, "repeat"))
    return rng.choice(candidates)


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


def _turns_from_repeat(repeat: Dict[str, Any], conversation: List[Dict[str, Any]], path: Path) -> List[Dict[str, Any]]:
    users = _user_messages(conversation)
    gts = _turn_gts(repeat, path)
    if len(users) != len(gts):
        raise ValueError(f"{path} has {len(users)} user prompts but {len(gts)} golden labels")
    return [
        {"prompt": str(users[idx]["content"]), "golden": gts[idx]}
        for idx in range(len(users))
    ]


def _oracle_from_target_set(target_set: Sequence[str]) -> str:
    if len(target_set) == 1:
        return str(target_set[0])
    return ",".join(target_set)


def _convert_train_case(path: Path, mode: str, category_kind: str, *, seed: int) -> Dict[str, Any]:
    payload = _load_payload(path)
    repeats = _repeat_records(payload, path)
    selected_turn = _select_turn_by_error_count(
        repeats,
        path=path,
        mode=mode,
        category_kind=category_kind,
        seed=seed,
    )
    repeat_index, repeat = _choose_repeat_for_turn(
        repeats,
        selected_turn=selected_turn,
        path=path,
        seed=seed,
    )
    conversation = _conversation_from_repeat(repeat, path)
    turns = _turns_from_repeat(repeat, conversation, path)
    target_set = tuple(turns[-1]["golden"])
    selected_gt = turns[selected_turn]["golden"]

    return {
        "case_id": f"{path.stem}_turn{selected_turn}",
        "challenge_type": mode,
        "cbm_challenge_type": normalize_cbm_challenge_type(mode),
        "oracle": _oracle_from_target_set(target_set),
        "target_set": list(target_set),
        "system_prompt": _extract_system_prompt(conversation),
        "messages": _build_training_messages(conversation, selected_turn),
        "gt_survivors": selected_gt,
        "selected_turn": selected_turn,
        "selected_prompt": turns[selected_turn]["prompt"],
        "turns": turns,
        "source_file": str(path),
        "source_category": category_kind,
        "source_repeat_index": repeat_index,
    }


def _convert_eval_case(path: Path, mode: str) -> Dict[str, Any]:
    payload = _load_payload(path)
    repeats = _repeat_records(payload, path)
    repeat = repeats[0]
    conversation = _conversation_from_repeat(repeat, path)
    turns = _turns_from_repeat(repeat, conversation, path)
    target_set = tuple(turns[-1]["golden"])
    return {
        "case_id": path.stem,
        "challenge_type": mode,
        "cbm_challenge_type": normalize_cbm_challenge_type(mode),
        "oracle": _oracle_from_target_set(target_set),
        "target_set": list(target_set),
        "system_prompt": _extract_system_prompt(conversation),
        "turns": turns,
        "source_file": str(path),
        "source_category": VALID_SUBDIR,
        "source_repeat_index": 0,
    }


def _case_sort_key(case: Dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(case.get("challenge_type", "")),
        ",".join(str(item) for item in case.get("target_set", [])),
        str(case.get("oracle", "")),
        str(case.get("case_id", "")),
    )


def _build_train_cases_for_mode(
    input_dirs: Sequence[str],
    *,
    mode: str,
    max_valid_cases_per_target_set: int | None,
    max_insufficient_cases_per_target_set: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for category_kind in TRAIN_CATEGORY_SUBDIRS:
        paths = _collect_category_paths(input_dirs, category_kind)
        cap = (
            max_valid_cases_per_target_set
            if category_kind == VALID_SUBDIR
            else max_insufficient_cases_per_target_set
        )
        for path in _cap_paths_per_target_set(paths, cap):
            try:
                cases.append(_convert_train_case(path, mode, category_kind, seed=seed))
            except ValueError as exc:
                print(f"[case-conversion] skip train case {path}: {exc}")
    return cases


def _build_eval_cases_for_mode(
    input_dirs: Sequence[str],
    *,
    mode: str,
    max_cases_per_target_set: int | None,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    paths = _collect_category_paths(input_dirs, VALID_SUBDIR)
    for path in _cap_paths_per_target_set(paths, max_cases_per_target_set):
        try:
            cases.append(_convert_eval_case(path, mode))
        except ValueError as exc:
            print(f"[case-conversion] skip test case {path}: {exc}")
    return cases


def build_train_cases(
    *,
    failed_stay_dirs: Sequence[str] = (),
    failed_update_dirs: Sequence[str] = (),
    max_valid_cases_per_target_set: int | None = None,
    max_insufficient_cases_per_target_set: int | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    cases.extend(
        _build_train_cases_for_mode(
            failed_stay_dirs,
            mode="failed_stay",
            max_valid_cases_per_target_set=max_valid_cases_per_target_set,
            max_insufficient_cases_per_target_set=max_insufficient_cases_per_target_set,
            seed=seed,
        )
    )
    cases.extend(
        _build_train_cases_for_mode(
            failed_update_dirs,
            mode="failed_update",
            max_valid_cases_per_target_set=max_valid_cases_per_target_set,
            max_insufficient_cases_per_target_set=max_insufficient_cases_per_target_set,
            seed=seed,
        )
    )
    cases.sort(key=_case_sort_key)
    if not cases:
        raise ValueError("No valid train cases found.")
    return cases


def build_eval_cases(
    *,
    failed_stay_dirs: Sequence[str] = (),
    failed_update_dirs: Sequence[str] = (),
    max_cases_per_target_set: int | None = None,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    cases.extend(
        _build_eval_cases_for_mode(
            failed_stay_dirs,
            mode="failed_stay",
            max_cases_per_target_set=max_cases_per_target_set,
        )
    )
    cases.extend(
        _build_eval_cases_for_mode(
            failed_update_dirs,
            mode="failed_update",
            max_cases_per_target_set=max_cases_per_target_set,
        )
    )
    cases.sort(key=_case_sort_key)
    if not cases:
        raise ValueError("No valid test cases found.")
    return cases


def _write_cases(cases: List[Dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scenario B train/test case JSON files")
    parser.add_argument("--build-scope", choices=("train", "test", "both"), default="both")
    parser.add_argument("--train-failed_stay-dirs", nargs="*", default=[])
    parser.add_argument("--train-failed_update-dirs", nargs="*", default=[])
    parser.add_argument("--test-failed_stay-dirs", nargs="*", default=[])
    parser.add_argument("--test-failed_update-dirs", nargs="*", default=[])
    parser.add_argument("--train-output-path", type=str, default=None)
    parser.add_argument("--test-output-path", type=str, default=None)
    parser.add_argument("--max-train-valid-cases-per-target-set", type=int, default=None)
    parser.add_argument("--max-train-insufficient-cases-per-target-set", type=int, default=None)
    parser.add_argument("--max-test-cases-per-target-set", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_train = args.build_scope in {"train", "both"}
    run_test = args.build_scope in {"test", "both"}

    if run_train:
        if not args.train_failed_stay_dirs and not args.train_failed_update_dirs:
            raise ValueError(f"--build-scope {args.build_scope} requires train failed_stay or failed_update directories")
        if not args.train_output_path:
            raise ValueError("training conversion requires --train-output-path")
        train_cases = build_train_cases(
            failed_stay_dirs=args.train_failed_stay_dirs,
            failed_update_dirs=args.train_failed_update_dirs,
            max_valid_cases_per_target_set=args.max_train_valid_cases_per_target_set,
            max_insufficient_cases_per_target_set=args.max_train_insufficient_cases_per_target_set,
            seed=args.seed,
        )
        _write_cases(train_cases, args.train_output_path)
        print(f"Saved {len(train_cases)} train cases to {args.train_output_path}")

    if run_test:
        if not args.test_failed_stay_dirs and not args.test_failed_update_dirs:
            raise ValueError(f"--build-scope {args.build_scope} requires test failed_stay or failed_update directories")
        if not args.test_output_path:
            raise ValueError("test conversion requires --test-output-path")
        test_cases = build_eval_cases(
            failed_stay_dirs=args.test_failed_stay_dirs,
            failed_update_dirs=args.test_failed_update_dirs,
            max_cases_per_target_set=args.max_test_cases_per_target_set,
        )
        _write_cases(test_cases, args.test_output_path)
        print(f"Saved {len(test_cases)} test cases to {args.test_output_path}")

    if not run_train and not run_test:
        raise ValueError("No build scope was selected.")


if __name__ == "__main__":
    main()
