"""Convert repeated trajectory case directories into train and eval case JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from utils.belieftrack_constants import normalize_cbm_challenge_type


VALID_SUBDIR = "belief_failure"
TRAIN_CATEGORY_SUBDIRS = (VALID_SUBDIR, "insufficient_capability")


def _iter_json_files(input_dir: str | Path) -> Iterable[Path]:
    base = Path(input_dir)
    if base.is_file() and base.suffix == ".json":
        yield base
        return
    for path in sorted(base.rglob("*.json")):
        if path.name in {"all_results.json", "summary.json", "stats_report.json", "cases_eval_report.json"}:
            continue
        yield path


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _conversation_from_trajectory(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversation = trajectory.get("conversation") or trajectory.get("messages")
    if not isinstance(conversation, list):
        raise ValueError("missing conversation/messages")
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


def _extract_turn_gts(
    *,
    path: Path,
    trajectory: Dict[str, Any],
    conversation: List[Dict[str, Any]],
    sequence: Dict[str, Any],
) -> List[List[str]]:
    turns = trajectory.get("turns")
    if isinstance(turns, list) and turns:
        result: List[List[str]] = []
        for turn in turns:
            gt = turn.get("gt_survivors") or turn.get("golden_hypotheses") or turn.get("golden")
            if gt is None:
                raise ValueError(f"{path} has a trajectory turn without golden hypotheses")
            result.append(sorted(str(item) for item in gt))
        return result

    users = _user_messages(conversation)
    if all("golden_hypotheses" in msg for msg in users):
        return [sorted(str(item) for item in msg["golden_hypotheses"]) for msg in users]

    ground_truth = sequence.get("ground_truth")
    if isinstance(ground_truth, list) and ground_truth:
        result = []
        for item in ground_truth:
            gt = item.get("survivors") or item.get("golden_hypotheses") or item.get("gt_survivors")
            if gt is None:
                raise ValueError(f"{path} has a ground_truth item without survivors")
            result.append(sorted(str(x) for x in gt))
        return result

    raise ValueError(f"{path} does not expose per-turn golden hypotheses")


def _infer_challenge_type(payload: Dict[str, Any], trajectory: Dict[str, Any], fallback: str | None = None) -> str:
    raw = (
        payload.get("mode")
        or payload.get("challenge_type")
        or payload.get("challenge_sequence", {}).get("challenge_type")
        or trajectory.get("mode")
        or trajectory.get("challenge_type")
        or fallback
        or ""
    )
    value = str(raw).lower()
    if "failed_stay" in value:
        return "failed_stay"
    if "failed_update" in value or "misrecord" in value or "correction" in value:
        return "failed_update"
    if value:
        return value
    raise ValueError("unable to infer mode")


def _oracle(payload: Dict[str, Any], trajectory: Dict[str, Any]) -> str:
    value = (
        payload.get("oracle")
        or payload.get("rule_name")
        or payload.get("challenge_sequence", {}).get("oracle")
        or payload.get("challenge_sequence", {}).get("rule_name")
        or trajectory.get("oracle")
        or trajectory.get("rule_name")
        or ""
    )
    if not value:
        raise ValueError("missing oracle/rule_name")
    return str(value)


def _turn_matches(trajectory: Dict[str, Any]) -> List[bool]:
    conversation = _conversation_from_trajectory(trajectory)
    assistants = _assistant_messages(conversation)
    matches: List[bool] = []
    for msg in assistants:
        if "model_matches_golden" in msg:
            matches.append(bool(msg["model_matches_golden"]))
        elif "model_hypotheses" in msg and "golden_hypotheses" in msg:
            matches.append(set(msg["model_hypotheses"]) == set(msg["golden_hypotheses"]))
    if matches:
        return matches
    turns = trajectory.get("turns") or []
    for turn in turns:
        if "model_matches_golden" in turn:
            matches.append(bool(turn["model_matches_golden"]))
        elif "hypotheses" in turn and "gt_survivors" in turn:
            matches.append(set(turn["hypotheses"]) == set(turn["gt_survivors"]))
    return matches


def _prefix_turn_count(mode: str, n_turns: int) -> int:
    if mode == "failed_stay":
        return min(2, n_turns)
    if mode == "failed_update":
        return min(3, n_turns)
    return max(n_turns - 1, 0)


def _first_failed_turn(matches: List[bool], turn_indices: Sequence[int]) -> int | None:
    for idx in turn_indices:
        if idx < len(matches) and not matches[idx]:
            return idx
    return None


def _first_failed_turn_after_correct_prefix(matches: List[bool], turn_indices: Sequence[int]) -> int | None:
    for idx in turn_indices:
        if idx >= len(matches):
            continue
        if all(matches[prefix_idx] for prefix_idx in range(idx)) and not matches[idx]:
            return idx
    return None


def _select_error_turn(
    payload: Dict[str, Any],
    *,
    mode: str,
    category_kind: str,
    seed: int,
    path: Path,
) -> int:
    repeats = payload.get("repeat_trajectories")
    if not isinstance(repeats, list) or not repeats:
        raise ValueError(f"{path} missing repeat_trajectories")

    first_traj = repeats[0].get("trajectory") or {}
    n_turns = len(_user_messages(_conversation_from_trajectory(first_traj)))
    prefix_turns = _prefix_turn_count(mode, n_turns)

    if category_kind == "insufficient_capability":
        window = list(range(prefix_turns))
        counter: Counter[int] = Counter()
        for repeat in repeats:
            trajectory = repeat.get("trajectory")
            if not isinstance(trajectory, dict):
                continue
            failed_turn = _first_failed_turn_after_correct_prefix(_turn_matches(trajectory), window)
            if failed_turn is not None:
                counter[failed_turn] += 1
        if not counter:
            raise ValueError(f"{path} has no first failed capability turn")
        max_count = max(counter.values())
        candidates = sorted(turn for turn, count in counter.items() if count == max_count)
        rng = random.Random(_stable_seed(seed, path, mode, category_kind, "tie"))
        return rng.choice(candidates)

    if mode == "failed_update":
        return n_turns - 1
    post_turns = 2 if mode == "failed_stay" else n_turns - prefix_turns
    window = list(range(prefix_turns, min(prefix_turns + post_turns, n_turns)))
    fallback = min(prefix_turns, n_turns - 1)

    counter: Counter[int] = Counter()
    for repeat in repeats:
        trajectory = repeat.get("trajectory")
        if not isinstance(trajectory, dict):
            continue
        failed_turn = _first_failed_turn(_turn_matches(trajectory), window)
        if failed_turn is not None:
            counter[failed_turn] += 1

    if not counter:
        return fallback
    max_count = max(counter.values())
    candidates = sorted(turn for turn, count in counter.items() if count == max_count)
    rng = random.Random(_stable_seed(seed, path, mode, category_kind, "tie"))
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


def _choose_repeat_for_turn(
    payload: Dict[str, Any],
    selected_turn: int,
    path: Path,
    seed: int,
    *,
    require_correct_prefix: bool = False,
) -> Dict[str, Any]:
    repeats = payload.get("repeat_trajectories")
    if not isinstance(repeats, list) or not repeats:
        raise ValueError(f"{path} missing repeat_trajectories")
    eligible = []
    for repeat in repeats:
        trajectory = repeat.get("trajectory")
        if not isinstance(trajectory, dict):
            continue
        matches = _turn_matches(trajectory)
        prefix_ok = not require_correct_prefix or all(
            prefix_idx < len(matches) and matches[prefix_idx]
            for prefix_idx in range(selected_turn)
        )
        if prefix_ok and selected_turn < len(matches) and not matches[selected_turn]:
            eligible.append(repeat)
    if not eligible:
        eligible = []
        for repeat in repeats:
            trajectory = repeat.get("trajectory")
            if not isinstance(trajectory, dict):
                continue
            if not require_correct_prefix:
                eligible.append(repeat)
                continue
            matches = _turn_matches(trajectory)
            if all(prefix_idx < len(matches) and matches[prefix_idx] for prefix_idx in range(selected_turn)):
                eligible.append(repeat)
        if not eligible:
            raise ValueError(f"{path} has no repeat with correct prefix before turn {selected_turn}")
    rng = random.Random(_stable_seed(seed, path, selected_turn, "repeat"))
    return rng.choice(eligible)


def _convert_eval_case(path: Path, fallback_mode: str | None = None) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    repeats = payload.get("repeat_trajectories")
    if not isinstance(repeats, list) or not repeats:
        raise ValueError(f"{path} missing repeat_trajectories")
    trajectory = repeats[0].get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError(f"{path} missing first repeat trajectory")
    conversation = _conversation_from_trajectory(trajectory)
    users = _user_messages(conversation)
    turn_gts = _extract_turn_gts(
        path=path,
        trajectory=trajectory,
        conversation=conversation,
        sequence=payload.get("challenge_sequence") or {},
    )
    if len(users) != len(turn_gts):
        raise ValueError(f"{path} has {len(users)} user prompts but {len(turn_gts)} golden labels")
    mode = _infer_challenge_type(payload, trajectory, fallback_mode)
    return {
        "case_id": str(payload.get("experiment_id") or path.stem),
        "challenge_type": mode,
        "cbm_challenge_type": normalize_cbm_challenge_type(mode),
        "oracle": _oracle(payload, trajectory),
        "system_prompt": _extract_system_prompt(conversation),
        "turns": [
            {"prompt": str(users[idx]["content"]), "golden": turn_gts[idx]}
            for idx in range(len(users))
        ],
        "source_file": str(path),
        "source_category": payload.get("category"),
    }


def _convert_train_case(path: Path, category_kind: str, *, seed: int) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    repeats = payload.get("repeat_trajectories")
    if not isinstance(repeats, list) or not repeats:
        raise ValueError(f"{path} missing repeat_trajectories")
    first_trajectory = repeats[0].get("trajectory")
    if not isinstance(first_trajectory, dict):
        raise ValueError(f"{path} missing first repeat trajectory")

    mode = _infer_challenge_type(payload, first_trajectory)
    selected_turn = _select_error_turn(payload, mode=mode, category_kind=category_kind, seed=seed, path=path)
    selected_repeat = _choose_repeat_for_turn(
        payload,
        selected_turn,
        path,
        seed,
        require_correct_prefix=True,
    )
    trajectory = selected_repeat["trajectory"]
    conversation = _conversation_from_trajectory(trajectory)
    users = _user_messages(conversation)
    turn_gts = _extract_turn_gts(
        path=path,
        trajectory=trajectory,
        conversation=conversation,
        sequence=payload.get("challenge_sequence") or {},
    )
    if selected_turn >= len(users) or selected_turn >= len(turn_gts):
        raise ValueError(f"{path} selected turn {selected_turn} out of range")

    return {
        "case_id": f"{payload.get('experiment_id') or path.stem}_turn{selected_turn}",
        "challenge_type": mode,
        "cbm_challenge_type": normalize_cbm_challenge_type(mode),
        "oracle": _oracle(payload, trajectory),
        "system_prompt": _extract_system_prompt(conversation),
        "messages": _build_training_messages(conversation, selected_turn),
        "gt_survivors": turn_gts[selected_turn],
        "selected_turn": selected_turn,
        "selected_prompt": str(users[selected_turn]["content"]),
        "source_file": str(path),
        "source_category": category_kind,
        "source_repeat_index": selected_repeat.get("repeat_index"),
    }


def _case_sort_key(case: Dict[str, Any]) -> tuple[str, str, str]:
    return (str(case.get("challenge_type", "")), str(case.get("oracle", "")), str(case.get("case_id", "")))


def _category_dirs(input_dir: str | Path, category_kind: str) -> List[Path]:
    base = Path(input_dir)
    if base.name == category_kind:
        return [base]
    candidate = base / category_kind
    if candidate.exists():
        return [candidate]
    return [path for path in sorted(base.rglob(category_kind)) if path.is_dir()]


def build_train_cases_from_target_dirs(
    target_dirs: Sequence[str],
    *,
    excluded_oracles: Iterable[str] = (),
    max_valid_cases_per_target: int | None = None,
    max_insufficient_cases_per_target: int | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    excluded = {str(item) for item in excluded_oracles}
    cases: List[Dict[str, Any]] = []
    for target_dir in target_dirs:
        for category_kind in TRAIN_CATEGORY_SUBDIRS:
            paths: List[Path] = []
            for category_dir in _category_dirs(target_dir, category_kind):
                paths.extend(_iter_json_files(category_dir))
            selected = sorted(set(paths))
            if category_kind == VALID_SUBDIR and max_valid_cases_per_target is not None:
                selected = selected[:max_valid_cases_per_target]
            if category_kind == "insufficient_capability" and max_insufficient_cases_per_target is not None:
                selected = selected[:max_insufficient_cases_per_target]
            for path in selected:
                try:
                    case = _convert_train_case(path, category_kind, seed=seed)
                except ValueError as exc:
                    print(f"[case-conversion] skip train case {path}: {exc}")
                    continue
                if case["oracle"] not in excluded:
                    cases.append(case)
    cases.sort(key=_case_sort_key)
    if not cases:
        raise ValueError(f"No valid train cases found in: {', '.join(target_dirs)}")
    return cases


def build_eval_cases_from_dirs(
    input_dirs: Sequence[str],
    *,
    excluded_oracles: Iterable[str] = (),
    max_cases_per_target: int | None = None,
) -> List[Dict[str, Any]]:
    excluded = {str(item) for item in excluded_oracles}
    cases: List[Dict[str, Any]] = []
    for input_dir in input_dirs:
        paths: List[Path] = []
        for category_dir in _category_dirs(input_dir, VALID_SUBDIR):
            paths.extend(_iter_json_files(category_dir))
        selected = sorted(set(paths))
        if max_cases_per_target is not None:
            selected = selected[:max_cases_per_target]
        for path in selected:
            try:
                case = _convert_eval_case(path)
            except ValueError:
                continue
            if case["oracle"] not in excluded:
                cases.append(case)
    cases.sort(key=_case_sort_key)
    if not cases:
        raise ValueError(f"No valid eval cases found in: {', '.join(input_dirs)}")
    return cases


def _write_cases(cases: List[Dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scenario A train/test case JSON files")
    parser.add_argument("--build-scope", choices=("train", "test", "both"), default="both")
    parser.add_argument("--train-target-dirs", nargs="+", default=None)
    parser.add_argument("--test-target-dirs", nargs="+", default=None)
    parser.add_argument("--train-dirs", nargs="+", default=None, help="Backward-compatible alias for train targets.")
    parser.add_argument("--test-dirs", nargs="+", default=None, help="Backward-compatible alias for test targets.")
    parser.add_argument("--train-output-path", type=str, default=None)
    parser.add_argument("--test-output-path", type=str, default=None)
    parser.add_argument("--exclude-oracles", nargs="*", default=[])
    parser.add_argument("--max-train-valid-cases-per-target", type=int, default=None)
    parser.add_argument("--max-train-insufficient-cases-per-target", type=int, default=None)
    parser.add_argument("--max-test-cases-per-target", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_dirs = args.train_target_dirs or args.train_dirs
    test_dirs = args.test_target_dirs or args.test_dirs
    run_train = args.build_scope in {"train", "both"}
    run_test = args.build_scope in {"test", "both"}

    if run_train:
        if not train_dirs:
            raise ValueError(f"--build-scope {args.build_scope} requires train target directories")
        if not args.train_output_path:
            raise ValueError("--train-target-dirs requires --train-output-path")
        train_cases = build_train_cases_from_target_dirs(
            train_dirs,
            excluded_oracles=args.exclude_oracles,
            max_valid_cases_per_target=args.max_train_valid_cases_per_target,
            max_insufficient_cases_per_target=args.max_train_insufficient_cases_per_target,
            seed=args.seed,
        )
        _write_cases(train_cases, args.train_output_path)
        print(f"Saved {len(train_cases)} train cases to {args.train_output_path}")

    if run_test:
        if not test_dirs:
            raise ValueError(f"--build-scope {args.build_scope} requires test target directories")
        if not args.test_output_path:
            raise ValueError("--test-target-dirs requires --test-output-path")
        test_cases = build_eval_cases_from_dirs(
            test_dirs,
            excluded_oracles=args.exclude_oracles,
            max_cases_per_target=args.max_test_cases_per_target,
        )
        _write_cases(test_cases, args.test_output_path)
        print(f"Saved {len(test_cases)} test cases to {args.test_output_path}")

    if not run_train and not run_test:
        raise ValueError("No build scope was selected.")


if __name__ == "__main__":
    main()
