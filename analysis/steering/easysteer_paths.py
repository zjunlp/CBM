"""Resolve the local EasySteer checkout used by steering scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("easysteer_config.json")


def _expand_path(value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_easysteer_root() -> Path:
    """Return EasySteer root from env override or JSON config."""

    env_root = os.environ.get("EASYSTEER_ROOT")
    if env_root:
        return _expand_path(env_root)

    config_path = Path(os.environ.get("EASYSTEER_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            "EasySteer root is not configured. Set EASYSTEER_ROOT or create "
            f"{config_path} with an 'easysteer_root' string."
        )

    payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("easysteer_root"):
        raise ValueError(f"{config_path} must contain an 'easysteer_root' string")
    return _expand_path(str(payload["easysteer_root"]))


def add_easysteer_to_sys_path() -> Path:
    """Prepend EasySteer and vllm-steer paths to sys.path, then return root."""

    import sys

    root = resolve_easysteer_root()
    for path in (root / "vllm-steer", root):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root
