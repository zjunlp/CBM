"""Compatibility patches for the local EasySteer/vLLM checkout."""

from __future__ import annotations

import os
from typing import Iterable, List


DEFAULT_EXTRA_DECODER_LAYERS = [
    "Qwen3_5DecoderLayer",
]


def patch_supported_decoder_layers(extra_layers: Iterable[str] | None = None) -> List[str]:
    """Register local model decoder layer names with EasySteer wrappers.

    EasySteer recognizes decoder layers by exact class name. The local
    Qwen3.5 implementation subclasses Qwen3NextDecoderLayer but its class name
    is Qwen3_5DecoderLayer, so it must be present in the wrapper allowlist for
    both hidden-state capture and steer-vector intervention.
    """

    from vllm.steer_vectors import config as steer_config

    requested = list(DEFAULT_EXTRA_DECODER_LAYERS)
    env_value = os.environ.get("EASYSTEER_EXTRA_DECODER_LAYERS", "")
    requested.extend(part.strip() for part in env_value.split(",") if part.strip())
    if extra_layers is not None:
        requested.extend(str(item).strip() for item in extra_layers if str(item).strip())

    added: List[str] = []
    for layer_name in requested:
        if layer_name not in steer_config.SUPPORTED_DECODER_LAYERS:
            steer_config.SUPPORTED_DECODER_LAYERS.append(layer_name)
            added.append(layer_name)
    return added
