#!/usr/bin/env python3
"""Render a chat prompt and report whether historical assistant think is kept."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "U0"},
    {
        "role": "assistant",
        "content": "<think>ABC</think>\n\n<hypothesis>X</hypothesis>",
    },
    {"role": "user", "content": "U1"},
]


def _load_messages(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_MESSAGES
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict) and "messages" in data:
        data = data["messages"]
    if not isinstance(data, list):
        raise ValueError("messages file must contain a messages list or an object with a messages field")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check how a model chat_template renders assistant <think> history."
    )
    parser.add_argument("--model-path", required=True, help="HF/vLLM model directory.")
    parser.add_argument(
        "--custom-chat-template-file",
        default=None,
        help="Optional jinja template file to override tokenizer.chat_template.",
    )
    parser.add_argument(
        "--messages-json",
        default=None,
        help="Optional JSON file containing a messages list, or {'messages': [...]} object.",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Instantiate vLLM LLM and use llm.get_tokenizer(); requires a working GPU/vLLM runtime.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    messages = _load_messages(args.messages_json)

    if args.use_vllm:
        from vllm import LLM

        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.4,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )
        tokenizer = llm.get_tokenizer()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.custom_chat_template_file:
        tokenizer.chat_template = Path(args.custom_chat_template_file).read_text(encoding="utf-8")

    prompt = tokenizer.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(prompt, list):
        prompt = prompt[0]

    print("===== rendered prompt =====")
    print(prompt)
    print("===== checks =====")
    history_think_block = "<think>\nABC\n</think>"
    print(f"contains_ABC={ 'ABC' in prompt }")
    print(f"contains_history_think_block={ history_think_block in prompt }")
    print(f"endswith_generation_think={ prompt.rstrip().endswith('<think>') }")


if __name__ == "__main__":
    main()
