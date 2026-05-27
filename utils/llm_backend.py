from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError as exc:
    OpenAI = None  # type: ignore[misc, assignment]
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None

Message = Dict[str, str]
Messages = List[Message]


class VLLMBackend:
    """vLLM backend for chat inference, with optional LoRA support."""

    def __init__(
        self,
        model_path: str,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        adapter_path: Optional[str] = None,
        max_lora_rank: int = 64,
        language_model_only: bool = False,
        preserve_thinking_tags: bool = True,
        enable_thinking: Optional[bool] = None,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        disable_custom_all_reduce: bool = False,
        enforce_eager: bool = False,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "Failed to import vLLM (often a broken torch/NCCL stack, not a missing package). "
                f"Original error: {exc}"
            ) from exc

        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if disable_custom_all_reduce:
            llm_kwargs["disable_custom_all_reduce"] = True
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if language_model_only:
            llm_kwargs["language_model_only"] = True
        if dtype:
            llm_kwargs["dtype"] = {
                "bf16": "bfloat16",
                "fp16": "float16",
                "fp32": "float32",
            }.get(dtype, dtype)
        if adapter_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank

        self.llm = LLM(**llm_kwargs)
        self.SamplingParams = SamplingParams
        self.preserve_thinking_tags = preserve_thinking_tags
        self.enable_thinking = enable_thinking
        self._lora_request = None
        self.sampling_overrides: Dict[str, Any] = {}
        if adapter_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, adapter_path)

    def chat_completion(
        self,
        messages: Messages,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> str:
        return self.batch_chat_completion(
            [messages],
            temperature=temperature,
            max_tokens=max_tokens,
            use_tqdm=True,
        )[0]

    def batch_chat_completion(
        self,
        messages_batch: List[Messages],
        temperature: Optional[float] = 0.7,
        max_tokens: int = 800,
        use_tqdm: bool = True,
    ) -> List[str]:
        if not messages_batch:
            return []

        tokenizer = self.llm.get_tokenizer()
        template_kwargs: Dict[str, Any] = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        prompts = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if isinstance(prompts, str):
            prompts = [prompts]

        sampling_kwargs: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "stop": [tokenizer.eos_token] if tokenizer.eos_token else None,
        }
        if temperature is not None:
            sampling_kwargs["temperature"] = max(temperature, 0.0)
        sampling_kwargs.update({k: v for k, v in self.sampling_overrides.items() if v is not None})

        sampling_params = self.SamplingParams(**sampling_kwargs)
        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=self._lora_request,
            use_tqdm=use_tqdm,
        )
        return [
            self._format_completion(output.outputs[0].text, prompt=prompt)
            for output, prompt in zip(outputs, prompts)
        ]

    def _format_completion(self, text: str, prompt: Optional[str] = None) -> str:
        text = text.strip()
        if not self.preserve_thinking_tags:
            return text

        prompt_opens_think = (
            isinstance(prompt, str)
            and prompt.rstrip().endswith("<think>")
        )
        model_continues_think = prompt_opens_think or "</think>" in text
        if model_continues_think and text and not text.lstrip().startswith("<think>"):
            return "<think>\n" + text
        return text


class APIBackend:
    """OpenAI-compatible chat completions via openai.OpenAI (errors propagate as-is)."""

    def __init__(
        self,
        *,
        api_base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        max_workers: int = 8,
        enable_thinking: Optional[bool] = True,
    ):
        if OpenAI is None:
            raise ImportError(
                "openai package is required for APIBackend. "
                f"Original error: {_OPENAI_IMPORT_ERROR}"
            ) from _OPENAI_IMPORT_ERROR

        base = (api_base_url or "").strip()
        if not base:
            raise ValueError("api_base_url is required for API backend")
        while base.endswith("/"):
            base = base[:-1]
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")].rstrip("/")

        self.model_name = (model_name or "").strip()
        if not self.model_name:
            raise ValueError("model_name is required for API backend")

        self._client = OpenAI(api_key=api_key or None, base_url=base)
        self.max_workers = max(1, int(max_workers))
        self.enable_thinking = enable_thinking
        self.sampling_overrides: Dict[str, Any] = {}

    def chat_completion(
        self,
        messages: Messages,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 800,
    ) -> str:
        return self.batch_chat_completion(
            [messages],
            temperature=temperature,
            max_tokens=max_tokens,
            use_tqdm=True,
        )[0]

    def batch_chat_completion(
        self,
        messages_batch: List[Messages],
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 800,
        use_tqdm: bool = True,
    ) -> List[str]:
        _ = use_tqdm
        if not messages_batch:
            return []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(messages_batch))) as ex:
            return list(
                ex.map(
                    lambda messages: self._chat_create(messages, temperature, max_tokens),
                    messages_batch,
                )
            )

    def _chat_create(
        self,
        messages: Messages,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = max(temperature, 0.0)

        for key in ("top_p", "presence_penalty", "frequency_penalty", "stop"):
            value = self.sampling_overrides.get(key)
            if value is not None:
                kwargs[key] = value

        extra: Dict[str, Any] = {}
        if self.enable_thinking is not None:
            extra["enable_thinking"] = self.enable_thinking
        extra.update({
            k: self.sampling_overrides[k]
            for k in ("top_k", "min_p", "repetition_penalty")
            if self.sampling_overrides.get(k) is not None
        })
        if extra:
            kwargs["extra_body"] = extra

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content: Any = msg.content
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            content = "".join(text_parts)
        text = str(content or "").strip()
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            text = f"<think>\n{reasoning.strip()}\n</think>\n{text}"
        return text


__all__ = [
    "VLLMBackend",
    "APIBackend",
]
