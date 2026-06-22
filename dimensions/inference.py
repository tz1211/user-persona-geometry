"""Lightweight inference wrapper for vLLM (batch) and OpenAI-compatible APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dimensions.postprocess import strip_think_tags

# Token ID for </think> in Qwen3 vocabulary
_THINK_CLOSE_TOKEN_ID = 151668


@dataclass
class GenerationResult:
    """Rich output from a single generation."""

    raw_text: str
    thinking_content: str | None
    response_content: str
    thinking_tokens: int
    response_tokens: int


def _split_think_tokens(token_ids: tuple[int, ...] | list[int]) -> tuple[int, int]:
    """Return (thinking_token_count, response_token_count) by splitting at </think>.

    Finds the last occurrence of _THINK_CLOSE_TOKEN_ID. Everything up to and
    including that token is counted as thinking; the rest as response.
    """
    ids = list(token_ids)
    try:
        last_idx = len(ids) - 1 - ids[::-1].index(_THINK_CLOSE_TOKEN_ID)
        thinking_count = last_idx + 1  # include the </think> token itself
        response_count = len(ids) - thinking_count
    except ValueError:
        thinking_count = 0
        response_count = len(ids)
    return thinking_count, response_count


class InferenceEngine:
    """Unified interface for batch text generation.

    Parameters
    ----------
    model_cfg : dict
        Parsed model YAML config. Required keys: ``model_id``, ``backend``.
        Optional: ``dtype``, ``gpu_memory_utilization``, ``strip_think_tags``.
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_id: str = model_cfg["model_id"]
        self.backend: str = model_cfg.get("backend", "vllm")
        self.strip_think_tags: bool = model_cfg.get("strip_think_tags", False)
        self._cfg = model_cfg
        self._engine: Any = None

    # ------------------------------------------------------------------
    # Lazy init — only load when first needed
    # ------------------------------------------------------------------

    def _init_vllm(self) -> None:
        from vllm import LLM

        self._engine = LLM(
            model=self.model_id,
            dtype=self._cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=self._cfg.get("gpu_memory_utilization", 0.9),
            trust_remote_code=True,
            seed=self._cfg.get("seed", 42),
        )

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        if self.backend == "vllm":
            self._init_vllm()
        elif self.backend == "openai":
            # Deferred — will init openai client when needed
            pass
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[dict[str, str]],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int = 2048,
    ) -> list[str]:
        """Run batch generation.

        Parameters
        ----------
        prompts : list of dict
            Each dict has keys ``"system"`` and ``"user"``.
        gen_kwargs : dict
            Sampling parameters (temperature, top_p, top_k, etc.).
        max_new_tokens : int
            Maximum tokens to generate per prompt.

        Returns
        -------
        list[str]
            Raw completion strings (one per prompt).
        """
        self._ensure_engine()
        if self.backend == "vllm":
            return self._generate_vllm(prompts, gen_kwargs, max_new_tokens)
        elif self.backend == "openai":
            return self._generate_openai(prompts, gen_kwargs, max_new_tokens)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def generate_chat_detailed(
        self,
        messages_batch: list[list[dict[str, str]]],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int = 2048,
    ) -> list[GenerationResult]:
        """Run chat generation from fully-formed message lists.

        This is used by multi-turn protocols where the benchmark needs to
        preserve assistant/user turns instead of flattening everything into
        one user message.
        """
        self._ensure_engine()
        if self.backend == "vllm":
            return self._generate_vllm_chat_detailed(
                messages_batch, gen_kwargs, max_new_tokens
            )
        elif self.backend == "openai":
            return self._generate_openai_chat_detailed(
                messages_batch, gen_kwargs, max_new_tokens
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_vllm(
        self,
        prompts: list[dict[str, str]],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int,
    ) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=gen_kwargs.get("temperature", 0.6),
            top_p=gen_kwargs.get("top_p", 0.95),
            top_k=gen_kwargs.get("top_k", 20),
            min_p=gen_kwargs.get("min_p", 0.0),
            presence_penalty=gen_kwargs.get("presence_penalty", 1.5),
            stop=gen_kwargs.get("stop", None),
        )

        # Build chat-style message lists for vLLM's chat interface
        conversations = []
        for p in prompts:
            messages: list[dict[str, str]] = []
            if p.get("system"):
                messages.append({"role": "system", "content": p["system"]})
            messages.append({"role": "user", "content": p["user"]})
            conversations.append(messages)

        outputs = self._engine.chat(conversations, sampling_params=params)
        return [out.outputs[0].text for out in outputs]

    def _generate_vllm_detailed(
        self,
        prompts: list[dict[str, str]],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int,
    ) -> list[GenerationResult]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=gen_kwargs.get("temperature", 0.6),
            top_p=gen_kwargs.get("top_p", 0.95),
            top_k=gen_kwargs.get("top_k", 20),
            min_p=gen_kwargs.get("min_p", 0.0),
            presence_penalty=gen_kwargs.get("presence_penalty", 1.5),
            stop=gen_kwargs.get("stop", None),
        )

        conversations = []
        for p in prompts:
            messages: list[dict[str, str]] = []
            if p.get("system"):
                messages.append({"role": "system", "content": p["system"]})
            messages.append({"role": "user", "content": p["user"]})
            conversations.append(messages)

        return self._generate_vllm_chat_detailed(
            conversations, gen_kwargs, max_new_tokens
        )

    def _generate_vllm_chat_detailed(
        self,
        messages_batch: list[list[dict[str, str]]],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int,
    ) -> list[GenerationResult]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=gen_kwargs.get("temperature", 0.6),
            top_p=gen_kwargs.get("top_p", 0.95),
            top_k=gen_kwargs.get("top_k", 20),
            min_p=gen_kwargs.get("min_p", 0.0),
            presence_penalty=gen_kwargs.get("presence_penalty", 1.5),
            stop=gen_kwargs.get("stop", None),
        )

        outputs = self._engine.chat(messages_batch, sampling_params=params)
        results: list[GenerationResult] = []
        for out in outputs:
            raw = out.outputs[0].text
            token_ids = out.outputs[0].token_ids
            thinking_count, response_count = _split_think_tokens(token_ids)
            cleaned, think_content = strip_think_tags(raw) if self.strip_think_tags else (raw, None)
            results.append(GenerationResult(
                raw_text=raw,
                thinking_content=think_content,
                response_content=cleaned,
                thinking_tokens=thinking_count,
                response_tokens=response_count,
            ))
        return results

    def _generate_openai(
        self,
        prompts: list[dict[str, str]],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int,
    ) -> list[str]:
        """OpenAI-compatible API backend (for frontier model comparison)."""
        import openai

        client = openai.OpenAI()
        results: list[str] = []
        for p in prompts:
            messages: list[dict[str, str]] = []
            if p.get("system"):
                messages.append({"role": "system", "content": p["system"]})
            messages.append({"role": "user", "content": p["user"]})
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=gen_kwargs.get("temperature", 0.6),
                top_p=gen_kwargs.get("top_p", 0.95),
                presence_penalty=gen_kwargs.get("presence_penalty", 0.0),
            )
            results.append(response.choices[0].message.content or "")
        return results

    def _generate_openai_detailed(
        self,
        prompts: list[dict[str, str]],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int,
    ) -> list[GenerationResult]:
        """OpenAI backend — token counts approximated from text."""
        import openai

        messages_batch = []
        for p in prompts:
            messages: list[dict[str, str]] = []
            if p.get("system"):
                messages.append({"role": "system", "content": p["system"]})
            messages.append({"role": "user", "content": p["user"]})
            messages_batch.append(messages)
        return self._generate_openai_chat_detailed(
            messages_batch, gen_kwargs, max_new_tokens
        )

    def _generate_openai_chat_detailed(
        self,
        messages_batch: list[list[dict[str, str]]],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int,
    ) -> list[GenerationResult]:
        """OpenAI backend for fully-formed chat messages."""
        import openai

        client = openai.OpenAI()
        results: list[GenerationResult] = []
        for messages in messages_batch:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=gen_kwargs.get("temperature", 0.6),
                top_p=gen_kwargs.get("top_p", 0.95),
                presence_penalty=gen_kwargs.get("presence_penalty", 0.0),
            )
            raw = response.choices[0].message.content or ""
            cleaned, think_content = strip_think_tags(raw)
            # Approximate token counts from usage if available
            usage = getattr(response, "usage", None)
            total_tokens = usage.completion_tokens if usage else len(raw.split())
            think_tok = len(think_content.split()) if think_content else 0
            resp_tok = total_tokens - think_tok
            results.append(GenerationResult(
                raw_text=raw,
                thinking_content=think_content,
                response_content=cleaned,
                thinking_tokens=think_tok,
                response_tokens=max(0, resp_tok),
            ))
        return results

    def generate_detailed(
        self,
        prompts: list[dict[str, str]],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int = 2048,
    ) -> list[GenerationResult]:
        """Run batch generation and return rich GenerationResult objects.

        Each result includes thinking/response content split and token counts.

        Parameters
        ----------
        prompts : list of dict
            Each dict has keys ``"system"`` and ``"user"``.
        gen_kwargs : dict
            Sampling parameters. Supports extra key ``"stop"`` (list of stop strings).
        max_new_tokens : int
            Maximum tokens to generate per prompt.
        """
        self._ensure_engine()
        if self.backend == "vllm":
            return self._generate_vllm_detailed(prompts, gen_kwargs, max_new_tokens)
        elif self.backend == "openai":
            return self._generate_openai_detailed(prompts, gen_kwargs, max_new_tokens)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
