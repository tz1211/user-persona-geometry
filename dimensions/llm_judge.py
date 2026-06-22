"""Batched LLM judge backends for benchmark evaluation."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any


DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. Return only valid JSON. "
    "Do not include prose or markdown."
)


@dataclass
class JudgeConfig:
    backend: str = "vllm"
    model: str = DEFAULT_JUDGE_MODEL
    max_tokens: int = 4096
    temperature: float = 0.0
    api_base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_concurrency: int = 32
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    batch_size: int = 4
    max_model_len: int | None = 16384


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from a judge response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    obj = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError:
            pass
    return {}


class BaseJudge:
    """Common interface for batched judge calls."""

    def complete(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError

    def complete_json(self, prompts: list[str]) -> list[dict[str, Any]]:
        return [extract_json(text) for text in self.complete(prompts)]


class NoopJudge(BaseJudge):
    def complete(self, prompts: list[str]) -> list[str]:
        return ["" for _ in prompts]


class VLLMJudge(BaseJudge):
    """In-process vLLM judge. Best for GPU jobs."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self._llm = None

    def _ensure_llm(self):
        if self._llm is not None:
            return
        from vllm import LLM

        llm_kwargs: dict[str, Any] = dict(
            model=self.config.model,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=True,
        )
        if self.config.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.config.max_model_len
        self._llm = LLM(**llm_kwargs)

    def complete(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        self._ensure_llm()
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        batch_size = max(1, self.config.batch_size)
        texts: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            conversations = [
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                for prompt in chunk
            ]
            outputs = self._llm.chat(conversations, sampling_params=params)
            texts.extend(out.outputs[0].text for out in outputs)
        return texts


class OpenAICompatibleJudge(BaseJudge):
    """Async OpenAI-compatible judge for OpenAI, Azure-compatible, or vLLM server APIs."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    async def _complete_one(self, client, semaphore: asyncio.Semaphore, prompt: str) -> str:
        async with semaphore:
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""

    async def _complete_all(self, prompts: list[str]) -> list[str]:
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {}
        if self.config.api_base_url:
            kwargs["base_url"] = self.config.api_base_url
        api_key = os.environ.get(self.config.api_key_env)
        if api_key:
            kwargs["api_key"] = api_key
        elif self.config.api_base_url:
            kwargs["api_key"] = "EMPTY"
        client = AsyncOpenAI(**kwargs)
        semaphore = asyncio.Semaphore(self.config.api_concurrency)
        try:
            tasks = [self._complete_one(client, semaphore, prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)
        finally:
            await client.close()

    def complete(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        return asyncio.run(self._complete_all(prompts))


class AnthropicJudge(BaseJudge):
    """Async Anthropic Messages API judge."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    async def _complete_one(self, client, semaphore: asyncio.Semaphore, prompt: str) -> str:
        key_env = (
            self.config.api_key_env
            if self.config.api_key_env != "OPENAI_API_KEY"
            else "ANTHROPIC_API_KEY"
        )
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"Missing Anthropic judge API key in ${key_env}")
        url = self.config.api_base_url or "https://api.anthropic.com/v1/messages"
        async with semaphore:
            response = await client.post(
                url,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "system": _SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()
            parts = data.get("content", [])
            return "".join(
                part.get("text", "")
                for part in parts
                if part.get("type") == "text"
            )

    async def _complete_all(self, prompts: list[str]) -> list[str]:
        import httpx

        timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        semaphore = asyncio.Semaphore(self.config.api_concurrency)
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [self._complete_one(client, semaphore, prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)

    def complete(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        return asyncio.run(self._complete_all(prompts))


class GeminiJudge(BaseJudge):
    """Async Google Gemini generateContent API judge."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    async def _complete_one(self, client, semaphore: asyncio.Semaphore, prompt: str) -> str:
        key_env = (
            self.config.api_key_env
            if self.config.api_key_env != "OPENAI_API_KEY"
            else "GOOGLE_API_KEY"
        )
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"Missing Gemini judge API key in ${key_env}")
        base_url = (
            self.config.api_base_url
            or "https://generativelanguage.googleapis.com/v1beta"
        )
        model_path = (
            self.config.model
            if self.config.model.startswith("models/")
            else f"models/{self.config.model}"
        )
        url = f"{base_url.rstrip('/')}/{model_path}:generateContent"
        async with semaphore:
            response = await client.post(
                url,
                params={"key": api_key},
                json={
                    "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": self.config.temperature,
                        "maxOutputTokens": self.config.max_tokens,
                        "responseMimeType": "application/json",
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            return "".join(part.get("text", "") for part in parts)

    async def _complete_all(self, prompts: list[str]) -> list[str]:
        import httpx

        timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        semaphore = asyncio.Semaphore(self.config.api_concurrency)
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [self._complete_one(client, semaphore, prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)

    def complete(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        return asyncio.run(self._complete_all(prompts))


def build_judge(config: JudgeConfig) -> BaseJudge:
    backend = config.backend.lower()
    if backend in {"none", "off", "disabled"}:
        return NoopJudge()
    if backend == "vllm":
        return VLLMJudge(config)
    if backend in {"openai", "api", "openai-compatible"}:
        return OpenAICompatibleJudge(config)
    if backend == "anthropic":
        return AnthropicJudge(config)
    if backend in {"gemini", "google"}:
        return GeminiJudge(config)
    raise ValueError(f"Unknown judge backend: {config.backend}")
