"""Provider adapters: a deterministic offline MockProvider plus API-key-from-env stubs.

The OpenAI/Anthropic adapters are SHAPES of a real integration, not working
clients: they read API keys ONLY from environment variables, never hardcode
or log them, and raise a clear error if the vendor SDK is not installed.
Network calls are never made by this package's tests or demo.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CallResult:
    model: str
    prompt_tokens: int
    completion_tokens: int
    text: str


class ProviderAdapter(ABC):
    @abstractmethod
    def complete(self, model: str, prompt: str, max_tokens: int = 256) -> CallResult:
        """Run one completion. Implementations must not log API keys."""


class MockProvider(ProviderAdapter):
    """Deterministic offline provider for demos and tests (no network).

    Token counts are a simple deterministic function of the prompt text so
    demos are reproducible; the returned text is clearly synthetic.
    """

    def complete(self, model: str, prompt: str, max_tokens: int = 256) -> CallResult:
        digest = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
        prompt_tokens = 20 + (int(digest[:4], 16) % 180)  # 20-199
        completion_tokens = min(max_tokens, 10 + (int(digest[4:8], 16) % 90))
        return CallResult(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            text=f"[mock:{model}] synthetic completion (sha {digest[:8]})",
        )


class _EnvKeyMixin:
    provider_env_var: str = ""

    def _api_key(self) -> str:
        key = os.environ.get(self.provider_env_var, "")
        if not key:
            raise RuntimeError(
                f"{self.provider_env_var} is not set; export it before using this adapter"
            )
        return key  # never log this value


class OpenAIAdapter(_EnvKeyMixin, ProviderAdapter):
    """Adapter shape for OpenAI-style chat APIs. Requires the `openai` package."""

    provider_env_var = "OPENAI_API_KEY"

    def complete(self, model: str, prompt: str, max_tokens: int = 256) -> CallResult:
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "the `openai` package is not installed; pip install openai"
            ) from exc
        client = openai.OpenAI(api_key=self._api_key())
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        usage = resp.usage
        return CallResult(
            model=model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            text=resp.choices[0].message.content or "",
        )


class AnthropicAdapter(_EnvKeyMixin, ProviderAdapter):
    """Adapter shape for Anthropic's API. Requires the `anthropic` package."""

    provider_env_var = "ANTHROPIC_API_KEY"

    def complete(self, model: str, prompt: str, max_tokens: int = 256) -> CallResult:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "the `anthropic` package is not installed; pip install anthropic"
            ) from exc
        client = anthropic.Anthropic(api_key=self._api_key())
        resp = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        usage = resp.usage
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return CallResult(
            model=model,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            text=text,
        )
