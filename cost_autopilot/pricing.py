"""Per-1K-token price tables for common LLM models.

IMPORTANT: All prices below are ESTIMATES captured from public provider pricing
pages as of late 2026. Provider prices change frequently; never treat these as
authoritative. Override with your own numbers via a JSON file — see
PricingTable.load / PricingTable.save, or ship a file next to your config.

JSON override format::

    {
        "gpt-4o": {"provider": "openai", "input_per_1k": 0.0025, "output_per_1k": 0.010},
        "my-custom-model": {"provider": "openai", "input_per_1k": 0.001, "output_per_1k": 0.002}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRICING_PATH = Path(__file__).with_name("pricing_estimates.json")


@dataclass(frozen=True)
class PriceEntry:
    """Price per 1,000 tokens (USD) for one model."""

    provider: str
    input_per_1k: float
    output_per_1k: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost in USD for a single call's token counts."""
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        return (prompt_tokens / 1000.0) * self.input_per_1k + (
            completion_tokens / 1000.0
        ) * self.output_per_1k


# NOTE: ESTIMATES ONLY. Providers change pricing without notice. These were
# captured from public pricing pages in late 2026; verify before relying on
# them. See the JSON override mechanism for pinning your own numbers.
_DEFAULT_PRICES: dict[str, dict[str, float | str]] = {
    "gpt-4o": {"provider": "openai", "input_per_1k": 0.0025, "output_per_1k": 0.010},
    "gpt-4o-mini": {"provider": "openai", "input_per_1k": 0.00015, "output_per_1k": 0.0006},
    "gpt-4-turbo": {"provider": "openai", "input_per_1k": 0.010, "output_per_1k": 0.030},
    "claude-3-5-sonnet": {"provider": "anthropic", "input_per_1k": 0.003, "output_per_1k": 0.015},
    "claude-3-haiku": {"provider": "anthropic", "input_per_1k": 0.00025, "output_per_1k": 0.00125},
    "claude-3-opus": {"provider": "anthropic", "input_per_1k": 0.015, "output_per_1k": 0.075},
    "gemini-1.5-flash": {"provider": "google", "input_per_1k": 0.000075, "output_per_1k": 0.0003},
    "gemini-1.5-pro": {"provider": "google", "input_per_1k": 0.00125, "output_per_1k": 0.005},
    "llama-3.1-70b": {"provider": "meta-hosted", "input_per_1k": 0.00088, "output_per_1k": 0.00088},
    "llama-3.1-8b": {"provider": "meta-hosted", "input_per_1k": 0.00018, "output_per_1k": 0.00018},
    "mistral-large": {"provider": "mistral", "input_per_1k": 0.002, "output_per_1k": 0.006},
}


class PricingTable:
    """Mutable, JSON-overridable registry of per-1K-token model prices (USD)."""

    def __init__(self, prices: dict[str, PriceEntry] | None = None) -> None:
        self._prices: dict[str, PriceEntry] = dict(prices) if prices else {
            name: PriceEntry(
                provider=str(spec["provider"]),
                input_per_1k=float(spec["input_per_1k"]),
                output_per_1k=float(spec["output_per_1k"]),
            )
            for name, spec in _DEFAULT_PRICES.items()
        }

    def get(self, model: str) -> PriceEntry:
        try:
            return self._prices[model]
        except KeyError:
            raise KeyError(
                f"no price entry for model {model!r}; add it via PricingTable.set() "
                "or a JSON override file"
            ) from None

    def set(
        self, model: str, provider: str, input_per_1k: float, output_per_1k: float
    ) -> None:
        self._prices[model] = PriceEntry(
            provider=provider, input_per_1k=input_per_1k, output_per_1k=output_per_1k
        )

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return self.get(model).cost(prompt_tokens, completion_tokens)

    def models(self) -> list[str]:
        return sorted(self._prices)

    def save(self, path: str | Path) -> None:
        data = {
            name: {
                "provider": e.provider,
                "input_per_1k": e.input_per_1k,
                "output_per_1k": e.output_per_1k,
            }
            for name, e in self._prices.items()
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PricingTable":
        """Load a price table from a JSON override file (see module docstring)."""
        data = json.loads(Path(path).read_text())
        prices = {
            name: PriceEntry(
                provider=str(spec["provider"]),
                input_per_1k=float(spec["input_per_1k"]),
                output_per_1k=float(spec["output_per_1k"]),
            )
            for name, spec in data.items()
        }
        return cls(prices)
