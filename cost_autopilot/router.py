"""Cost-aware model routing: pick the cheapest model that meets a quality floor.

Quality scores are USER-SUPPLIED (e.g. from your own evals or benchmarks), not
measured by this tool. Scores must be on a common scale (say 0-100) per task.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pricing import PricingTable


@dataclass(frozen=True)
class RouteDecision:
    model: str
    estimated_cost_usd: float
    quality_score: float
    reason: str


class NoEligibleModel(Exception):
    """Raised when no candidate model meets the quality floor."""


class CostAwareRouter:
    """Route each task to the cheapest model whose quality score clears the floor.

    ``quality`` maps model name -> quality score on a common scale.
    ``tie_break`` decides among equal-cost models ("quality" picks the higher
    score; "name" picks deterministically by model name).
    """

    def __init__(
        self,
        pricing: PricingTable | None = None,
        quality: dict[str, float] | None = None,
        tie_break: str = "quality",
    ) -> None:
        if tie_break not in ("quality", "name"):
            raise ValueError("tie_break must be 'quality' or 'name'")
        self.pricing = pricing or PricingTable()
        self.quality: dict[str, float] = dict(quality or {})
        self.tie_break = tie_break

    def set_quality(self, model: str, score: float) -> None:
        self.quality[model] = score

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return self.pricing.cost(model, prompt_tokens, completion_tokens)

    def route(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        min_quality: float,
        candidates: list[str] | None = None,
    ) -> RouteDecision:
        """Return the cheapest candidate model with quality >= min_quality."""
        pool = candidates if candidates is not None else self.pricing.models()
        eligible: list[tuple[float, float, str]] = []  # (cost, -quality, model)
        for model in pool:
            score = self.quality.get(model)
            if score is None:
                continue  # unknown quality: cannot vouch for the model, skip it
            if score < min_quality:
                continue
            cost = self.estimate(model, prompt_tokens, completion_tokens)
            eligible.append((cost, -score, model))
        if not eligible:
            raise NoEligibleModel(
                f"no candidate model meets quality floor {min_quality} "
                f"(candidates considered: {pool})"
            )
        eligible.sort(key=lambda e: (e[0], e[2] if self.tie_break == "name" else e[1]))
        cost, neg_score, model = eligible[0]
        return RouteDecision(
            model=model,
            estimated_cost_usd=cost,
            quality_score=-neg_score,
            reason=(
                f"cheapest of {len(eligible)} models meeting quality >= {min_quality}"
            ),
        )

    def estimate_batch(
        self,
        prompts: list[tuple[int, int]],
        min_quality: float,
        candidates: list[str] | None = None,
    ) -> dict:
        """Dry-run: estimate total cost of routing a batch of (prompt, completion) token pairs."""
        decisions = [
            self.route(p, c, min_quality, candidates) for p, c in prompts
        ]
        total = sum(d.estimated_cost_usd for d in decisions)
        per_model: dict[str, dict[str, float | int]] = {}
        for d in decisions:
            entry = per_model.setdefault(d.model, {"calls": 0, "cost_usd": 0.0})
            entry["calls"] += 1  # type: ignore[operator]
            entry["cost_usd"] += d.estimated_cost_usd  # type: ignore[operator]
        return {
            "total_estimated_cost_usd": total,
            "calls": len(decisions),
            "per_model": per_model,
            "decisions": decisions,
        }
