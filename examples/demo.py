"""End-to-end demo (fully offline): tracking + budget alert + cost-aware routing.

Run:  python examples/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cost_autopilot import (  # noqa: E402
    BudgetPolicy,
    CostAwareRouter,
    MockProvider,
    PricingTable,
    UsageTracker,
)

# Quality scores are illustrative, user-supplied numbers on a 0-100 scale.
QUALITY = {
    "gpt-4o": 92.0,
    "claude-3-5-sonnet": 91.0,
    "gemini-1.5-pro": 88.0,
    "gpt-4o-mini": 82.0,
    "claude-3-haiku": 78.0,
    "gemini-1.5-flash": 76.0,
}

PROMPTS = [
    ("summarize-q3", "Summarize the Q3 sales report for the exec team."),
    ("draft-email", "Draft a polite follow-up email to a vendor."),
    ("classify-ticket", "Classify this support ticket: 'my export button is broken'."),
    ("extract-fields", "Extract name, date, and total from this invoice text."),
    ("translate-note", "Translate this release note into Spanish."),
]


def main() -> None:
    pricing = PricingTable()
    tracker = UsageTracker(db_path=":memory:", pricing=pricing)
    provider = MockProvider()
    router = CostAwareRouter(pricing=pricing, quality=QUALITY)

    alerts: list[str] = []
    policy = BudgetPolicy(
        daily_cap=0.001,  # tiny caps so the alerts fire in the demo
        project_tag="demo",
        project_cap=0.0008,
        alerts=[lambda scope, spent, cap: alerts.append(
            f"ALERT: {scope} budget breached: spent ${spent:.4f} of ${cap:.4f} cap"
        )],
    )

    print("== routing each prompt to the cheapest model with quality >= 85 ==")
    for tag, prompt in PROMPTS:
        decision = router.route(
            prompt_tokens=120, completion_tokens=80, min_quality=85.0
        )
        result = provider.complete(decision.model, prompt)
        record = tracker.log_call(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            provider="mock",
            tag="demo",
        )
        policy.check(tracker)
        print(
            f"[{tag}] -> {decision.model} "
            f"(est ${decision.estimated_cost_usd:.5f}, actual ${record.cost_usd:.5f})"
        )

    print("\n== budget alerts ==")
    for alert in alerts:
        print(alert)
    assert alerts, "expected at least one budget alert to fire in the demo"

    print("\n== batch dry-run estimate (quality floor 80) ==")
    batch = [(150, 100)] * 1000
    estimate = router.estimate_batch(batch, min_quality=80.0)
    print(f"1000 calls: est. ${estimate['total_estimated_cost_usd']:.2f}")
    for model, info in estimate["per_model"].items():
        print(f"  {model}: {info['calls']} calls, ${info['cost_usd']:.2f}")

    print("\n== spend summary ==")
    print(f"total: ${tracker.total_cost():.4f}")
    print(f"by model: {tracker.cost_by_model()}")
    print("\nDemo complete. No network calls were made.")
    tracker.close()


if __name__ == "__main__":
    main()
