# LLM Cost Autopilot

A provider-agnostic Python package that **tracks LLM API usage and cost**, **enforces budgets**, and **routes calls to the cheapest model that meets a quality floor**.

Built as a weekend portfolio project: the boring infrastructure layer that sits between your app and the LLM vendors and answers "where did the money go, and could we have spent less?"

## Install

```bash
pip install -e .
```

Requires Python 3.9+. Only dependency is `pytest` (for the test suite); the package itself is stdlib-only.

## Quickstart

```python
from cost_autopilot import (
    BudgetPolicy, CostAwareRouter, MockProvider, PricingTable, UsageTracker,
)

pricing = PricingTable()                       # estimates; override via JSON (see below)
tracker = UsageTracker(db_path="usage.db", pricing=pricing)
router = CostAwareRouter(pricing=pricing, quality={"gpt-4o": 92.0, "gpt-4o-mini": 82.0})

alerts = []
policy = BudgetPolicy(daily_cap=5.0, alerts=[lambda s, spent, cap: alerts.append((s, spent, cap))])

provider = MockProvider()                      # offline; swap in OpenAIAdapter for real calls
decision = router.route(prompt_tokens=800, completion_tokens=200, min_quality=80.0)
result = provider.complete(decision.model, "Summarize this report...")
tracker.log_call(result.model, result.prompt_tokens, result.completion_tokens,
                 provider="mock", tag="summarizer")
policy.check(tracker)

print(f"routed to {decision.model} (est ${decision.estimated_cost_usd:.5f})")
print(f"total spend: ${tracker.total_cost():.4f}")
```

Run the offline demo:

```bash
python examples/demo.py
```

Run the tests:

```bash
pytest
```

## Architecture

```
cost_autopilot/
├── pricing.py    # Per-1K-token price table (USD). JSON-overridable estimates.
├── tracker.py    # UsageTracker: per-call token/cost records in SQLite.
├── budgets.py    # BudgetPolicy: daily/project/total caps, alert callbacks, optional blocking.
├── router.py     # CostAwareRouter: cheapest model meeting a user-supplied quality floor.
└── providers.py  # MockProvider (deterministic, offline) + OpenAI/Anthropic adapter stubs.
```

- **tracker** records every call (model, provider, tag, token counts, timestamp, computed cost) in SQLite and answers `total_cost()`, `cost_by_model()`, `cost_by_tag()`.
- **budgets** checks caps against a tracker; on breach it fires each registered alert callback once per scope (reset with `reset_alerts()`), and optionally raises `BudgetExceeded` to block further calls.
- **router** picks the cheapest candidate model whose *user-supplied* quality score clears the floor; models with no score are skipped (never vouched for). `estimate_batch` dry-runs cost for a batch of token counts.
- **providers** ships a deterministic `MockProvider` for demos/tests. The OpenAI/Anthropic adapters read API keys **only** from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` environment variables, never hardcode or log them, and raise a clear error if the vendor SDK isn't installed.

## Overriding prices

Prices are estimates (see Limitations). Pin your own:

```python
table = PricingTable()
table.set("my-fine-tune", "openai", input_per_1k=0.003, output_per_1k=0.012)
table.save("my_prices.json")
# later:
table = PricingTable.load("my_prices.json")
```

## Limitations (honest)

- **Prices are estimates and change.** The bundled table was captured from public provider pricing pages in late 2026. Providers change pricing without notice; verify against the vendor's current pricing page before relying on any number. Override via JSON for anything that matters.
- **Quality scores are user-supplied, not measured by this tool.** The router trusts whatever scores you hand it (from your own evals, benchmarks, or judgment). Garbage scores in, garbage routing out.
- **Token counts from real providers come from the vendor's usage fields**, which this tool passes through unchanged; it does not re-tokenize or audit them.
- **No latency, rate-limit, or availability awareness.** Routing optimizes on cost subject to the quality floor only.
- **SQLite is single-file storage**, fine for a service or notebook, not a substitute for a real observability pipeline at scale.

## License

MIT — see [LICENSE](LICENSE).
