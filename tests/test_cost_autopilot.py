"""Pytest suite for cost_autopilot. Fully offline."""

import json

import pytest

from cost_autopilot import (
    BudgetExceeded,
    BudgetPolicy,
    CostAwareRouter,
    MockProvider,
    PricingTable,
    UsageTracker,
)
from cost_autopilot.router import NoEligibleModel


@pytest.fixture()
def pricing():
    return PricingTable()


@pytest.fixture()
def tracker(pricing):
    t = UsageTracker(db_path=":memory:", pricing=pricing)
    yield t
    t.close()


# ---- pricing / tracker math ----


def test_price_entry_cost_math(pricing):
    # gpt-4o-mini: 0.00015/1k in, 0.0006/1k out
    assert pricing.cost("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075)
    assert pricing.cost("gpt-4o-mini", 0, 0) == 0.0


def test_tracker_logs_cost_from_pricing(tracker):
    rec = tracker.log_call("gpt-4o-mini", 2000, 1000, tag="t1")
    assert rec.cost_usd == pytest.approx(0.0009)
    assert rec.tag == "t1"
    assert rec.call_id is not None


def test_tracker_rejects_negative_tokens(tracker):
    with pytest.raises(ValueError):
        tracker.log_call("gpt-4o-mini", -1, 10)


def test_tracker_unknown_model_raises(tracker):
    with pytest.raises(KeyError):
        tracker.log_call("not-a-real-model", 10, 10)


def test_total_cost_filters(tracker):
    tracker.log_call("gpt-4o-mini", 1000, 0, tag="a")
    tracker.log_call("gpt-4o-mini", 1000, 0, tag="b")
    assert tracker.total_cost() == pytest.approx(0.0003)
    assert tracker.total_cost(tag="a") == pytest.approx(0.00015)
    by_tag = tracker.cost_by_tag()
    assert by_tag["a"] == pytest.approx(0.00015)
    assert by_tag["b"] == pytest.approx(0.00015)


def test_cost_by_model(tracker):
    tracker.log_call("gpt-4o-mini", 1000, 0)
    tracker.log_call("claude-3-haiku", 1000, 0)
    by_model = tracker.cost_by_model()
    assert by_model["gpt-4o-mini"] == pytest.approx(0.00015)
    assert by_model["claude-3-haiku"] == pytest.approx(0.00025)


def test_records_roundtrip(tracker):
    tracker.log_call("gpt-4o-mini", 500, 250, tag="x")
    recs = list(tracker.records(tag="x"))
    assert len(recs) == 1
    assert recs[0].prompt_tokens == 500
    assert recs[0].completion_tokens == 250


# ---- pricing overrides ----


def test_pricing_set_override(pricing):
    pricing.set("my-model", "custom", 0.01, 0.02)
    assert pricing.cost("my-model", 1000, 1000) == pytest.approx(0.03)


def test_pricing_json_roundtrip(pricing, tmp_path):
    path = tmp_path / "prices.json"
    pricing.set("my-model", "custom", 0.01, 0.02)
    pricing.save(path)
    loaded = PricingTable.load(path)
    assert loaded.cost("my-model", 1000, 1000) == pytest.approx(0.03)
    assert loaded.cost("gpt-4o-mini", 1000, 0) == pytest.approx(0.00015)


def test_pricing_json_override_format(tmp_path):
    path = tmp_path / "override.json"
    path.write_text(
        json.dumps({"tiny": {"provider": "x", "input_per_1k": 0.5, "output_per_1k": 1.5}})
    )
    table = PricingTable.load(path)
    assert table.cost("tiny", 2000, 1000) == pytest.approx(2.5)


# ---- budgets ----


def test_budget_alert_fires_once(tracker):
    fired = []
    policy = BudgetPolicy(total_cap=0.0001, alerts=[lambda s, spent, cap: fired.append((s, spent, cap))])
    tracker.log_call("gpt-4o-mini", 1000, 0)  # $0.00015 > $0.0001 cap
    breaches = policy.check(tracker)
    assert len(breaches) == 1
    assert breaches[0][0] == "total"
    assert len(fired) == 1
    policy.check(tracker)  # alert must not fire twice
    assert len(fired) == 1


def test_budget_blocking_raises(tracker):
    policy = BudgetPolicy(total_cap=0.0001, block_on_breach=True)
    tracker.log_call("gpt-4o-mini", 1000, 0)
    with pytest.raises(BudgetExceeded):
        policy.check_call(tracker)


def test_budget_no_breach_no_alert(tracker):
    fired = []
    policy = BudgetPolicy(total_cap=100.0, alerts=[lambda s, spent, cap: fired.append(s)])
    tracker.log_call("gpt-4o-mini", 1000, 0)
    assert policy.check(tracker) == []
    assert fired == []


def test_budget_project_cap(tracker):
    fired = []
    policy = BudgetPolicy(
        project_tag="proj", project_cap=0.0002,
        alerts=[lambda s, spent, cap: fired.append(s)],
    )
    tracker.log_call("gpt-4o-mini", 1000, 0, tag="proj")  # 0.00015
    assert policy.check(tracker) == []
    tracker.log_call("gpt-4o-mini", 1000, 0, tag="proj")  # 0.00030 total
    breaches = policy.check(tracker)
    assert [b[0] for b in breaches] == ["project:proj"]
    assert fired == ["project:proj"]


def test_budget_alerts_resettable(tracker):
    fired = []
    policy = BudgetPolicy(total_cap=0.0001, alerts=[lambda s, spent, cap: fired.append(s)])
    tracker.log_call("gpt-4o-mini", 1000, 0)
    policy.check(tracker)
    policy.reset_alerts()
    policy.check(tracker)
    assert len(fired) == 2


# ---- router ----


@pytest.fixture()
def router(pricing):
    return CostAwareRouter(
        pricing=pricing,
        quality={"gpt-4o": 92.0, "gpt-4o-mini": 82.0, "claude-3-haiku": 78.0},
    )


def test_router_picks_cheapest_meeting_floor(router):
    d = router.route(1000, 500, min_quality=80.0)
    assert d.model == "gpt-4o-mini"  # cheapest of the >=80 models
    assert d.quality_score == 82.0


def test_router_raises_floor(router):
    d = router.route(1000, 500, min_quality=90.0)
    assert d.model == "gpt-4o"  # only one clearing 90


def test_router_no_eligible_model(router):
    with pytest.raises(NoEligibleModel):
        router.route(1000, 500, min_quality=99.0)


def test_router_skips_unknown_quality(router):
    d = router.route(
        1000, 500, min_quality=50.0, candidates=["gpt-4o-mini", "gemini-1.5-flash"]
    )
    assert d.model == "gpt-4o-mini"  # gemini has no quality score -> skipped


def test_router_estimate_math(router):
    assert router.estimate("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075)


def test_router_batch_estimate(router):
    result = router.estimate_batch([(1000, 500)] * 10, min_quality=80.0)
    assert result["calls"] == 10
    assert result["total_estimated_cost_usd"] == pytest.approx(10 * 0.00045)
    assert result["per_model"]["gpt-4o-mini"]["calls"] == 10


def test_router_tie_break_by_name(pricing):
    r = CostAwareRouter(
        pricing=pricing,
        quality={"m-a": 80.0, "m-b": 80.0},
        tie_break="name",
    )
    pricing.set("m-a", "x", 0.001, 0.001)
    pricing.set("m-b", "x", 0.001, 0.001)
    d = r.route(1000, 1000, min_quality=80.0, candidates=["m-b", "m-a"])
    assert d.model == "m-a"  # equal cost -> alphabetical


# ---- mock provider ----


def test_mock_provider_deterministic():
    p = MockProvider()
    a = p.complete("gpt-4o-mini", "hello")
    b = p.complete("gpt-4o-mini", "hello")
    assert a == b
    c = p.complete("gpt-4o-mini", "different prompt")
    assert (a.prompt_tokens, a.completion_tokens) != (
        c.prompt_tokens,
        c.completion_tokens,
    )


def test_mock_provider_respects_max_tokens():
    p = MockProvider()
    r = p.complete("gpt-4o-mini", "hello", max_tokens=5)
    assert r.completion_tokens <= 5


def test_adapters_require_env_without_sdk(monkeypatch):
    from cost_autopilot.providers import AnthropicAdapter, OpenAIAdapter

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAIAdapter().complete("gpt-4o-mini", "hi")
    with pytest.raises(RuntimeError):
        AnthropicAdapter().complete("claude-3-haiku", "hi")
