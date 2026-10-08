"""LLM Cost Autopilot: track LLM API usage and cost, enforce budgets, route to the cheapest capable model."""

from .budgets import BudgetExceeded, BudgetPolicy
from .pricing import DEFAULT_PRICING_PATH, PriceEntry, PricingTable
from .providers import AnthropicAdapter, MockProvider, OpenAIAdapter, ProviderAdapter
from .router import CostAwareRouter, RouteDecision
from .tracker import UsageRecord, UsageTracker

__version__ = "0.1.0"
__all__ = [
    "AnthropicAdapter",
    "BudgetExceeded",
    "BudgetPolicy",
    "CostAwareRouter",
    "MockProvider",
    "OpenAIAdapter",
    "PriceEntry",
    "PricingTable",
    "DEFAULT_PRICING_PATH",
    "ProviderAdapter",
    "RouteDecision",
    "UsageRecord",
    "UsageTracker",
]
