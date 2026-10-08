"""Budget policies: caps on LLM spend with alert callbacks and optional blocking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

AlertCallback = Callable[[str, float, float], None]  # (scope, spent, cap)


class BudgetExceeded(Exception):
    """Raised by BudgetPolicy.check_call when blocking is enabled and a cap is hit."""

    def __init__(self, scope: str, spent: float, cap: float) -> None:
        super().__init__(
            f"budget exceeded for {scope}: spent ${spent:.4f} of ${cap:.4f} cap"
        )
        self.scope = scope
        self.spent = spent
        self.cap = cap


@dataclass
class BudgetPolicy:
    """USD caps on LLM spend.

    - ``daily_cap``: spend since UTC midnight.
    - ``project_cap``: spend for one tag (set ``project_tag``).
    - ``total_cap``: all-time spend in the tracker.
    - ``alerts``: callbacks invoked once per breach per scope
      (signature ``(scope: str, spent: float, cap: float)``).
    - ``block_on_breach``: if True, ``check_call`` raises ``BudgetExceeded``
      instead of just alerting.
    """

    daily_cap: float | None = None
    project_cap: float | None = None
    project_tag: str = ""
    total_cap: float | None = None
    alerts: list[AlertCallback] = field(default_factory=list)
    block_on_breach: bool = False
    _alerted: set[str] = field(default_factory=set, repr=False)

    def add_alert(self, callback: AlertCallback) -> None:
        self.alerts.append(callback)

    def _utc_midnight_iso(self) -> str:
        return datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

    def _breaches(self, tracker: "UsageTracker") -> list[tuple[str, float, float]]:  # noqa: F821
        from .tracker import UsageTracker  # local import avoids a cycle at module load

        breaches: list[tuple[str, float, float]] = []
        if self.total_cap is not None:
            spent = tracker.total_cost()
            if spent >= self.total_cap:
                breaches.append(("total", spent, self.total_cap))
        if self.daily_cap is not None:
            spent = tracker.total_cost(since=self._utc_midnight_iso())
            if spent >= self.daily_cap:
                breaches.append(("daily", spent, self.daily_cap))
        if self.project_cap is not None:
            spent = tracker.total_cost(tag=self.project_tag)
            if spent >= self.project_cap:
                breaches.append((f"project:{self.project_tag}", spent, self.project_cap))
        return breaches

    def check(self, tracker: "UsageTracker") -> list[tuple[str, float, float]]:  # noqa: F821
        """Check caps against a tracker; fire alert callbacks for new breaches.

        Returns the list of currently-breached (scope, spent, cap) tuples.
        """
        breaches = self._breaches(tracker)
        for scope, spent, cap in breaches:
            if scope not in self._alerted:
                self._alerted.add(scope)
                for cb in self.alerts:
                    cb(scope, spent, cap)
        return breaches

    def check_call(self, tracker: "UsageTracker") -> None:  # noqa: F821
        """Check caps and raise BudgetExceeded if blocking is enabled and breached."""
        breaches = self.check(tracker)
        if breaches and self.block_on_breach:
            scope, spent, cap = breaches[0]
            raise BudgetExceeded(scope, spent, cap)

    def reset_alerts(self) -> None:
        """Allow alerts to fire again (e.g. at the start of a new day)."""
        self._alerted.clear()
