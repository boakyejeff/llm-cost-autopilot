"""SQLite-backed tracking of LLM API usage and cost."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .pricing import PricingTable


@dataclass(frozen=True)
class UsageRecord:
    call_id: int | None
    timestamp: str  # ISO-8601 UTC
    model: str
    provider: str
    tag: str  # call-site tag, e.g. "rag-answer", "summarizer"
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_log (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT '',
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_log_ts ON usage_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_log_tag ON usage_log(tag);
"""


class UsageTracker:
    """Record per-call LLM usage in SQLite and answer cost queries.

    Thread-safe. ``db_path=":memory:"`` keeps everything in RAM (useful for tests).
    """

    def __init__(
        self,
        db_path: str | Path = "usage.db",
        pricing: PricingTable | None = None,
        cost_fn: Callable[[str, int, int], float] | None = None,
    ) -> None:
        self._pricing = pricing or PricingTable()
        self._cost_fn = cost_fn
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _price(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        if self._cost_fn is not None:
            return self._cost_fn(model, prompt_tokens, completion_tokens)
        return self._pricing.cost(model, prompt_tokens, completion_tokens)

    def log_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        provider: str = "",
        tag: str = "",
        timestamp: str | None = None,
    ) -> UsageRecord:
        """Record one LLM call and return the stored record (with cost)."""
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        entry = self._pricing.get(model) if self._cost_fn is None else None
        provider = provider or (entry.provider if entry else "unknown")
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        cost = self._price(model, prompt_tokens, completion_tokens)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO usage_log (timestamp, model, provider, tag,"
                " prompt_tokens, completion_tokens, cost_usd)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, model, provider, tag, prompt_tokens, completion_tokens, cost),
            )
            self._conn.commit()
            call_id = cur.lastrowid
        return UsageRecord(
            call_id=call_id,
            timestamp=ts,
            model=model,
            provider=provider,
            tag=tag,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )

    def total_cost(self, tag: str | None = None, since: str | None = None) -> float:
        """Total USD spent, optionally filtered by tag and/or ISO-8601 start time."""
        query = "SELECT COALESCE(SUM(cost_usd), 0.0) FROM usage_log"
        clauses, params = [], []
        if tag is not None:
            clauses.append("tag = ?")
            params.append(tag)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._lock:
            return float(self._conn.execute(query, params).fetchone()[0])

    def cost_by_model(self, tag: str | None = None) -> dict[str, float]:
        query = "SELECT model, COALESCE(SUM(cost_usd), 0.0) FROM usage_log"
        params: list = []
        if tag is not None:
            query += " WHERE tag = ?"
            params.append(tag)
        query += " GROUP BY model"
        with self._lock:
            return {row[0]: row[1] for row in self._conn.execute(query, params)}

    def cost_by_tag(self) -> dict[str, float]:
        with self._lock:
            return {
                row[0]: row[1]
                for row in self._conn.execute(
                    "SELECT tag, COALESCE(SUM(cost_usd), 0.0) FROM usage_log GROUP BY tag"
                )
            }

    def records(
        self, tag: str | None = None, limit: int = 1000
    ) -> Iterable[UsageRecord]:
        query = (
            "SELECT call_id, timestamp, model, provider, tag,"
            " prompt_tokens, completion_tokens, cost_usd FROM usage_log"
        )
        params: list = []
        if tag is not None:
            query += " WHERE tag = ?"
            params.append(tag)
        query += " ORDER BY call_id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [UsageRecord(*row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
