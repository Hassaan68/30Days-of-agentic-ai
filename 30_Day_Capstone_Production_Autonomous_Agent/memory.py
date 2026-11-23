"""Short-term conversation state and SQLite-backed long-term incident memory."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9_-]+", text.lower()) if len(tok) > 1]


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class ShortTermMemory:
    """Token-bounded conversation history preserving the system prompt."""

    def __init__(self, system_prompt: str, max_tokens: int = 3200) -> None:
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._trim()

    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def token_count(self) -> int:
        return sum(_estimate_tokens(m["content"]) for m in self._messages)

    def _trim(self) -> None:
        """Drop oldest non-system messages when total tokens exceed budget."""
        while len(self._messages) > 2 and self.token_count() > self.max_tokens:
            self._messages.pop(1)


@dataclass(frozen=True)
class MemoryRecord:
    """One durable runbook or resolved-incident record stored in SQLite."""

    record_id: int
    kind: str
    title: str
    content: str
    tags: str
    created_at: str


class LongTermMemory:
    """SQLite-backed incident & runbook memory with cosine-style lexical search."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                kind       TEXT NOT NULL,
                title      TEXT NOT NULL,
                content    TEXT NOT NULL,
                tags       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        self._seed_defaults_if_empty()

    def _seed_defaults_if_empty(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count > 0:
            return
        self.remember(
            kind="runbook",
            title="checkout-api 502 DB Pool Exhaustion Runbook",
            content=(
                "When checkout-api spikes above 5% HTTP 502 with 'connection pool exhausted' "
                "immediately after a release, verify health metrics and error logs, open an "
                "incident ticket, and rollback to the previous stable version with operator approval."
            ),
            tags="checkout-api 502 rollback db-pool",
        )
        self.remember(
            kind="incident",
            title="INC-2026-081 payments-worker queue lag",
            content=(
                "payments-worker lag was caused by an expired webhook signing secret; "
                "rotating the secret and restarting the worker cleared the backlog in 4 minutes."
            ),
            tags="payments-worker queue lag restart",
        )

    def remember(self, kind: str, title: str, content: str, tags: str = "") -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO memories (kind, title, content, tags, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (kind, title, content, tags, now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def recall(self, query: str, top_k: int = 3) -> list[MemoryRecord]:
        """Rank stored memories by TF cosine similarity against the query."""
        q_counts = Counter(_tokenize(query))
        if not q_counts:
            return []
        q_norm = math.sqrt(sum(v * v for v in q_counts.values()))

        rows = self._conn.execute(
            "SELECT id, kind, title, content, tags, created_at FROM memories"
        ).fetchall()

        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            rec = MemoryRecord(*row)
            doc_counts = Counter(_tokenize(f"{rec.title} {rec.content} {rec.tags}"))
            dot = sum(q_counts[t] * doc_counts[t] for t in q_counts)
            if dot == 0:
                continue
            doc_norm = math.sqrt(sum(v * v for v in doc_counts.values()))
            sim = dot / (q_norm * doc_norm) if doc_norm else 0.0
            scored.append((sim, rec))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def close(self) -> None:
        self._conn.close()
