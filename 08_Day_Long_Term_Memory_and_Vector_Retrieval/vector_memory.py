"""Day 8: Long-Term Memory and Vector Retrieval.

Implements a lightweight, persistent SQLite vector memory store with:
1. Cosine similarity search over dense embeddings.
2. Pluggable OpenAI-compatible /embeddings provider AND an offline deterministic
   n-gram + token feature-hashing provider (--demo, zero external installs).
3. Mandatory user_id isolation, category/metadata filtering, similarity-based
   deduplication, exponential time decay, and stale-memory pruning.

Run offline demo (no API key needed):
    python 08_Day_Long_Term_Memory_and_Vector_Retrieval/vector_memory.py --demo

Run with live OpenAI-compatible embeddings:
    export OPENAI_API_KEY="your-key"
    python 08_Day_Long_Term_Memory_and_Vector_Retrieval/vector_memory.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Embedding providers & vector math
# ---------------------------------------------------------------------------

class EmbeddingProvider(Protocol):
    """Interface for turning text into normalized dense float vectors."""

    @property
    def name(self) -> str:
        ...

    def embed(self, text: str) -> list[float]:
        ...


class LocalHashEmbeddingProvider:
    """Deterministic local embedding using word tokens, bigrams, and char n-grams.

    Uses signed feature hashing into a fixed-size vector followed by L2
    normalization. Not a substitute for neural embeddings on broad synonyms,
    but great for offline testing, demos, and inspecting cosine similarity math.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9_]+")

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return f"local-hash-{self.dim}d"

    def _add_feature(self, vec: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw_int = int.from_bytes(digest, "little")
        bucket = raw_int % self.dim
        sign = 1.0 if (raw_int >> 63) == 0 else -1.0
        vec[bucket] += sign * weight

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = self._TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec

        for tok in tokens:
            self._add_feature(vec, f"w:{tok}", 2.0)
            padded = f"<{tok}>"
            for n in (3, 4):
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        self._add_feature(vec, f"c{n}:{padded[i : i + n]}", 0.65)

        for i in range(len(tokens) - 1):
            self._add_feature(vec, f"b:{tokens[i]}_{tokens[i + 1]}", 1.4)

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec


class OpenAIEmbeddingProvider:
    """Calls an OpenAI-compatible /embeddings endpoint using stdlib urllib."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-3-small",
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed(self, text: str) -> list[float]:
        body = {"model": self.model, "input": text}
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embeddings HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embeddings network error: {exc.reason}") from exc

        vec = [float(x) for x in parsed["data"][0]["embedding"]]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0.0 else vec


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity in [-1.0, 1.0] between two vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Data models & SQLite-backed vector memory store
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """One durable long-term memory entry."""

    id: str
    user_id: str
    category: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    access_count: int = 1
    embedding: list[float] = field(default_factory=list)


@dataclass
class SearchResult:
    """Ranked recall hit with transparent score breakdown."""

    record: MemoryRecord
    similarity: float
    recency_score: float
    combined_score: float


class SQLiteVectorMemory:
    """Persistent long-term vector memory backed by Python's built-in sqlite3."""

    def __init__(self, db_path: str | Path, embedder: EmbeddingProvider) -> None:
        self.db_path = str(db_path)
        self.embedder = embedder
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                importance REAL NOT NULL,
                created_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL,
                embedding_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_user_category
                ON memories(user_id, category);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add_memory(
        self,
        user_id: str,
        content: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        created_at: float | None = None,
        dedup_threshold: float = 0.88,
    ) -> tuple[str, str]:
        """Store a memory, updating an existing record if a near-duplicate exists.

        Returns (memory_id, action) where action is 'inserted' or 'updated_duplicate'.
        """
        now = created_at if created_at is not None else time.time()
        importance = max(0.0, min(1.0, float(importance)))
        meta = dict(metadata or {})
        vec = self.embedder.embed(content)

        # Check near-duplicates within the same user_id and category.
        existing_rows = self.conn.execute(
            """
            SELECT id, content, metadata_json, importance, access_count, embedding_json
            FROM memories
            WHERE user_id = ? AND category = ?
            """,
            (user_id, category),
        ).fetchall()

        best_match_id: str | None = None
        best_sim = -1.0
        best_row: sqlite3.Row | None = None

        for row in existing_rows:
            existing_vec = json.loads(row["embedding_json"])
            sim = cosine_similarity(vec, existing_vec)
            if sim >= dedup_threshold and sim > best_sim:
                best_sim = sim
                best_match_id = row["id"]
                best_row = row

        if best_match_id is not None and best_row is not None:
            merged_meta = json.loads(best_row["metadata_json"])
            merged_meta.update(meta)
            merged_importance = max(float(best_row["importance"]), importance)
            new_access_count = int(best_row["access_count"]) + 1
            self.conn.execute(
                """
                UPDATE memories
                SET content = ?,
                    metadata_json = ?,
                    importance = ?,
                    last_accessed_at = ?,
                    access_count = ?,
                    embedding_json = ?
                WHERE id = ?
                """,
                (
                    content,
                    json.dumps(merged_meta, sort_keys=True),
                    merged_importance,
                    now,
                    new_access_count,
                    json.dumps(vec),
                    best_match_id,
                ),
            )
            self.conn.commit()
            return best_match_id, "updated_duplicate"

        mem_id = f"mem_{uuid.uuid4().hex[:10]}"
        self.conn.execute(
            """
            INSERT INTO memories (
                id, user_id, category, content, metadata_json,
                importance, created_at, last_accessed_at, access_count, embedding_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem_id,
                user_id,
                category,
                content,
                json.dumps(meta, sort_keys=True),
                importance,
                now,
                now,
                1,
                json.dumps(vec),
            ),
        )
        self.conn.commit()
        return mem_id, "inserted"

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        min_similarity: float = 0.12,
        half_life_days: float = 30.0,
        now: float | None = None,
        touch: bool = True,
    ) -> list[SearchResult]:
        """Retrieve ranked memories scoped strictly to user_id."""
        now_ts = now if now is not None else time.time()
        query_vec = self.embedder.embed(query)

        sql = "SELECT * FROM memories WHERE user_id = ?"
        params: list[Any] = [user_id]
        if category is not None:
            sql += " AND category = ?"
            params.append(category)

        rows = self.conn.execute(sql, params).fetchall()
        scored: list[SearchResult] = []

        for row in rows:
            meta = json.loads(row["metadata_json"])
            if metadata_filter:
                if any(meta.get(k) != v for k, v in metadata_filter.items()):
                    continue

            vec = json.loads(row["embedding_json"])
            sim = cosine_similarity(query_vec, vec)
            if sim < min_similarity:
                continue

            age_days = max(0.0, (now_ts - float(row["last_accessed_at"])) / 86400.0)
            recency = 0.5 ** (age_days / max(0.1, half_life_days))
            importance = float(row["importance"])

            # Multiplicative gating prevents brand-new unrelated memories from
            # outranking genuinely relevant older memories, while letting freshness
            # and importance break ties among relevant matches.
            combined = sim * (0.65 + 0.25 * recency + 0.10 * importance)
            record = MemoryRecord(
                id=row["id"],
                user_id=row["user_id"],
                category=row["category"],
                content=row["content"],
                metadata=meta,
                importance=importance,
                created_at=float(row["created_at"]),
                last_accessed_at=float(row["last_accessed_at"]),
                access_count=int(row["access_count"]),
                embedding=vec,
            )
            scored.append(
                SearchResult(
                    record=record,
                    similarity=sim,
                    recency_score=recency,
                    combined_score=combined,
                )
            )

        scored.sort(key=lambda r: r.combined_score, reverse=True)
        top_hits = scored[:top_k]

        if touch and top_hits:
            for hit in top_hits:
                self.conn.execute(
                    """
                    UPDATE memories
                    SET last_accessed_at = ?, access_count = access_count + 1
                    WHERE id = ?
                    """,
                    (now_ts, hit.record.id),
                )
            self.conn.commit()

        return top_hits

    def prune_stale(
        self,
        max_age_days: float = 90.0,
        max_importance: float = 0.4,
        now: float | None = None,
    ) -> int:
        """Delete old, low-importance memories that have not been accessed recently."""
        now_ts = now if now is not None else time.time()
        cutoff = now_ts - max_age_days * 86400.0
        cursor = self.conn.execute(
            """
            DELETE FROM memories
            WHERE last_accessed_at < ? AND importance <= ?
            """,
            (cutoff, max_importance),
        )
        self.conn.commit()
        return cursor.rowcount

    def count(self, user_id: str | None = None) -> int:
        if user_id is None:
            row = self.conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE user_id = ?", (user_id,)
            ).fetchone()
        return int(row["c"])


# ---------------------------------------------------------------------------
# Runnable demonstration
# ---------------------------------------------------------------------------

def _should_use_demo(force_demo: bool) -> bool:
    if force_demo or os.getenv("AGENT_DEMO_MODE", "").strip() == "1":
        return True
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    return not api_key and not is_local


def run_demo(embedder: EmbeddingProvider, db_path: Path) -> None:
    print(f"=== Day 8: Long-Term Memory & Vector Retrieval ({embedder.name}) ===")
    print(f"SQLite store: {db_path}\n")

    now = time.time()
    day = 86400.0

    store = SQLiteVectorMemory(db_path=db_path, embedder=embedder)

    # 1. Seed memories across two users with different timestamps & categories.
    seeds = [
        (
            "alice",
            "preference",
            "User prefers concise bullet points and Python type hints in code answers.",
            {"source": "onboarding"},
            0.85,
            now - 10 * day,
        ),
        (
            "alice",
            "preference",
            "Preferred analytics database is MySQL for staging reports.",
            {"topic": "database"},
            0.55,
            now - 75 * day,  # Older preference
        ),
        (
            "alice",
            "preference",
            "Migrated analytics database preference to PostgreSQL with read replicas.",
            {"topic": "database"},
            0.80,
            now - 1 * day,  # Newer preference on same topic
        ),
        (
            "alice",
            "project",
            "Working on checkout-api migration in eu-west-1 targeting Python 3.12.",
            {"service": "checkout-api"},
            0.90,
            now - 3 * day,
        ),
        (
            "alice",
            "note",
            "Temporary reminder to check staging banner color before Friday standup.",
            {"ephemeral": True},
            0.20,
            now - 120 * day,  # Old low-importance note -> candidate for pruning
        ),
        (
            "bob",
            "preference",
            "Bob prefers Rust examples and stores analytics database tables in ClickHouse.",
            {"topic": "database"},
            0.90,
            now - 1 * day,
        ),
    ]

    print("1) Inserting initial memories:")
    for user_id, category, content, meta, imp, ts in seeds:
        mem_id, action = store.add_memory(
            user_id=user_id,
            content=content,
            category=category,
            metadata=meta,
            importance=imp,
            created_at=ts,
        )
        age_days = int((now - ts) / day)
        print(f"   [{action:17s}] {mem_id} user={user_id:5s} cat={category:10s} age={age_days:3d}d")

    # 2. Test similarity-based deduplication.
    print("\n2) Testing deduplication with near-identical preference phrasing:")
    dup_id, dup_action = store.add_memory(
        user_id="alice",
        content="User prefers concise bullet points and Python type hints in code answers!",
        category="preference",
        metadata={"confirmed_via": "session_12"},
        importance=0.90,
        created_at=now,
    )
    print(
        f"   Result: {dup_action} ({dup_id}) | total alice memories: {store.count('alice')} (unchanged!)"
    )

    # 3. Search with recency decay + strict user isolation.
    print("\n3) Recalling Alice's database preference (recency decay ranks PostgreSQL above old MySQL):")
    hits = store.search(
        user_id="alice",
        query="Which analytics database does the user prefer?",
        top_k=3,
        category="preference",
        now=now,
        touch=False,
    )
    for rank, hit in enumerate(hits, start=1):
        age_d = (now - hit.record.last_accessed_at) / day
        print(
            f"   #{rank} score={hit.combined_score:.3f} "
            f"(sim={hit.similarity:.3f}, recency={hit.recency_score:.3f}, age={age_d:.0f}d) "
            f"-> {hit.record.content}"
        )
    assert all(h.record.user_id == "alice" for h in hits), "Cross-user leak detected!"
    assert "PostgreSQL" in hits[0].record.content, "Expected fresher PostgreSQL preference to rank #1!"

    # 4. Close and reopen SQLite file to verify persistence across restarts.
    store.close()
    reopened = SQLiteVectorMemory(db_path=db_path, embedder=embedder)
    print("\n4) Reopened SQLite store from disk -> querying Alice's project memories:")
    proj_hits = reopened.search(
        user_id="alice",
        query="What Python version and region is checkout-api using?",
        top_k=2,
        category="project",
        metadata_filter={"service": "checkout-api"},
        now=now,
    )
    for hit in proj_hits:
        print(f"   [{hit.record.id}] score={hit.combined_score:.3f} -> {hit.record.content}")
    assert len(proj_hits) == 1 and "eu-west-1" in proj_hits[0].record.content

    # 5. Verify Bob's memories are isolated from Alice and searchable for Bob.
    bob_hits = reopened.search(user_id="bob", query="analytics database", top_k=2, now=now)
    print(f"\n5) Tenant isolation check -> Bob query returned {len(bob_hits)} hit(s) belonging solely to Bob:")
    for hit in bob_hits:
        print(f"   [{hit.record.user_id}] {hit.record.content}")
    assert all(h.record.user_id == "bob" for h in bob_hits)

    # 6. Prune stale low-importance memories.
    before = reopened.count("alice")
    pruned = reopened.prune_stale(max_age_days=90.0, max_importance=0.4, now=now)
    after = reopened.count("alice")
    print(f"\n6) Pruned stale low-importance memories: removed={pruned} (alice count: {before} -> {after})")
    reopened.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 8: SQLite vector memory store")
    parser.add_argument("--demo", action="store_true", help="Use offline local hashing embeddings")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional path to SQLite DB file (defaults to a temporary file cleaned up on exit).",
    )
    args = parser.parse_args()

    if _should_use_demo(args.demo):
        embedder: EmbeddingProvider = LocalHashEmbeddingProvider(dim=256)
    else:
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        embedder = OpenAIEmbeddingProvider(api_key=api_key, base_url=base_url, model=emb_model)

    if args.db:
        run_demo(embedder, Path(args.db))
    else:
        with tempfile.TemporaryDirectory(prefix="day08_memory_") as tmp_dir:
            run_demo(embedder, Path(tmp_dir) / "agent_memory.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
