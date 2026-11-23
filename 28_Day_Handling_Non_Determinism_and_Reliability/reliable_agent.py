"""Day 28: Handling Non-Determinism & Reliability in AI Agents.

Demonstrates four reliability mechanisms required when agents call real tools:
  1. SQLite-backed `IdempotencyStore` + end-to-end idempotency keys passed to
     downstream side-effecting APIs (so network timeouts after remote commit
     and duplicate LLM retries never double-charge or double-create tickets).
  2. Exponential backoff with full jitter (`retry_with_backoff`) distinguishing
     transient failures (timeouts, 429, 503) from permanent errors (400, 403).
  3. Loop and oscillation detection (`LoopDetector`) catching both repeated
     identical calls (A -> A -> A) and 2-step oscillations (A -> B -> A -> B).
  4. Graceful degradation & fallbacks (`FallbackExecutor`) when primary tools
     or providers suffer an outage.

Run from the repository root:
    python3 28_Day_Handling_Non_Determinism_and_Reliability/reliable_agent.py --demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Error Taxonomy
# ---------------------------------------------------------------------------


class TransientToolError(RuntimeError):
    """Temporary failure (timeout, 429 rate limit, 502/503) safe to retry."""


class PermanentToolError(RuntimeError):
    """Non-retryable failure (400 validation, 401/403 auth, 404 missing)."""


class IdempotencyConflictError(PermanentToolError):
    """Raised when an existing idempotency key is reused with different arguments."""


class LoopDetectedError(RuntimeError):
    """Raised when the agent is stuck repeating identical or oscillating calls."""


# ---------------------------------------------------------------------------
# Canonical Hashing & SQLite Idempotency Store
# ---------------------------------------------------------------------------


def canonical_args_hash(arguments: dict[str, Any]) -> str:
    """Stable SHA-256 hash of tool arguments independent of dict key order."""
    serialized = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def derive_idempotency_key(run_id: str, step_intent: str, tool_name: str) -> str:
    """Create a deterministic key scoped to one workflow run and logical step."""
    raw = f"{run_id}:{step_intent}:{tool_name}".encode("utf-8")
    return f"idem_{hashlib.sha256(raw).hexdigest()[:16]}"


class SQLiteIdempotencyStore:
    """Persists completed tool side effects keyed by idempotency key."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_records (
                idempotency_key TEXT PRIMARY KEY,
                tool_name       TEXT NOT NULL,
                args_hash       TEXT NOT NULL,
                status          TEXT NOT NULL,
                result_json     TEXT,
                updated_at      TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def lookup(
        self, idempotency_key: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return cached result if completed; raise on argument mismatch."""
        args_hash = canonical_args_hash(arguments)
        row = self._conn.execute(
            """
            SELECT tool_name, args_hash, status, result_json
            FROM idempotency_records
            WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()

        if row is None:
            return None

        stored_tool, stored_hash, status, result_json = row
        if stored_tool != tool_name or stored_hash != args_hash:
            raise IdempotencyConflictError(
                f"Idempotency key '{idempotency_key}' was already used for "
                f"different tool/arguments (stored tool={stored_tool}, hash={stored_hash[:10]})."
            )

        if status == "completed" and result_json is not None:
            return json.loads(result_json)
        return None

    def save_completed(
        self,
        idempotency_key: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO idempotency_records (
                idempotency_key, tool_name, args_hash, status, result_json, updated_at
            ) VALUES (?, ?, ?, 'completed', ?, ?)
            ON CONFLICT(idempotency_key) DO UPDATE SET
                status = 'completed',
                result_json = excluded.result_json,
                updated_at = excluded.updated_at
            """,
            (
                idempotency_key,
                tool_name,
                canonical_args_hash(arguments),
                json.dumps(result, sort_keys=True),
                now,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Exponential Backoff + Full Jitter Retry Wrapper
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    max_attempts: int = 4
    base_delay_s: float = 0.05
    max_delay_s: float = 0.50
    sleep_fn: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=lambda: random.Random(7))


def retry_with_backoff(
    fn: Callable[[], T],
    config: RetryConfig,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    """Retry transient errors with capped exponential backoff and full jitter."""
    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            return fn()
        except TransientToolError as exc:
            last_error = exc
            if attempt == config.max_attempts:
                break
            cap = min(config.max_delay_s, config.base_delay_s * (2 ** (attempt - 1)))
            delay = config.rng.uniform(0.0, cap)
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            config.sleep_fn(delay)
        except PermanentToolError:
            # Never retry validation, permission, or idempotency conflict errors.
            raise

    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Loop & Oscillation Detector
# ---------------------------------------------------------------------------


@dataclass
class LoopCheckOutcome:
    status: str  # "ok", "steer", "halt"
    reason: str


class LoopDetector:
    """Detects identical repeated tool calls and A->B->A->B oscillations."""

    def __init__(self, max_identical: int = 2, max_oscillations: int = 2) -> None:
        self.max_identical = max_identical
        self.max_oscillations = max_oscillations
        self._history: list[tuple[str, str]] = []
        self._steered_once = False

    def record_and_check(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> LoopCheckOutcome:
        sig = (tool_name, canonical_args_hash(arguments))
        self._history.append(sig)

        # 1. Check consecutive identical calls (A -> A -> A).
        if len(self._history) >= self.max_identical:
            tail = self._history[-self.max_identical :]
            if all(item == sig for item in tail):
                if not self._steered_once:
                    self._steered_once = True
                    return LoopCheckOutcome(
                        "steer",
                        f"Identical tool call '{tool_name}' repeated {self.max_identical} times; "
                        "inject steering prompt to try a different strategy.",
                    )
                return LoopCheckOutcome(
                    "halt",
                    f"Agent kept repeating '{tool_name}' even after steering; halting loop.",
                )

        # 2. Check 2-step oscillation (A -> B -> A -> B).
        cycle_len = 2 * self.max_oscillations
        if len(self._history) >= cycle_len:
            window = self._history[-cycle_len:]
            a, b = window[0], window[1]
            if a != b and all(window[i] == (a if i % 2 == 0 else b) for i in range(cycle_len)):
                return LoopCheckOutcome(
                    "halt",
                    f"Oscillation loop detected between '{a[0]}' and '{b[0]}'; halting run.",
                )

        return LoopCheckOutcome("ok", "progressing normally")


# ---------------------------------------------------------------------------
# Simulated Flaky Side-Effecting API (Proves Exactly-Once Execution)
# ---------------------------------------------------------------------------


class FlakyRefundService:
    """Simulates a remote payment gateway that supports idempotency keys.

    Failure script:
      - Call 1: Commits refund in ledger, then times out BEFORE sending HTTP 200.
      - Call 2: Fails with HTTP 503 before touching the ledger.
      - Call 3+: Returns the committed refund record using the idempotency key.
    """

    def __init__(self) -> None:
        self.http_attempts = 0
        self.ledger: list[dict[str, Any]] = []
        self._by_idempotency_key: dict[str, dict[str, Any]] = {}

    def issue_refund(
        self, order_id: str, amount_usd: float, idempotency_key: str
    ) -> dict[str, Any]:
        self.http_attempts += 1

        # Attempt 1: Remote commit succeeds, then response drops (classic split-brain).
        if self.http_attempts == 1:
            record = {
                "refund_id": f"RF-{len(self.ledger) + 1001}",
                "order_id": order_id,
                "amount_usd": amount_usd,
                "idempotency_key": idempotency_key,
                "status": "settled",
            }
            self.ledger.append(record)
            self._by_idempotency_key[idempotency_key] = record
            raise TransientToolError(
                "ReadTimeout after remote server committed refund (client got no response)"
            )

        # Attempt 2: Upstream gateway temporarily overloaded.
        if self.http_attempts == 2:
            raise TransientToolError("HTTP 503 Service Unavailable (gateway overloaded)")

        # Attempt 3+: Downstream checks idempotency key and returns existing record.
        if idempotency_key in self._by_idempotency_key:
            return dict(self._by_idempotency_key[idempotency_key], replayed_by_provider=True)

        record = {
            "refund_id": f"RF-{len(self.ledger) + 1001}",
            "order_id": order_id,
            "amount_usd": amount_usd,
            "idempotency_key": idempotency_key,
            "status": "settled",
        }
        self.ledger.append(record)
        self._by_idempotency_key[idempotency_key] = record
        return dict(record, replayed_by_provider=False)


# ---------------------------------------------------------------------------
# Reliable Tool Executor & Graceful Fallback Wrapper
# ---------------------------------------------------------------------------


class ReliableToolRunner:
    """Combines SQLite idempotency caching + exponential backoff retries."""

    def __init__(
        self,
        store: SQLiteIdempotencyStore,
        retry_config: RetryConfig,
    ) -> None:
        self.store = store
        self.retry_config = retry_config

    def execute_idempotent(
        self,
        idempotency_key: str,
        tool_name: str,
        arguments: dict[str, Any],
        tool_fn: Callable[..., dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        """Run a side-effecting tool safely; returns (result, from_local_cache)."""
        cached = self.store.lookup(idempotency_key, tool_name, arguments)
        if cached is not None:
            return cached, True

        def _invoke() -> dict[str, Any]:
            return tool_fn(**arguments, idempotency_key=idempotency_key)

        result = retry_with_backoff(
            _invoke,
            self.retry_config,
            on_retry=lambda attempt, delay, err: print(
                f"  [retry {attempt}] transient failure: {err} -> backing off {delay * 1000:.0f}ms"
            ),
        )
        self.store.save_completed(idempotency_key, tool_name, arguments, result)
        return result, False


def execute_with_fallback(
    primary: Callable[[], dict[str, Any]],
    secondary: Callable[[], dict[str, Any]],
    retry_config: RetryConfig,
) -> dict[str, Any]:
    """Try primary with retries, fall back to secondary, then degrade gracefully."""
    try:
        return retry_with_backoff(primary, retry_config)
    except Exception as primary_err:
        print(f"  [fallback] primary failed ({primary_err}); switching to secondary...")
        try:
            res = secondary()
            return {**res, "degraded_mode": True, "source": "secondary_cache"}
        except Exception as secondary_err:
            return {
                "degraded_mode": True,
                "source": "graceful_escalation",
                "message": (
                    "Live shipping systems are temporarily unavailable. "
                    "A support ticket has been queued for manual follow-up."
                ),
                "errors": [str(primary_err), str(secondary_err)],
            }


# ---------------------------------------------------------------------------
# End-to-End Reliability Demo
# ---------------------------------------------------------------------------


def run_demo() -> None:
    store = SQLiteIdempotencyStore(":memory:")
    retry_cfg = RetryConfig(max_attempts=4, base_delay_s=0.02, max_delay_s=0.10)
    runner = ReliableToolRunner(store, retry_cfg)
    refund_api = FlakyRefundService()

    # Scenario 1: Flaky side-effecting tool commits on attempt 1 then times out.
    print("=== Scenario 1: Flaky Side-Effecting Tool + Idempotency + Backoff ===")
    run_id = "run_2026_11_23_001"
    idem_key = derive_idempotency_key(run_id, "refund_damaged_order", "issue_refund")
    args = {"order_id": "ORD-8821", "amount_usd": 49.00}

    result1, cached1 = runner.execute_idempotent(
        idem_key, "issue_refund", args, refund_api.issue_refund
    )
    print(f"  First logical call result (local_cache={cached1}): {result1}")
    print(
        f"  HTTP attempts made: {refund_api.http_attempts} | "
        f"Actual refunds in billing ledger: {len(refund_api.ledger)}"
    )
    assert len(refund_api.ledger) == 1, "Side effect must execute exactly once!"

    # Scenario 2: Non-deterministic LLM decides to invoke the exact same tool again,
    # and then tries to reuse the key with a mutated amount.
    print("\n=== Scenario 2: Duplicate LLM Retry & Argument Conflict Detection ===")
    result2, cached2 = runner.execute_idempotent(
        idem_key, "issue_refund", args, refund_api.issue_refund
    )
    print(
        f"  Duplicate call result (local_cache={cached2}): {result2['refund_id']} "
        f"(HTTP attempts still {refund_api.http_attempts}, ledger count still {len(refund_api.ledger)})"
    )
    assert cached2 is True and len(refund_api.ledger) == 1

    try:
        runner.execute_idempotent(
            idem_key,
            "issue_refund",
            {"order_id": "ORD-8821", "amount_usd": 99.00},
            refund_api.issue_refund,
        )
    except IdempotencyConflictError as exc:
        print(f"  Conflict guard blocked mutated retry: {exc}")

    # Scenario 3: Loop and oscillation detection.
    print("\n=== Scenario 3: Repeated-Call & Oscillation Loop Detection ===")
    detector_repeat = LoopDetector(max_identical=2)
    for step in range(1, 5):
        outcome = detector_repeat.record_and_check("search_docs", {"query": "refund policy"})
        print(f"  [repeat test] step {step}: status={outcome.status:<5} | {outcome.reason}")
        if outcome.status == "halt":
            break

    detector_osc = LoopDetector(max_identical=3, max_oscillations=2)
    osc_sequence = [
        ("read_file", {"path": "config.yaml"}),
        ("validate_config", {"env": "prod"}),
        ("read_file", {"path": "config.yaml"}),
        ("validate_config", {"env": "prod"}),
    ]
    for step, (tool, call_args) in enumerate(osc_sequence, start=1):
        outcome = detector_osc.record_and_check(tool, call_args)
        print(f"  [oscillation] step {step}: {tool} -> status={outcome.status:<5} | {outcome.reason}")
        if outcome.status == "halt":
            break

    # Scenario 4: Graceful degradation when primary tool has an outage.
    print("\n=== Scenario 4: Graceful Fallback When Primary Tool Is Down ===")
    fast_retry = RetryConfig(max_attempts=2, base_delay_s=0.01, max_delay_s=0.02)

    def broken_live_carrier() -> dict[str, Any]:
        raise TransientToolError("Carrier API 504 Gateway Timeout")

    def cached_tracking_snapshot() -> dict[str, Any]:
        return {
            "order_id": "ORD-8821",
            "last_known_status": "In Transit (snapshot from 15m ago)",
        }

    fallback_out = execute_with_fallback(
        broken_live_carrier, cached_tracking_snapshot, fast_retry
    )
    print(f"  Fallback response: {fallback_out}")
    store.close()
    print("\nAll reliability invariants verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 28 Reliability & Non-Determinism Demo")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the offline reliability demonstration (default behavior).",
    )
    parser.parse_args()
    run_demo()


if __name__ == "__main__":
    main()
