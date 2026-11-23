[<< Day 27](../27_Day_Cost_Latency_and_Model_Routing/README.md) | [Home](../README.md) | [Day 29 >>](../29_Day_Deploying_Agents_as_APIs_and_Streaming_UX/README.md)

# Day 28: Handling Non-Determinism & Reliability

Today we make side-effecting agents safe under real network failures, retries, and stochastic model behavior. Even at `temperature=0`, LLMs remain non-deterministic across tool failures and context shifts, while real networks drop responses *after* a remote write succeeds.

## Table of Contents

- [Why Non-Determinism Hurts Side-Effecting Workflows](#why-non-determinism-hurts-side-effecting-workflows)
- [How Reliability Controls Work Under the Hood](#how-reliability-controls-work-under-the-hood)
  - [Two-Layer Idempotency Keys](#two-layer-idempotency-keys)
  - [Retries with Exponential Backoff & Full Jitter](#retries-with-exponential-backoff--full-jitter)
  - [Loop & Oscillation Detection](#loop--oscillation-detection)
  - [Graceful Degradation & Fallbacks](#graceful-degradation--fallbacks)
- [Code Walkthrough: `reliable_agent.py`](#code-walkthrough-reliable_agentpy)
- [Common Pitfalls & Bugs](#common-pitfalls--bugs)
- [💻 Exercises - Day 28](#-exercises---day-28)

---

## Why Non-Determinism Hurts Side-Effecting Workflows

Read-only agents fail politely—you rerun the query and move on. Agents that charge cards, issue refunds, send customer emails, or file Jira tickets fail expensively when something flickers:

1. **Timeout after remote commit:** Your tool sends `POST /refunds`. The payment provider commits the $49 refund, and then the TCP connection resets before the HTTP 200 reaches your process. Without an idempotency key, retrying issues a second $49 refund.
2. **Duplicate LLM tool calls:** The model receives an ambiguous observation or loses track of a long context and invokes `create_ticket` or `send_email` a second time for the exact same task.
3. **Argument drift on retries:** On retry, the model subtly changes an argument (e.g., `amount_usd: 49.00` becomes `49.99` or a different recipient).
4. **Stuck loops and oscillations:** A tool keeps returning `0 results` or a validation error, so the agent repeats `search_docs("refund policy")` ten times or alternates `read_file -> validate_config -> read_file -> validate_config` until it burns its step and dollar budget.

You cannot prompt-engineer your way out of distributed systems failures. You solve them in deterministic Python code around the model.

---

## How Reliability Controls Work Under the Hood

Every mutating tool invocation should pass through four outer guards:

```text
  LLM emits tool_call(tool_name, args)
        |
        v
  +------------------------+   "steer" -> inject hint to model
  | 1. LoopDetector        |   "halt"  -> abort run cleanly
  +-----------+------------+
              | "ok"
              v
  +------------------------+   HIT (same args) -> return cached result
  | 2. Local SQLite        |   CONFLICT (diff args) -> raise error
  |    IdempotencyStore    |
  +-----------+------------+
              | MISS
              v
  +------------------------+   Transient error (timeout/429/503) -> backoff + jitter
  | 3. retry_with_backoff  |   Permanent error (400/403/404)     -> fail immediately
  +-----------+------------+
              |
              v  (passes idempotency_key header/arg downstream!)
  +------------------------+   Exhausted retries?
  | 4. Remote Tool API     | --------------------> Fallback / Degraded Response
  +------------------------+
```

### Two-Layer Idempotency Keys

An idempotency key is a deterministic identifier scoped to one logical action (for example `sha256(f"{run_id}:{step_intent}:{tool_name}")`). Real protection requires **both** layers:

- **Agent-side SQLite store (`SQLiteIdempotencyStore`):** Records `(idempotency_key, tool_name, canonical_args_hash, status, result_json)`. If the model asks to run the same side-effecting call again, the runner immediately returns the cached result without touching the network. If the key is reused with different arguments, it raises `IdempotencyConflictError` rather than silently executing a mutated call.
- **Downstream API idempotency key:** Pass the exact same `idempotency_key` to the payment gateway, ticketing API, or database upsert (`ON CONFLICT DO NOTHING`). That closes the split-brain window where attempt 1 succeeded remotely and timed out on the way back.

### Retries with Exponential Backoff & Full Jitter

Classify tool errors before retrying:

- **Retry (`TransientToolError`):** Network timeouts, connection resets, HTTP `429 Too Many Requests`, and `502`/`503`/`504`.
- **Fail fast (`PermanentToolError`):** HTTP `400 Bad Request`, `401`/`403` auth errors, `404 Not Found`, schema validation errors, and `IdempotencyConflictError`. Retrying a 400 error four times only wastes latency.

When retrying, add **full jitter** (`random.uniform(0, min(max_delay, base * 2^attempt))`). Without jitter, hundreds of concurrent agent workers that hit a rate limit at the same second all retry in lockstep and knock the upstream service down again (the thundering herd problem).

### Loop & Oscillation Detection

A plain `max_steps=15` limit is a blunt safety net—it still lets a broken agent waste 14 LLM calls doing the exact same thing. Track a rolling list of `(tool_name, canonical_args_hash)` tuples:

1. **Identical repeats (`A -> A -> A`):** On the first repeat, inject a steering message (`"You already called search_docs with these exact arguments and got the same result; try a different query or explain what is missing"`). If the agent ignores the hint and repeats again, halt immediately.
2. **2-step oscillations (`A -> B -> A -> B`):** Detect repeating alternating cycles and halt cleanly so the orchestrator can return a partial summary or escalate.

### Graceful Degradation & Fallbacks

When an upstream dependency stays down after retries are exhausted, crashing with an unhandled `TimeoutError` leaves users stranded. Build an explicit fallback ladder:

1. **Primary tool/model** (live carrier API, primary LLM provider).
2. **Secondary tool/model** (stale read-only cache snapshot with `degraded_mode: true`, or a backup LLM provider).
3. **Graceful escalation** (return a structured message telling the user what succeeded so far and queueing a manual follow-up ticket).

---

## Code Walkthrough: `reliable_agent.py`

Open [`reliable_agent.py`](./reliable_agent.py) to inspect the implementation:

- `SQLiteIdempotencyStore` uses Python's built-in `sqlite3` module to persist completed tool calls keyed by `idempotency_key` and checks `canonical_args_hash` to block argument drift.
- `retry_with_backoff()` implements capped exponential backoff with full jitter and splits `TransientToolError` from `PermanentToolError`.
- `LoopDetector` catches both identical repeats (with a one-time `"steer"` warning before `"halt"`) and `A -> B -> A -> B` oscillations.
- `FlakyRefundService` simulates the nastiest real-world failure mode: attempt 1 commits a refund in the billing ledger and throws a timeout before replying, attempt 2 returns HTTP 503, and attempt 3 replays the committed refund via the idempotency key.
- `execute_with_fallback()` demonstrates falling back to a cached snapshot when a live tool stays unreachable.

Run the script directly from the repository root:

```bash
python3 28_Day_Handling_Non_Determinism_and_Reliability/reliable_agent.py --demo
```

Sample output:

```text
=== Scenario 1: Flaky Side-Effecting Tool + Idempotency + Backoff ===
  [retry 1] transient failure: ReadTimeout after remote server committed refund (client got no response) -> backing off 6ms
  [retry 2] transient failure: HTTP 503 Service Unavailable (gateway overloaded) -> backing off 6ms
  First logical call result (local_cache=False): {'refund_id': 'RF-1001', 'order_id': 'ORD-8821', 'amount_usd': 49.0, 'idempotency_key': 'idem_80d4207cda529aeb', 'status': 'settled', 'replayed_by_provider': True}
  HTTP attempts made: 3 | Actual refunds in billing ledger: 1

=== Scenario 2: Duplicate LLM Retry & Argument Conflict Detection ===
  Duplicate call result (local_cache=True): RF-1001 (HTTP attempts still 3, ledger count still 1)
  Conflict guard blocked mutated retry: Idempotency key 'idem_80d4207cda529aeb' was already used for different tool/arguments (stored tool=issue_refund, hash=197221e08e).

=== Scenario 3: Repeated-Call & Oscillation Loop Detection ===
  [repeat test] step 1: status=ok    | progressing normally
  [repeat test] step 2: status=steer | Identical tool call 'search_docs' repeated 2 times; inject steering prompt to try a different strategy.
  [repeat test] step 3: status=halt  | Agent kept repeating 'search_docs' even after steering; halting loop.
  [oscillation] step 1: read_file -> status=ok    | progressing normally
  [oscillation] step 2: validate_config -> status=ok    | progressing normally
  [oscillation] step 3: read_file -> status=ok    | progressing normally
  [oscillation] step 4: validate_config -> status=halt  | Oscillation loop detected between 'read_file' and 'validate_config'; halting run.

=== Scenario 4: Graceful Fallback When Primary Tool Is Down ===
  [fallback] primary failed (Carrier API 504 Gateway Timeout); switching to secondary...
  Fallback response: {'order_id': 'ORD-8821', 'last_known_status': 'In Transit (snapshot from 15m ago)', 'degraded_mode': True, 'source': 'secondary_cache'}

All reliability invariants verified.
```

---

## Common Pitfalls & Bugs

1. **Generating a fresh `uuid4()` inside the retry loop:** If you generate a new idempotency key on every retry attempt, the downstream API treats each attempt as a brand-new payment or ticket. Generate the key *once* per logical step before entering the retry loop.
2. **Storing idempotency records only in RAM:** If your container restarts mid-workflow or a worker hands off to another pod, an in-memory dictionary vanishes. Back `SQLiteIdempotencyStore` with persistent storage (SQLite on disk, Postgres, or Redis with persistence).
3. **Retrying non-idempotent side effects blindly:** If a legacy internal API does not support idempotency keys and cannot be made idempotent via a unique database constraint, **do not auto-retry timeouts**. Surface the uncertain state to a reconciliation check or human operator.
4. **Ignoring `Retry-After` headers on 429s:** Many LLM and SaaS APIs send an exact `Retry-After: 4` header when rate-limited. Respecting it avoids burning attempts before the window resets.
5. **Over-aggressive loop detection on legitimate pagination:** Calling `list_tickets(page=1)` followed by `list_tickets(page=2)` is *not* a loop because the canonical argument hashes differ. Always include arguments—never just `tool_name`—in your loop signature.

---

## 💻 Exercises - Day 28

### Exercise: Level 1

1. Change `SQLiteIdempotencyStore(":memory:")` to a persistent file path (e.g., `/tmp/agent_idem.db`), run `reliable_agent.py` twice in a row with the same `run_id`, and confirm the second process serves the refund straight from SQLite with zero HTTP attempts.
2. Extend `RetryConfig` and `retry_with_backoff` to honor an optional `retry_after_s: float | None` attribute on `TransientToolError` so 429 responses wait at least the server-requested duration.
3. Add a test in `run_demo()` verifying that raising `PermanentToolError("HTTP 400 Invalid order_id")` fails on attempt 1 with zero retries.

### Exercise: Level 2

1. Add a 3-step cycle detector (`A -> B -> C -> A -> B -> C`) to `LoopDetector` and write a small test case proving it halts cleanly without flagging normal pagination calls (`page=1`, `page=2`, `page=3`).
2. Add an `in_progress` reservation state to `SQLiteIdempotencyStore`: insert a row with `status='in_progress'` *before* calling the remote tool, mark it `'completed'` on success, and add a `reconcile_in_progress()` flow for crashes that happen mid-call.
3. Implement a `CircuitBreaker` class for flaky upstream tools: after 5 consecutive `TransientToolError` failures within 30 seconds, open the circuit for 60 seconds and immediately route to the fallback without wasting retry latency.

### Exercise: Level 3

1. Build a multi-provider LLM fallback client (`PrimaryProvider -> BackupProvider`) using `urllib.request` that wraps `/chat/completions` with `retry_with_backoff`, automatically fails over from your primary endpoint to a local Ollama endpoint (or secondary model) on 5xx/429 exhaustion, and tags the returned trace with `fallback_used=True`.
2. Combine Day 17 (SQLite checkpointing) with Day 28 (`SQLiteIdempotencyStore`) in a 3-step workflow (`lookup_order -> issue_refund -> send_receipt_email`). Simulate a process crash (`raise SystemExit`) right after `issue_refund` completes, resume from the checkpoint, and prove `issue_refund` is not re-executed while `send_receipt_email` finishes cleanly.
3. Write a `pytest` suite fuzzing `FlakyRefundService` with 50 randomized failure scripts (timeouts before commit, timeouts after commit, 503 bursts, duplicate LLM calls) and assert that the billing ledger count equals `1` in 100% of runs.

[<< Day 27](../27_Day_Cost_Latency_and_Model_Routing/README.md) | [Home](../README.md) | [Day 29 >>](../29_Day_Deploying_Agents_as_APIs_and_Streaming_UX/README.md)
