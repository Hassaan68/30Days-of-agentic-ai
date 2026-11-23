[<< Day 9](../09_Day_Routing_and_Classification_Workflows/README.md) | [Home](../README.md) | [Day 11 >>](../11_Day_ReAct_and_Chain_of_Thought_Reasoning/README.md)

# Day 10: Parallel Execution and Map-Reduce

Calling tools or LLMs one after another in a plain `for` loop turns a 6-source research or log-analysis task into a 10-second wait—and one hung network call can stall or crash the entire run. Today we use `asyncio.gather`, `asyncio.Semaphore`, per-task timeouts, and `return_exceptions=True` to run bounded concurrent Map steps, handle partial failures honestly, and aggregate the surviving results in a clean Reduce step.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `parallel_map_reduce.py`](#walkthrough-parallel_map_reducepy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 10](#-exercises---day-10)

---

## Core Concept & Why It Matters

Agentic workloads spend most of their wall-clock time waiting on network I/O—LLM API calls, database queries, web fetches, or log searches. Whenever you have independent sub-tasks (evaluating 10 candidate documents, fetching metrics from 6 microservices, or summarizing 12 chapters of a long report), running them sequentially leaves performance on the table.

At the same time, naive parallelism (`asyncio.gather(*all_100_tasks)`) breaks quickly in production:

1. **Unbounded fan-out triggers HTTP 429 rate limits.** Firing 50 concurrent LLM requests at once hits provider requests-per-minute (RPM) and tokens-per-minute (TPM) ceilings, causing cascading retries that make the batch slower than running with 5 workers. An `asyncio.Semaphore(N)` caps how many tasks run simultaneously.
2. **One slow endpoint stalls the whole batch.** Without `asyncio.wait_for(..., timeout=...)`, a single hung socket holds up `asyncio.gather` indefinitely.
3. **Default `gather` behavior discards completed work on error.** By default (`return_exceptions=False`), if task 5 of 6 raises `RuntimeError`, `asyncio.gather` raises immediately and throws away the 4 results you already paid tokens for. Setting `return_exceptions=True` lets you collect every completed result, inspect which tasks failed, verify you met a minimum coverage threshold (for example, `>= 50%`), and surface exact coverage gaps in the final output.

### When to Use Map-Reduce in Agents

- **Large inputs that exceed a single context window** (or degrade in quality when stuffed into one massive prompt): split into chunks, Map each chunk to a compact structured JSON summary, and Reduce those summaries into a final answer.
- **Multi-source triage and research**: query independent tools or services in parallel, extract normalized findings from each source, and synthesize one consolidated brief.

---

## How It Works Under the Hood

```text
  6 Service Log Chunks (chunk_1 .. chunk_6)
                     │
                     ▼
     ┌───────────────────────────────────────────────────┐
     │  MAP PHASE: asyncio.gather(..., return_exceptions=True)
     │  Bounded by asyncio.Semaphore(3) + wait_for(0.45s)│
     ├───────────────┬───────────────────┬───────────────┤
     │ Slot 1        │ Slot 2            │ Slot 3        │
     │ chunk_1 -> OK │ chunk_2 -> OK     │ chunk_3 -> OK │
     │ chunk_4 -> TO │ chunk_5 -> 503    │ chunk_6 -> OK │
     └───────┬───────┴─────────┬─────────┴───────┬───────┘
             └─────────────────┼─────────────────┘
                               ▼
          Partition results: 4 succeeded, 2 failed
          Check min_coverage (66.7% >= 50% -> proceed)
                               │
                               ▼
     ┌───────────────────────────────────────────────────┐
     │  REDUCE PHASE: Single synthesis call              │
     │  Input: 4 structured MapOutputs + 2 failed IDs    │
     │  Output: Ranked brief + explicit "Coverage Gaps"  │
     └───────────────────────────────────────────────────┘
```

### Handling Synchronous HTTP Calls Inside `asyncio`

Python's built-in `urllib.request.urlopen` is synchronous (blocking). Calling it directly inside an `async def` coroutine blocks the entire event loop, eliminating concurrency. Wrapping blocking calls with `await asyncio.to_thread(self._sync_chat, ...)` offloads each HTTP call onto Python's thread pool so multiple requests run in parallel without requiring extra third-party HTTP libraries. When you scale to hundreds of concurrent sockets, switch to an async HTTP client such as `httpx.AsyncClient`.

### Subtlety with Semaphore + Timeout Ordering

Notice where `asyncio.wait_for` sits inside `_run_one_bounded`:

```python
async def _run_one_bounded(chunk, backend, semaphore, timeout_s):
    async with semaphore:
        return await asyncio.wait_for(backend.map_chunk(chunk), timeout=timeout_s)
```

Placing `wait_for` *inside* `async with semaphore` starts the timeout clock only after the task acquires a concurrency slot. If you put `wait_for` *outside* `async with semaphore`, later tasks sitting in the queue waiting for a slot can time out before they even start executing.

---

## Walkthrough: `parallel_map_reduce.py`

Open [`parallel_map_reduce.py`](./parallel_map_reduce.py). It compares sequential vs. bounded parallel Map timing across 6 microservice incident log chunks, then injects a timeout and an HTTP 503 failure to verify partial-failure handling and Reduce synthesis.

Key components in the file:

- **`DocumentChunk`, `MapOutput`, `MapReduceReport`**: Typed dataclasses keeping Map outputs compact and machine-readable before the Reduce phase.
- **`DemoAsyncBackend`**: Simulates realistic per-chunk latency (`0.20s–0.26s`), a hung call (`timeout`), and an upstream failure (`upstream_503`) so you can benchmark speedups and test resilience offline.
- **`OpenAIAsyncBackend`**: Runs live OpenAI-compatible Map and Reduce calls concurrently via `asyncio.to_thread`.
- **`run_parallel_map_reduce(...)`**: Enforces `asyncio.Semaphore(max_concurrency)` and `asyncio.wait_for(..., timeout=per_task_timeout_s)`, gathers results with `return_exceptions=True`, enforces `min_coverage`, and explicitly passes failed chunk metadata into Reduce so blind spots are never hidden.

### Run the Offline Demo

From the repository root:

```bash
python 10_Day_Parallel_Execution_and_Map_Reduce/parallel_map_reduce.py --demo
```

Sample output:

```text
=== Day 10: Parallel Execution & Map-Reduce (offline async demo backend) ===

1) Timing benchmark across 6 healthy chunks (max_concurrency=3):
   Sequential Map time:                1.407s
   Parallel Map time (Semaphore=3):    0.483s (2.91x faster)
   Reduce synthesis time:              0.082s

2) Map-Reduce with injected timeout (chunk_4) and HTTP 503 failure (chunk_5):
   Succeeded: 4/6 (66.7% coverage) | Failed: 2 | Total time: 0.735s

   Per-chunk Map statuses:
     - chunk_1 [checkout-api    ]: OK (sev1, 0.25s)
     - chunk_2 [payments-worker ]: OK (sev2, 0.22s)
     - chunk_3 [auth-service    ]: OK (ok, 0.20s)
     - chunk_4 [search-indexer  ]: FAILED (Timed out after 0.45s)
     - chunk_5 [notification-svc]: FAILED (RuntimeError: HTTP 503 Service Unavailable while analyzing notification-svc)
     - chunk_6 [billing-cron    ]: OK (ok, 0.23s)

3) Final Reduced Synthesis:
------------------------------------------------------------------------
Executive Brief (4/6 services analyzed, 4,501 total errors observed):

Priority Findings:
  - [SEV1] checkout-api (4182 errors): ConnectionPoolExhausted on db-eu-west-1 after DEPLOY-772. -> Action: Roll back DEPLOY-772 and increase pool wait timeout alerts.
  - [SEV2] payments-worker (319 errors): Stripe webhook retries spiked due to downstream checkout 502s. -> Action: Replay dead-letter queue once checkout-api stabilizes.
  - [OK] auth-service (0 errors): Token issuance and p95 latency (28ms) remained healthy. -> Action: No action required.
  - [OK] billing-cron (0 errors): Nightly invoice reconciliation completed cleanly. -> Action: No action required.

Coverage Gaps (excluded from synthesis — manual check required):
  - search-indexer (chunk_4): Timed out after 0.45s
  - notification-svc (chunk_5): RuntimeError: HTTP 503 Service Unavailable while analyzing notification-svc
------------------------------------------------------------------------
```

### Tune Concurrency or Run Against Live Models

Try changing concurrency to 1, 2, or 6 in demo mode to see the wall-clock curve:

```bash
python 10_Day_Parallel_Execution_and_Map_Reduce/parallel_map_reduce.py --demo --concurrency 6
```

Or run against a live OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
python 10_Day_Parallel_Execution_and_Map_Reduce/parallel_map_reduce.py --concurrency 3
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Calling blocking I/O directly inside `async def`.** Using `requests.post(...)` or `time.sleep(...)` inside an `async` coroutine blocks the event loop so all 6 tasks run sequentially even inside `asyncio.gather`. Always use async libraries (`httpx`, `aiohttp`, `asyncio.sleep`) or offload blocking functions with `asyncio.to_thread(...)`.
2. **Putting `asyncio.wait_for` outside the semaphore.** If you wrap the semaphore acquisition inside `wait_for`, tasks waiting for a free worker slot burn through their timeout while queued. Acquire the semaphore first, then start the per-task timeout.
3. **Hiding failed chunks during Reduce.** If 2 out of 6 service logs failed to load and your Reduce prompt only receives the 4 successful summaries, the final brief may confidently claim `"All systems except checkout-api are healthy"`. Always pass the failed chunk list into Reduce so the synthesis explicitly discloses coverage gaps.
4. **Overflowing the Reduce context window.** Map-Reduce over 200 chunks can produce 200 summaries that still exceed one Reduce prompt. For large fan-outs, use **hierarchical (tree) reduction**: reduce batches of 10–15 summaries into intermediate summaries, then reduce those into the final brief.
5. **Forgetting to cancel/bound retries across concurrent tasks.** If 10 concurrent Map workers all hit HTTP 429 and retry simultaneously on the same fixed delay (`sleep(1.0)`), they create a thundering herd that triggers another 429 wave. Add randomized jitter to backoff delays and share a concurrency semaphore or token-bucket rate limiter across workers.

---

## 💻 Exercises - Day 10

### Exercise: Level 1

1. **Measure concurrency scaling.** Run `parallel_map_reduce.py --demo --concurrency 1`, `--concurrency 2`, `--concurrency 3`, and `--concurrency 6`. Record the Map duration for each and explain why `--concurrency 6` stops improving beyond the slowest single chunk (`0.26s`).
2. **Test the `min_coverage` abort guard.** In `async_main`, raise `min_coverage` from `0.50` to `0.80` on the faulty-chunk run (which has `66.7%` coverage) and verify that Reduce cleanly aborts with an explicit explanation instead of producing a partial brief.
3. **Inspect timeout vs. exception handling.** Change `--timeout 0.21` in `--demo` mode and verify which chunks finish in time (`auth-service` at `0.20s`) and which get cancelled cleanly by `asyncio.wait_for`.

### Exercise: Level 2

1. **Add bounded async retries with jitter for transient 503 errors.** Wrap `backend.map_chunk(chunk)` in a retry helper inside `_run_one_bounded` that retries up to 2 times with exponential backoff + random jitter (`random.uniform(0, 0.05)`) only on transient `RuntimeError`s (not on timeouts), and test it with a chunk that fails once and succeeds on attempt 2.
2. **Implement a real document splitter.** Add a `chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[DocumentChunk]` helper that splits a long markdown or log file on paragraph boundaries and feeds those chunks into `run_parallel_map_reduce`.
3. **Add streaming progress output with `asyncio.as_completed`.** Build an alternative Map runner using `asyncio.as_completed` that prints each chunk's completion status the moment it finishes rather than waiting for the entire `gather` to complete, while preserving original chunk ordering in the final `MapReduceReport`.

### Exercise: Level 3

1. **Build hierarchical (tree) Reduce.** Implement `hierarchical_reduce(succeeded, failed, backend, group_size=3)` so that when `len(succeeded) > group_size`, summaries are reduced in bounded parallel batches of `group_size` recursively until a single final synthesis remains.
2. **Add an async token-bucket rate limiter.** Alongside `asyncio.Semaphore` (which limits in-flight concurrency), build an async `RateLimiter(requests_per_second=4.0)` so bursty Map batches respect strict provider RPM limits even when individual requests return quickly.
3. **Compare sequential vs. parallel live LLM latency and token spend.** Run a 10-chunk dataset against a live OpenAI-compatible endpoint (or local Ollama) at `--concurrency 1`, `3`, and `5`, logging p50/p95 per-chunk latency, total wall-clock time, HTTP 429 count, and total prompt/completion tokens.

---

[<< Day 9](../09_Day_Routing_and_Classification_Workflows/README.md) | [Home](../README.md) | [Day 11 >>](../11_Day_ReAct_and_Chain_of_Thought_Reasoning/README.md)
