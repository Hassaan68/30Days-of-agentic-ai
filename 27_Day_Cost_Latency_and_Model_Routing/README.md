[<< Day 26](../26_Day_Evaluation_and_Benchmarking_Agents/README.md) | [Home](../README.md) | [Day 28 >>](../28_Day_Handling_Non_Determinism_and_Reliability/README.md)

# Day 27: Cost, Latency & Model Routing Optimization

Today we look at why agentic workloads get expensive and slow much faster than plain chatbots, and how to bring both down by 60%–85% using model tiering, prompt caching, response caching, context trimming, and per-run budget circuit breakers.

## Table of Contents

- [Why Agent Spend Compounds Faster Than Expected](#why-agent-spend-compounds-faster-than-expected)
- [How Cost & Latency Optimization Works Under the Hood](#how-cost--latency-optimization-works-under-the-hood)
  - [Token Cost Math & Prompt Caching](#token-cost-math--prompt-caching)
  - [Complexity-Based Model Routing](#complexity-based-model-routing)
  - [Context & Tool-Output Trimming](#context--tool-output-trimming)
  - [Exact & Semantic Response Caching](#exact--semantic-response-caching)
  - [Hard Per-Run Budget Guards](#hard-per-run-budget-guards)
- [Code Walkthrough: `cost_router_optimizer.py`](#code-walkthrough-cost_router_optimizerpy)
- [Common Pitfalls & Bugs](#common-pitfalls--bugs)
- [💻 Exercises - Day 27](#-exercises---day-27)

---

## Why Agent Spend Compounds Faster Than Expected

A single chat completion has a predictable bill. A 10-step agent loop does not:

1. **Quadratic input growth:** On step 1, you send 1,500 tokens (system prompt + tool schemas + user query). On step 2, you resend those 1,500 tokens *plus* step 1's tool call and tool output. If one tool returns a 4,000-token log dump on step 2, you pay for those 4,000 tokens again on steps 3, 4, 5, ..., 10.
2. **Output token asymmetry:** Output tokens typically cost 3x to 5x more per million than input tokens—and every generated output token adds serial decoding latency (`tokens / tokens_per_sec`).
3. **Over-provisioning one frontier model:** Routing simple intent classification, JSON extraction, and repetitive FAQ answers to your most expensive flagship model burns money and adds seconds of Time to First Token (TTFT) where a small model gets the exact same accuracy in 200ms.

---

## How Cost & Latency Optimization Works Under the Hood

A cost-aware agent puts four lightweight controls in front of the LLM call:

```text
  Incoming Request
        |
        v
  +------------------+   HIT (2ms, $0)
  | Exact TTL Cache  | ------------------> Return Cached Response
  +--------+---------+
           | MISS
           v
  +------------------+
  | Context Trimmer  |  (bound oversized tool blobs before prompting)
  +--------+---------+
           |
           v
  +------------------+
  | Complexity Router| ---> fast (extract/classify/FAQ)
  +--------+---------+ ---> balanced (standard tool steps / SQL / debug)
           |           ---> frontier (postmortems / architecture / hard planning)
           v
  +------------------+
  | BudgetGuard      |  (pre-flight check: tokens + USD <= hard cap)
  +--------+---------+
           | OK
           v
  Provider API (with static system prompt + tools first for prefix cache hit)
```

### Token Cost Math & Prompt Caching

Model billing splits into three separate meters:

```text
cost_usd = (uncached_input / 1e6) * input_rate
         + (cached_input   / 1e6) * cached_input_rate
         + (output_tokens  / 1e6) * output_rate
```

Modern inference providers offer **prompt prefix caching** (typically 50%–90% cheaper on cached input tokens). The catch: caching is prefix-based. If you inject a dynamic timestamp like `Current time: 2026-11-23T14:05:11Z` on line 1 of your system prompt, you bust the prefix cache on every single request. Keep static instructions and tool JSON schemas at the very top of the message array, and put per-request state (timestamps, user IDs, retrieved memory) at the end.

### Complexity-Based Model Routing

Not every step in an agent run needs the same reasoning horsepower:

- **Fast tier (`gpt-4o-mini` / small local models):** Intent routing, entity extraction, formatting JSON, guardrails, summarizing short snippets, and FAQ lookups. Costs roughly 10x–15x less than frontier models and starts streaming in under 250ms.
- **Balanced tier (`gpt-4.1-mini` / mid-size models):** Everyday multi-step tool calling, SQL generation, code debugging, and policy comparisons.
- **Frontier tier (`gpt-4o` / flagship reasoning models):** Multi-document root-cause analysis, high-stakes planning, security reviews, and complex architectural trade-offs.

You can even mix tiers *within a single agent run*: let a frontier model produce the initial 4-step plan, let a fast model run routine extraction steps, and reserve the frontier model for final synthesis.

### Context & Tool-Output Trimming

One careless `query_logs` or `fetch_webpage` tool can drop 15,000 tokens of boilerplate into your conversation history. Trim tool outputs *before* appending them to message history: keep the head and tail (where errors and summaries usually live), save the full raw payload to disk or object storage under an artifact ID, and give the agent a `read_artifact_slice(id, offset)` tool if it genuinely needs more lines.

### Exact & Semantic Response Caching

- **Exact TTL caching** normalizes whitespace and casing, hashes `(task_type, normalized_prompt)`, and caches deterministic or read-only lookups (like policy FAQs or intent classifications) for a short TTL (e.g., 5–15 minutes).
- **Semantic caching** embeds the incoming query and returns a cached reply when cosine similarity exceeds a strict threshold (often `>= 0.96`).

Be disciplined about **what not to cache**: never cache user-specific account lookups, live order statuses, or side-effecting tool actions under a shared cache key.

### Hard Per-Run Budget Guards

Retry loops and stuck tool calls can quietly spend tens of dollars in a few minutes. Put a `BudgetGuard` directly in your LLM client wrapper that checks projected tokens and projected USD spend *before* sending the HTTP request. If the next call would cross the limit, raise `BudgetExceededError`, halt the loop cleanly, and return a partial result or escalate to a human.

---

## Code Walkthrough: `cost_router_optimizer.py`

Open [`cost_router_optimizer.py`](./cost_router_optimizer.py) to read the implementation:

- `MODEL_CATALOG` defines pricing (`input_per_1m`, `cached_input_per_1m`, `output_per_1m`) and latency (`ttft_ms`, `tokens_per_sec`) for `fast`, `balanced`, and `frontier` tiers.
- `ExactTTLCache` normalizes whitespace/casing, enforces TTL expiration, and evicts oldest entries at capacity.
- `trim_tool_output()` caps noisy log blobs while keeping the beginning and the tail (where the actual `ERROR db_pool exhausted` line sits).
- `ComplexityRouter` scores task type, input size, and domain markers to pick a tier.
- `BudgetGuard` performs a pre-flight check against `max_total_tokens` and `max_cost_usd`.
- `build_benchmark_workload()` runs 18 mixed requests under both a naive frontier-only strategy and the optimized strategy, then triggers `demo_budget_circuit_breaker()`.

Run the benchmark offline:

```bash
python3 27_Day_Cost_Latency_and_Model_Routing/cost_router_optimizer.py --demo
```

Sample output:

```text
=== Benchmark: Naive Single Frontier vs. Routed + Cached + Trimmed ===
Metric                                    Naive          Optimized          Delta
--------------------------------------------------------------------------------
Total requests                               18                 18               
Exact cache hits                              0                  4        22% hit
Input tokens (total)                     34,594             21,442               
  - Prompt-cached input                       0             16,093               
Output tokens (total)                     2,220              1,975               
Avg latency / request                   3117 ms            2123 ms         -31.9%
Total USD spend                        $0.10869           $0.02696         -75.2%
--------------------------------------------------------------------------------
Total token reduction: 36.4% | Optimized tier split (non-cached calls): {'fast': 8, 'balanced': 2, 'frontier': 4}

=== BudgetGuard Circuit Breaker Demo ===
  step 01: OK  | cumulative=$0.00925 | tokens=2,650
  step 02: HALTED by BudgetGuard -> Projected spend ($0.02050) exceeds limit ($0.01500)
```

Notice that spend dropped **75.2%** and average latency dropped **31.9%** while the 4 hardest tasks still received the frontier model.

---

## Common Pitfalls & Bugs

1. **Busting prompt caches with dynamic prefixes:** Putting `request_id`, current timestamp, or user metadata at the start of the system prompt drops prefix cache hit rates to 0%. Always place static instructions and tool definitions first.
2. **Caching across tenant boundaries:** Sharing a global cache key across different users or workspaces leaks private data between customers. Always include `tenant_id` (or mark `cacheable=False`) for anything user-specific.
3. **Aggressive routing without evals:** Moving a classification or tool-calling step to a cheaper model saves money only if accuracy holds up. Always run your Day 26 eval suite before and after changing routing thresholds.
4. **Checking budget *after* the call instead of before:** Post-call checks still let a single huge request blow past your cap. Estimate input + `max_tokens` output before sending the request, and set an explicit `max_tokens` cap on every API call.
5. **Truncating the tail of stack traces:** Naive `text[:1000]` slicing drops the bottom of Python tracebacks and log dumps, which is usually where the root error message lives. Keep both head and tail (or search for `ERROR`/`Exception` lines).

---

## 💻 Exercises - Day 27

### Exercise: Level 1

1. Add a fourth model tier `"local_ollama"` to `MODEL_CATALOG` with `$0.00` token cost and realistic local hardware latency (`ttft_ms=120`, `tokens_per_sec=45`), and route `"classify"` tasks to it.
2. Run the script with `--max-budget 0.01` and update `run_optimized_strategy` to catch `BudgetExceededError` gracefully, reporting how many requests completed before the guard tripped.
3. Add a `cost_per_tier` breakdown dictionary to `RunMetrics` and print what percentage of total spend came from `fast`, `balanced`, and `frontier` calls.

### Exercise: Level 2

1. Implement a `ConversationContextTrimmer` class that keeps the system prompt and the last `N` turns intact while dropping or summarizing older middle turns whenever total estimated tokens exceed `max_context_tokens`.
2. Add a confidence-based escalation pattern (often called **cascading routing**): send ambiguous medium queries to `fast` first with an instruction to output `{"answer": ..., "confidence": 0.0-1.0}`, and automatically escalate to `frontier` if `confidence < 0.75`. Measure the cost/latency trade-off on the benchmark workload.
3. Extend `ExactTTLCache` to include `tenant_id` and `prompt_version` in `_make_key()` so prompt updates invalidate stale cache entries automatically without cross-tenant leakage.

### Exercise: Level 3

1. Build a lightweight **semantic cache** using bag-of-words TF-IDF or real embedding vectors (from Day 8) with a configurable cosine-similarity threshold (e.g., `0.95`), and test it on paraphrased FAQ queries like *"What's the return period?"* vs. *"What is our standard 30-day return window?"*.
2. Integrate `BudgetGuard` with a live OpenAI-compatible client that reads exact `usage.prompt_tokens`, `usage.prompt_tokens_details.cached_tokens`, and `usage.completion_tokens` returned by the API rather than estimates, reconciling estimated pre-flight reservations with actual billed usage.
3. Combine Day 26 (`agent_eval_harness.py`) and Day 27 (`ComplexityRouter`): write a benchmark runner that plots or prints a Pareto table of `(accuracy %, avg latency ms, total USD)` across three routing thresholds so you can pick the best quality-per-dollar operating point using real data.

[<< Day 26](../26_Day_Evaluation_and_Benchmarking_Agents/README.md) | [Home](../README.md) | [Day 28 >>](../28_Day_Handling_Non_Determinism_and_Reliability/README.md)
