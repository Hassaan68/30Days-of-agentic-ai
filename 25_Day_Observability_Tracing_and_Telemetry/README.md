[<< Day 24](../24_Day_Audit_Logs_and_Tamper_Evident_Receipts/README.md) | [Home](../README.md) | [Day 26 >>](../26_Day_Evaluation_and_Benchmarking_Agents/README.md)

# Day 25: Observability, Tracing, and Telemetry

A single user message to an agent often fans out into three or four LLM calls, multiple tool invocations, retries after transient timeouts, and guardrail checks. Today we build a lightweight, OpenTelemetry-aligned context-manager tracer that tracks nested spans, token counts, cost estimates, and error paths, exports structured JSONL traces, and renders a terminal waterfall timeline.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
  - [Audit Logs vs. Observability Traces](#audit-logs-vs-observability-traces)
  - [Traces, Nested Spans, and `contextvars`](#traces-nested-spans-and-contextvars)
  - [Agent-Specific Telemetry: Tokens, Cost, Latency, and Error Paths](#agent-specific-telemetry-tokens-cost-latency-and-error-paths)
  - [OpenTelemetry Compatibility and JSONL Export](#opentelemetry-compatibility-and-jsonl-export)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Code Walkthrough: `agent_tracer.py`](#code-walkthrough-agent_tracerpy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 25](#-exercises---day-25)

---

## Core Concept & Why It Matters

When a user reports *"the assistant took 18 seconds and gave a weird answer,"* reading flat console logs rarely tells you why. Was the bottleneck a slow database query, a model call returning 1,200 unnecessary tokens, or a tool failing twice before succeeding on attempt three?

Structured tracing gives you a tree of timed operations for every single agent run so you can spot latency spikes, cost regressions, and hidden retries in seconds.

### Audit Logs vs. Observability Traces

Day 24 and Day 25 complement each other, and both share the same `trace_id`:

- **Audit logs (Day 24)** answer *accountability and compliance* questions: who authorized a refund, which policy blocked a path traversal, and can we prove no entry was altered?
- **Observability traces (Day 25)** answer *engineering and reliability* questions: where did the time go, how many tokens did step 2 burn, which tool threw `TimeoutError`, and what is our p95 cost per conversation?

### Traces, Nested Spans, and `contextvars`

Distributed tracing uses two simple IDs defined by the W3C Trace Context standard:

- **`trace_id`** (128-bit / 32 hex chars): shared by every operation triggered by one top-level user request.
- **`span_id`** (64-bit / 16 hex chars) + **`parent_span_id`**: uniquely identifies one unit of work and links it to its caller (`agent.run` -> `step.1` -> `llm.chat_completion` / `tool.fetch_order`).

Passing `parent_span` explicitly through every helper function's signature gets messy fast. Python's standard-library `contextvars.ContextVar` solves this cleanly: entering `with tracer.span(...)` sets the active span ID for the current execution context (safely isolated across threads and `asyncio` tasks), and exiting restores the previous span ID—even when an exception is raised.

### Agent-Specific Telemetry: Tokens, Cost, Latency, and Error Paths

Standard HTTP tracing only measures request duration and status codes. Agent runs need domain-specific attributes:

- **Token usage & cost:** Capture `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and computed `gen_ai.usage.cost_usd` on every LLM span. Notice how input tokens grow on each turn of a multi-step agent loop (410 -> 620 -> 890 in our demo) as tool outputs accumulate in the conversation history.
- **Latency breakdown:** Separate cumulative LLM time from tool time so you know whether to optimize prompts/models or fix a slow API.
- **Error paths and retries:** Never swallow exceptions inside a tracer, and never hide retried failures. Recording `attempt=1` as `status="ERROR"` (`TimeoutError`) next to `attempt=2` (`OK`) makes flaky dependencies immediately visible in the waterfall.

### OpenTelemetry Compatibility and JSONL Export

In `agent_tracer.py`, span fields follow OpenTelemetry GenAI semantic conventions (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, W3C hex IDs, nanosecond timestamps, `events`, and `status`). You can inspect the exported JSONL locally with `jq`, load it into DuckDB/pandas for cost analysis, or swap the exporter to OpenTelemetry's OTLP exporter to ship spans to Jaeger, Grafana Tempo, Arize Phoenix, or Langfuse without changing your instrumentation code.

---

## How It Works Under the Hood

```text
  with tracer.span("agent.run")                    # parent=None, span_id=S1
    ├─ with tracer.span("step.1_fetch_context")    # parent=S1,   span_id=S2
    │    ├─ llm.chat_completion (plan)             # parent=S2,   478 tokens
    │    ├─ tool.fetch_order (attempt=1)  [ERROR]  # parent=S2,   TimeoutError
    │    └─ tool.fetch_order (attempt=2)  [OK]     # parent=S2,   HTTP 200
    ├─ with tracer.span("step.2_verify_fulfillment")
    │    ├─ llm.chat_completion (tool_select)      # 715 tokens
    │    ├─ tool.check_inventory                   # OK
    │    └─ tool.calculate_shipping                # OK
    └─ with tracer.span("step.3_synthesize_reply")
         ├─ llm.chat_completion (final_answer)     # 1030 tokens
         └─ guardrail.output_scan                  # 0 redactions

  Timing rule:
  - Wall-clock `start_time_ns` (time.time_ns) orders spans across systems.
  - Monotonic `time.perf_counter()` computes `duration_ms` so NTP clock
    adjustments can never produce negative span durations.
```

---

## Code Walkthrough: `agent_tracer.py`

Open [`agent_tracer.py`](./agent_tracer.py). It has zero external dependencies and runs directly from the repository root.

Key components:
- `MODEL_PRICING_PER_1M_TOKENS` & `estimate_cost_usd()`: calculates dollar estimates per LLM span (including `$0.00` for local Ollama models like `llama3.2`).
- `SpanRecord` & `SpanEvent`: structured span model with `set_attribute()`, `add_event()`, and `record_llm_usage()`.
- `AgentTracer`: manages `trace_id`, `ContextVar`-backed nesting via `with tracer.span(...)`, exception capture (always re-raising), rollup metrics in `summarize()`, JSONL export in `export_jsonl()`, and terminal visualization in `render_waterfall()`.
- `run_traced_agent_demo()`: runs a realistic 3-step agent workflow including a simulated transient `TimeoutError` and retry on `tool.fetch_order`, prints the waterfall and summary metrics, and verifies JSONL export round-tripping.

### Running the Script

Run from the repository root:

```bash
python3 25_Day_Observability_Tracing_and_Telemetry/agent_tracer.py
```

To keep the exported JSONL trace file for inspection:

```bash
python3 25_Day_Observability_Tracing_and_Telemetry/agent_tracer.py --export /tmp/day25_trace.jsonl
```

Sample output:

```text
=== Day 25: Agent Observability & Trace Waterfall ===

Trace ID: 4b7c9e201f3a8d14c09a5e6b12f38a7c  (total wall time: 306.8 ms)
Span                                 | Timeline                               | Dur (ms) | Details
------------------------------------------------------------------------------------------------------------------------
agent.run                            | ██████████████████████████████████████ |    306.8 |
├─ step.1_fetch_context              | ████████████████...................... |    128.4 |
│  ├─ llm.chat_completion (plan)     | ██████................................ |     46.1 | 478 tok ($0.00010)
│  ├─ tool.fetch_order               | ......!!!!!!!......................... |     56.2 | [ERR: TimeoutError] attempt=1
│  └─ tool.fetch_order               | .............███...................... |     26.0 | attempt=2
├─ step.2_verify_fulfillment         | ................███████████████....... |    117.6 |
│  ├─ llm.chat_completion (tool_sel… | ................██████................ |     51.0 | 715 tok ($0.00015)
│  ├─ tool.check_inventory           | ......................████............ |     30.9 |
│  └─ tool.calculate_shipping        | ..........................█████....... |     35.7 |
└─ step.3_synthesize_reply           | ...............................███████ |     69.4 |
   ├─ llm.chat_completion (final_an… | ...............................██████. |     60.9 | 1030 tok ($0.00022)
   └─ guardrail.output_scan          | .....................................█ |      8.4 | redactions=0

=== Trace Summary Metrics ===
  Trace ID:         4b7c9e201f3a8d14c09a5e6b12f38a7c
  Total wall time:  306.8 ms
  Spans recorded:   12 (LLM: 3, Tools: 4, Errors: 1)
  LLM latency sum:  158.0 ms
  Tool latency sum: 148.8 ms
  Tokens used:      2223 (in=1920, out=303)
  Est. model cost:  $0.000470
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Storing unredacted prompts and tool outputs inside span attributes:** Traces are often viewable by the whole engineering team and shipped to third-party observability SaaS tools. Never put raw customer PII, passwords, or full prompt bodies in span attributes without running your redactor (Day 21) or gating prompt capture behind a scrubbed debug flag.
2. **Double-counting duration across nested spans:** In `TraceSummary`, summing `duration_ms` across *all* spans double-counts parent spans (`agent.run` includes `step.1`, which includes `llm.chat_completion`). Always compute total wall time from `max(end_time_ns) - min(start_time_ns)`, and sum leaf span kinds (`llm`, `tool`) separately—noting that parallel tool calls can sum to *more* than wall time.
3. **Using wall-clock `time.time()` for span durations:** Wall clocks can jump backward on NTP adjustments, causing negative durations and broken waterfall bars. Measure duration with `time.perf_counter()` and derive `end_time_ns` from `start_time_ns + elapsed_ns`.
4. **High-cardinality span names:** Naming a span `tool.fetch_order.ORD-7741` creates millions of distinct span names and explodes your metrics backend. Keep span names low-cardinality (`tool.fetch_order`) and put dynamic IDs in `attributes={"order.id": "ORD-7741"}`.
5. **Swallowing exceptions inside a tracing context manager:** A tracer must observe failures without changing control flow. Always re-raise in `except BaseException:` after recording `status="ERROR"` and resetting the `ContextVar` token in `finally:`.

---

## 💻 Exercises - Day 25

### Exercise: Level 1

1. Run `agent_tracer.py --export /tmp/day25_trace.jsonl` and inspect one `llm` span and one failed `tool` span in the JSONL file.
2. Add a `gpt-4o` step or change the demo model from `gpt-4o-mini` to `gpt-4o` and compare `estimated_cost_usd` in the summary output.
3. Add a `record_retry_event()` call using `span.add_event("retry_scheduled", {"next_attempt": 2})` on `step.1_fetch_context` and print span events below the waterfall when any span has events.
4. Add a `span_percentile_report()` helper that groups spans by `name` and prints count, mean duration, and error rate.

### Exercise: Level 2

1. Instrument **parallel tool calls** using `asyncio.gather()` inside `step.2_verify_fulfillment` so `tool.check_inventory` and `tool.calculate_shipping` run concurrently. Verify that `contextvars` attaches both spans to the same `step.2_verify_fulfillment` parent and that their bars overlap in `render_waterfall()`!
2. Add **TTFT (Time-to-First-Token)** tracking on LLM spans (`gen_ai.latency.ttft_ms`) for streaming responses, distinguishing network/prefill latency from token-generation duration.
3. Implement **cost and latency budget alerts**: pass `max_cost_usd` and `max_wall_ms` to `AgentTracer` so `summarize()` flags any trace that crosses your SLA or spend threshold.
4. Write a JSONL trace reader CLI (`--render <trace.jsonl>`) that loads a previously exported JSONL file from disk and renders the ASCII waterfall and summary without re-running the agent.

### Exercise: Level 3

1. Build a **multi-trace JSONL analyzer** that reads 50+ exported traces from a directory and outputs a terminal dashboard showing p50/p95/p99 wall latency, p95 token usage, total daily spend by model, and top failing tools ranked by error rate.
2. Connect `AgentTracer` to an actual OpenAI or local Ollama `/chat/completions` call: extract real `usage.prompt_tokens` and `usage.completion_tokens` from the API JSON response, populate `span.record_llm_usage(...)`, and render the real waterfall of a multi-turn tool-using agent run.
3. Write an **OTLP JSON exporter** function `to_otlp_json(spans)` that maps `SpanRecord` objects into the standard OpenTelemetry `resourceSpans -> scopeSpans -> spans` JSON schema so traces can be POSTed directly to a local OpenTelemetry Collector or Jaeger OTLP endpoint (`/v1/traces`).

---

[<< Day 24](../24_Day_Audit_Logs_and_Tamper_Evident_Receipts/README.md) | [Home](../README.md) | [Day 26 >>](../26_Day_Evaluation_and_Benchmarking_Agents/README.md)
