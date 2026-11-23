[<< Day 4](../04_Day_Tool_and_Function_Calling_Basics/README.md) | [Home](../README.md) | [Day 6 >>](../06_Day_Error_Handling_and_Self_Correction/README.md)

# Day 5: Building Your First Agent Loop

Everything from Days 1 through 4—system instructions, context budgets, schema validation, and tool dispatch—comes together today in one clean control structure: the agent loop. Today we build a working multi-step `model -> tool call -> tool execution -> observation -> next step` loop with explicit termination conditions, hard `max_steps` limits, duplicate-call loop detection, and structured step traces.

## Table of Contents

- [Core Concepts & Why They Matter](#core-concepts--why-they-matter)
  - [Why Loops Need Hard Boundaries](#why-loops-need-hard-boundaries)
  - [Three Essential Termination Conditions](#three-essential-termination-conditions)
  - [Cost Scales Quadratically Across Steps](#cost-scales-quadratically-across-steps)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Code Walkthrough](#code-walkthrough)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [Exercises - Day 5](#-exercises---day-5)

---

## Core Concepts & Why They Matter

### Why Loops Need Hard Boundaries

Unlike a single-turn LLM call, an agent loop hands next-step control to a probabilistic model. Most of the time, the model gathers the facts it needs and stops. Occasionally, a tool returns ambiguous data, a prompt is underspecified, or a smaller model gets stuck repeating the exact same query ten times in a row.

An unbounded `while True:` loop around a paid LLM API is a classic production incident waiting to happen—it burns money, ties up worker threads, and leaves users staring at a spinner. Every production agent loop needs explicit stop conditions enforced in Python.

### Three Essential Termination Conditions

| Stop Reason | When It Triggers | What the Caller Should Do |
| :--- | :--- | :--- |
| **`completed`** | The model returns an assistant message with no `tool_calls`. | Return `final_answer` to the user or downstream workflow. |
| **`max_steps_reached`** | The loop finishes `max_steps` iterations without a final answer. | Stop immediately, log the partial trace, and either return a graceful fallback or ask a human to step in. |
| **`duplicate_loop_detected`** | The model emits the exact same tool name + arguments multiple times in a row. | Abort early instead of wasting remaining steps repeating a broken call. |

### Cost Scales Quadratically Across Steps

Each iteration of an agent loop resends the **entire conversation so far** (system prompt + user goal + all previous assistant `tool_calls` + all previous `tool` observations). In a 4-step run where each step adds ~120 tokens of observations, prompt tokens look like `180 + 295 + 420 + 510 = 1,405` tokens—not `510`. Keeping tool outputs compact and capping `max_steps` directly controls both latency and token bills.

---

## How It Works Under the Hood

Here is the exact control flow inside `AgentLoop.run()`:

```text
  [User Goal + System Prompt]
               |
               v
     +---> [for step in 1..max_steps]
     |         |
     |         v
     |     [Call LLM with messages + tool schemas]
     |         |
     |         +--- No tool_calls? ----------> Return AgentResult(stop_reason="completed")
     |         |
     |         +--- Identical call repeated? -> Return AgentResult(stop_reason="duplicate_loop_detected")
     |         |
     |         v
     |     [Append assistant tool_calls message]
     |         |
     |         v
     |     [For each tool_call: validate -> execute -> append role:"tool" observation]
     |         |
     |         v
     |     [Record StepRecord(step, latency_ms, tokens, tool_traces)]
     |         |
     +---------+
               |
         (Loop exhausted)
               v
     Return AgentResult(stop_reason="max_steps_reached")
```

Notice how clean the state model is: **`messages` is the working state of the agent**, and `steps: list[StepRecord]` is your structured observability log.

---

## Code Walkthrough

Open [`agent_loop.py`](./agent_loop.py). It implements an SRE incident triage agent equipped with four local tools:
- `get_service_metrics(service_name)` — returns error rate, p95 latency, top error, and owning team.
- `calculate_error_budget_burn(error_rate_pct, slo_target_pct)` — computes SLO burn multiplier and severity.
- `search_runbooks(query)` — finds relevant remediation runbooks.
- `get_oncall_engineer(team)` — looks up the primary on-call contact and Slack channel.

Key classes in the file:
- `ChatModel` protocol with two implementations: `OpenAICompatibleModel` (live HTTP calls via `urllib.request`) and `DemoIncidentModel` (deterministic 4-step mock for offline runs).
- `AgentLoop`: runs the bounded loop, checks all three termination conditions, tracks cumulative prompt/completion tokens, and records a `StepRecord` per iteration.
- `AgentResult`: returns `final_answer`, `stop_reason`, full `steps` trace, and total token counts so callers can make explicit decisions on incomplete runs.

Run in offline demo mode:

```bash
python3 05_Day_Building_Your_First_Agent_Loop/agent_loop.py --demo
```

Sample output:

```text
=== Run 1: Multi-Step Incident Investigation [DEMO, max_steps=6] ===
Step 1 | finish=tool_calls | tokens=180in/24out | 0.0 ms
  -> get_service_metrics({"service_name": "payments-api"})
  <- [OK] {"ok": true, "result": {"found": true, "service": "payments-api", "owner_team": "payments-core", "error_rat...
Step 2 | finish=tool_calls | tokens=295in/48out | 0.0 ms
  -> calculate_error_budget_burn({"error_rate_pct": 2.4, "slo_target_pct": 99.9})
  <- [OK] {"ok": true, "result": {"error_rate_pct": 2.4, "slo_target_pct": 99.9, "allowed_error_pct": 0.1, "burn_rate...
  -> search_runbooks({"query": "payments 504 timeout"})
  <- [OK] {"ok": true, "result": {"query": "payments 504 timeout", "matches": [{"id": "RB-402", "title": "Payments Up...
Step 3 | finish=tool_calls | tokens=420in/22out | 0.0 ms
  -> get_oncall_engineer({"team": "payments-core"})
  <- [OK] {"ok": true, "result": {"found": true, "team": "payments-core", "primary": "priya.nair@example.com", "slack...
Step 4 | finish=stop       | tokens=510in/88out | 0.0 ms

Stop reason : completed
Total tokens: 1405 prompt + 182 completion

Final Answer:
Incident Brief for payments-api:
- Live Metrics: 2.4% errors, p95 1480 ms (UpstreamGatewayTimeout (HTTP 504)).
- SLO Impact: 24.0x burn rate against 99.9% SLO (page_immediately).
- Runbook RB-402: Enable circuit breaker fallback to secondary processor and page payments-core on-call.
- Page On-Call: priya.nair@example.com (#inc-payments).

=== Run 2: Safety Guard Demonstration [DEMO, max_steps=2] ===
Step 1 | finish=tool_calls | tokens=180in/24out | 0.0 ms
  -> get_service_metrics({"service_name": "payments-api"})
  <- [OK] {"ok": true, "result": {"found": true, "service": "payments-api", "owner_team": "payments-core", "error_rat...
Step 2 | finish=tool_calls | tokens=295in/48out | 0.0 ms
  -> calculate_error_budget_burn({"error_rate_pct": 2.4, "slo_target_pct": 99.9})
  <- [OK] {"ok": true, "result": {"error_rate_pct": 2.4, "slo_target_pct": 99.9, "allowed_error_pct": 0.1, "burn_rate...
  -> search_runbooks({"query": "payments 504 timeout"})
  <- [OK] {"ok": true, "result": {"query": "payments 504 timeout", "matches": [{"id": "RB-402", "title": "Payments Up...

Stop reason : max_steps_reached
Total tokens: 475 prompt + 72 completion

Final Answer: <none — stopped early by safety guard>
```

To run against a live model (OpenAI or local Ollama):

```bash
python3 05_Day_Building_Your_First_Agent_Loop/agent_loop.py \
  "Check search-api metrics and tell me if we need to page anyone." \
  --max-steps 5
```

Notice how each step depends on information learned in the prior step: the agent does not know `owner_team` is `"payments-core"` or that `error_rate_pct` is `2.4%` until `get_service_metrics` returns in Step 1. That dynamic dependency chain is the exact case where an agent loop earns its keep over a rigid static script.

---

## Common Pitfalls & Bugs Developers Run Into

1. **Using `while True:` with no `max_steps` or token cap.** Even if your happy path takes 3 steps, always set a hard ceiling (for example, `max_steps=6` or `8`) and test what happens when the limit triggers.
2. **Returning empty strings silently when `max_steps` hits.** Distinguish `stop_reason == "completed"` from `stop_reason == "max_steps_reached"` in your return type (`AgentResult`) so callers never present an incomplete or empty string as a valid answer.
3. **Comparing raw argument strings naively for duplicate-loop detection.** Two JSON strings `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` represent the exact same call. Normalize parsed JSON with `sort_keys=True` when building loop-detection signatures.
4. **Hiding intermediate steps behind `print` statements only.** Printing to stdout works for local debugging, but storing structured `StepRecord` objects lets you write unit tests on tool order, assert token budgets, and export traces to JSONL logs.
5. **Letting the agent exit on `finish_reason == "length"` without noticing.** If the final answer gets cut off mid-sentence because of `max_tokens`, `tool_calls` will be empty and a naive loop treats it as `completed`. Always record and inspect `finish_reason`.

---

## 💻 Exercises - Day 5

### Exercise: Level 1

1. Run `agent_loop.py --demo --max-steps 3` and verify which step runs last and what `stop_reason` is returned.
2. Write a small `StuckMockModel` class in `agent_loop.py` (or a test script) that always returns the exact same `get_service_metrics({"service_name": "payments-api"})` tool call. Run `AgentLoop` with it and confirm `stop_reason == "duplicate_loop_detected"` triggers before `max_steps`.
3. Add a third service `"auth-api"` to `SERVICE_METRICS`, a matching runbook to `RUNBOOKS`, and an on-call entry to `ONCALL_ROSTER`, then run `agent_loop.py` against a live model asking it to investigate `auth-api`.

### Exercise: Level 2

1. Add a **total token budget** (`max_total_tokens: int = 4000`) to `AgentLoop` so the loop stops with `stop_reason="token_budget_exceeded"` if `total_prompt_tokens + total_completion_tokens` crosses the cap.
2. Improve duplicate-call detection in `AgentLoop.run()` so JSON arguments are parsed and re-serialized with `json.dumps(parsed, sort_keys=True)`—ensuring `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` count as identical calls.
3. Add a **graceful final synthesis pass** when `max_steps` is reached: make one final model call with `tools=[]` (disabling further tool calls) asking the model to summarize what it has discovered so far and clearly list what remains unknown.

### Exercise: Level 3

1. Add a **wall-clock timeout** (`max_duration_s: float = 30.0`) to `AgentLoop.run()` that checks elapsed time before each step and returns `stop_reason="timeout"` cleanly if the overall run exceeds the SLA.
2. Add a `--trace-file` CLI flag that serializes the entire `AgentResult` (including every `StepRecord` and `ToolCallTrace`) to a JSON file using `dataclasses.asdict()`, so runs can be replayed or inspected after the fact.
3. Combine Day 2's context budget check with Day 5's loop: before each `self.model.complete()` call, estimate total tokens in `messages` and warn (or stop cleanly) before sending a prompt that would exceed a configured `max_input_tokens` ceiling.

---

[<< Day 4](../04_Day_Tool_and_Function_Calling_Basics/README.md) | [Home](../README.md) | [Day 6 >>](../06_Day_Error_Handling_and_Self_Correction/README.md)
