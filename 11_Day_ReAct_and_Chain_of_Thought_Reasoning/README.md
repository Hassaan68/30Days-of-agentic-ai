[<< Day 10](../10_Day_Parallel_Execution_and_Map_Reduce/README.md) | [Home](../README.md) | [Day 12 >>](../12_Day_Planning_and_Task_Decomposition/README.md)

# Day 11: ReAct and Chain-of-Thought Reasoning

Today we build a ReAct (Reason + Act) agent that writes down a short explicit thought before every tool call and every final response. Adding a structured reasoning step before acting reduces wrong tool calls, missing arguments, and rushed guesses—while keeping the model's internal scratchpad strictly separate from what your user sees.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [When NOT to Force Verbose Reasoning](#when-not-to-force-verbose-reasoning)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `react_agent.py`](#walkthrough-react_agentpy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 11](#-exercises---day-11)

---

## Core Concept & Why It Matters

In Day 5, we built a basic tool-calling loop: send the prompt to the model, run whatever tool it picks, feed the result back, and repeat. On straightforward single-tool requests, that works fine.

On multi-step requests with business rules, plain tool calling often trips over itself:

- **Premature tool calls:** The model calls `calculate` with a guessed 10% fee before checking the actual return policy.
- **Wrong argument formatting:** It passes `"electronics return"` instead of `"electronics"` because it did not pause to inspect the schema.
- **Lost intermediate state:** After three tool outputs land in the message history, the model forgets a constraint from step 1 (for example, that the item was opened or that shipping is non-refundable).

**ReAct** (introduced by Yao et al., 2022) fixes this by interleaving **Chain-of-Thought (CoT) reasoning** with **tool actions**:

1. **Thought:** State what is currently known, what is still missing, and why the next action makes sense.
2. **Action + Action Input:** Emit one structured tool invocation.
3. **Observation:** The runtime executes the tool and injects the real output back into the conversation.
4. Repeat until the model writes a `Final Answer`.

Because autoregressive models generate tokens left-to-right, writing the `Thought` first puts the relevant constraints right into the model's recent context window before it picks the tool name and JSON arguments. That extra sentence of reasoning acts like working memory.

### Separating Reasoning from User-Facing Output

Never dump raw `Thought:` / `Action:` / `Observation:` text straight to an end user.

- End users do not want to read three paragraphs of internal monologue and raw JSON payloads.
- Internal thoughts can accidentally repeat sensitive database fields, internal policy IDs, or tool error stack traces.
- Parsing the turn into structured fields (`thought`, `action`, `action_input`, `final_answer`) lets you log the full trace for debugging and observability while returning only `final_answer` in your API response or UI.

---

## When NOT to Force Verbose Reasoning

Adding Chain-of-Thought is not free. Every reasoning token costs money and adds time-to-first-token (latency). Skip or shorten explicit reasoning in these cases:

| Scenario | Better Choice | Why |
| :--- | :--- | :--- |
| Simple routing or intent classification (`billing` vs `support`) | Direct structured JSON output (`enum`) | Extra reasoning adds 300–800 ms with almost no accuracy gain. |
| Single-lookup queries ("What is the status of ORD-1042?") | Native function calling with no CoT prompt | There is only one obvious tool call. |
| Models with built-in reasoning tokens (e.g., o3/o4-mini, Claude extended thinking, DeepSeek-R1) | Rely on the model's native reasoning mode | Asking a reasoning model to also write verbose `Thought:` blocks makes you pay for reasoning twice. |
| High-throughput batch extraction | Constrained JSON schema + low temperature | Token volume dominates cost at scale; keep outputs minimal. |

A practical rule of thumb: require a 1–2 sentence `Thought` before actions when tasks need **2+ dependent tool calls**, **policy/math logic**, or **error recovery**. Keep thoughts capped and concise.

---

## How It Works Under the Hood

Here is the exact runtime loop implemented in [`react_agent.py`](./react_agent.py):

```text
  User Query
      │
      ▼
┌───────────────────────────────────────────┐
│  1. Send messages + stop=["\nObservation:"]│
│     to LLM                                │
└────────────────────┬──────────────────────┘
                     │ raw text
                     ▼
┌───────────────────────────────────────────┐
│  2. Parse structured fields:              │
│     Thought / Action / Action Input       │
│     OR Thought / Final Answer             │
└───────┬──────────────────────────┬────────┘
        │                          │
   Parse error?               Final Answer?
        │ yes                      │ yes
        ▼                          ▼
  Feed format error          Return AgentResult
  back as Observation        (answer + separate trace)
        │
        │ valid Action
        ▼
┌───────────────────────────────────────────┐
│  3. Look up tool & execute handler(args)  │
└────────────────────┬──────────────────────┘
                     │ tool JSON string
                     ▼
┌───────────────────────────────────────────┐
│  4. Append "Observation: <result>"        │
│     and loop (up to max_steps)            │
└───────────────────────────────────────────┘
```

Two implementation details matter a lot in real code:

1. **Use a stop sequence (`stop=["\nObservation:"]`).** Without a stop sequence, many LLMs hallucinate their own `Observation:` right after `Action Input:` and then answer their own fake tool output. Stopping on `\nObservation:` forces the model to hand control back to your runtime.
2. **Feed parse errors back as observations instead of crashing.** If the model emits broken JSON or forgets `Action Input:`, return a concise format correction as the next `Observation:` so it can self-correct on the next turn (still bounded by `max_steps`).

### Text-Based ReAct vs. Native Tool Calling with a Thought Field

This lesson parses explicit `Thought:` / `Action:` / `Action Input:` text blocks so you can see the raw ReAct mechanics clearly. In production systems using OpenAI or Anthropic native tool calling, you can achieve the exact same benefit in two cleaner ways:

- Allow the model to output a brief assistant text message alongside its structured `tool_calls` array.
- Add a required `"reasoning": {"type": "string"}` property to your tool schema or structured response schema, then strip `"reasoning"` before forwarding arguments to the underlying function.

---

## Walkthrough: `react_agent.py`

[`react_agent.py`](./react_agent.py) implements a complete customer-support ReAct agent with three tools:

- `lookup_order`: fetches item category, subtotal, shipping paid, delivery age, and opened status.
- `get_return_policy`: fetches return window, opened-item restocking fee, and shipping refund rules.
- `calculate`: evaluates arithmetic expressions safely with Python's `ast` module (never `eval()`).

Key pieces in the file:

- `ParsedTurn`, `ReActStep`, `AgentResult`: typed dataclasses separating the internal reasoning trace (`result.steps`) from the user-facing response (`result.final_answer`).
- `parse_react_turn()`: regex + JSON parser that catches missing fields, invalid JSON, and turns where the model tries to emit both `Action:` and `Final Answer:`.
- `OpenAICompatibleClient` and `DemoReActLLM`: zero-dependency stdlib client for OpenAI/Ollama/vLLM plus an offline deterministic mock for `--demo` mode.

### Running the Script

Run the offline demo with the full internal trace printed above the user-facing answer:

```bash
python3 11_Day_ReAct_and_Chain_of_Thought_Reasoning/react_agent.py --demo
```

Sample output:

```text
Mode : demo
Query: Can order ORD-1042 be returned, and what exact refund amount would the customer receive?
------------------------------------------------------------------------
INTERNAL REASONING TRACE (hidden from end users in production):

[Step 1]
  Thought     : I need to check the order details for ORD-1042 first (item category, delivery date, opened status, and amounts) before applying any policy.
  Action      : lookup_order
  Action Input: {"order_id": "ORD-1042"}
  Observation : {"order_id": "ORD-1042", "customer": "Priya Nair", "item": "Mechanical Keyboard (75%)", "category": "electronics", "subtotal_usd": 180.0, "shipping_paid_usd": 12.0, "days_since_delivery": 19, "opened": true, "status": "delivered"}

[Step 2]
  Thought     : Order ORD-1042 is in the 'electronics' category, delivered 19 days ago, opened=True. I should look up the exact return policy for this category instead of guessing.
  Action      : get_return_policy
  Action Input: {"category": "electronics"}
  Observation : {"category": "electronics", "return_window_days": 30, "opened_restocking_fee_pct": 15, "shipping_refundable": false, "notes": "Opened electronics within 30 days incur a 15% restocking fee on item subtotal. Original shipping is non-refundable."}

[Step 3]
  Thought     : The order is within the 30-day window (19 days since delivery). Because it is opened, a 15% restocking fee applies to the $180.00 item subtotal, and shipping is non-refundable. Let me use the calculator tool so the refund amount is exact.
  Action      : calculate
  Action Input: {"expression": "180.0 * (1 - 15 / 100)"}
  Observation : {"expression": "180.0 * (1 - 15 / 100)", "result": 153.0}

[Step 4]
  Thought     : I now have verified order facts, return policy rules, and the calculated refund amount. I can give a clear user-facing answer without exposing raw tool JSON.
------------------------------------------------------------------------
USER-FACING FINAL ANSWER:
Yes, order ORD-1042 (Mechanical Keyboard (75%)) is eligible for a return because it was delivered 19 days ago (within the 30-day window for electronics). Since the item was opened, a 15% restocking fee applies to the $180.00 subtotal, and the $12.00 original shipping charge is non-refundable. Your total refund will be $153.00.

Status: completed in 4 step(s).
```

Test an ineligible order and hide the internal trace (simulating user-facing output):

```bash
python3 11_Day_ReAct_and_Chain_of_Thought_Reasoning/react_agent.py --demo --query "Can ORD-1088 be returned?" --hide-trace
```

Run against a live OpenAI-compatible model (OpenAI, Ollama, OpenRouter, etc.):

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
python3 11_Day_ReAct_and_Chain_of_Thought_Reasoning/react_agent.py --query "Can order ORD-1042 be returned, and how much is the refund?"
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Hallucinated observations:** If you forget `stop=["\nObservation:"]`, the model often writes `Action: lookup_order`, invents a fake `Observation: {...}`, and immediately outputs a confident `Final Answer:` without ever calling your Python function.
2. **Mixed Action + Final Answer turns:** Smaller models occasionally emit an `Action:` and a speculative `Final Answer:` in the same message. Reject that turn in the parser (as `parse_react_turn()` does) rather than blindly returning the premature answer.
3. **Infinite retry loops on bad arguments:** If a tool keeps returning an error (for example, `Order '1042' not found` when the key is `ORD-1042`) and the model keeps calling the same tool with identical inputs, you burn your entire step budget. Track `(action, canonical_json_args)` and detect duplicate failing calls early.
4. **Leaking internal thoughts to the UI:** Streaming raw LLM tokens directly to the browser exposes `Thought:` lines before you know whether the turn ends in `Action:` or `Final Answer:`. Buffer until you see `Final Answer:` (or use structured tool calling with a separate output stream).
5. **Unbounded thought length:** Without a line or token limit in the system prompt, some models write half a page of repetitive self-talk on every turn, blowing up latency and context usage.

---

## 💻 Exercises - Day 11

### Exercise: Level 1

1. Run `react_agent.py --demo` on an unknown order ID (`--query "Can ORD-9999 be returned?"`) and trace how many steps it takes to stop cleanly.
2. Add a third order (`ORD-1095`, category `"apparel"`, subtotal `$64.00`, shipping `$8.00`, delivered `12` days ago, `opened=True`) to `ORDERS_DB` and verify that the refund calculation handles `shipping_refundable: True` properly.
3. Add a unit test script (or quick assertions) for `parse_react_turn()` covering: valid action, valid final answer, malformed JSON in `Action Input`, markdown-fenced JSON, and a turn that contains both `Action:` and `Final Answer:`.

### Exercise: Level 2

1. Add duplicate-call detection inside `run_react_agent()`: if the model calls the exact same `(action, action_input)` twice in a row, inject a warning observation telling it to try different arguments or stop.
2. Add a `convert_currency` tool (for example, USD to EUR/GBP using a fixed lookup table) and test a multi-step query asking for the ORD-1042 refund in EUR.
3. Extend `ReActStep` to record wall-clock latency (`duration_ms`) for each LLM call and each tool call, then print a breakdown summary at the end of the run.

### Exercise: Level 3

1. Replace the text-based `Thought:` / `Action:` prompt with a strict JSON-per-turn protocol (`{"thought": "...", "action": "...", "action_input": {...}}` or `{"thought": "...", "final_answer": "..."}`), compare parse error rates on a live model (or local Ollama model), and measure the token difference.
2. Implement a **thought-budget guard**: track total thought tokens across all steps and automatically instruct the model to skip further tool calls and produce its best grounded answer if the reasoning budget is exceeded.
3. Add a `--redacted-trace` output mode that masks customer names and dollar amounts in the stored trace before writing JSON logs to disk, simulating production telemetry hygiene.

---

[<< Day 10](../10_Day_Parallel_Execution_and_Map_Reduce/README.md) | [Home](../README.md) | [Day 12 >>](../12_Day_Planning_and_Task_Decomposition/README.md)
