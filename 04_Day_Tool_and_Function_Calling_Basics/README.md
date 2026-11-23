[<< Day 3](../03_Day_Structured_Outputs_and_Schema_Validation/README.md) | [Home](../README.md) | [Day 5 >>](../05_Day_Building_Your_First_Agent_Loop/README.md)

# Day 4: Tool & Function Calling Basics

Tool calling is the bridge between a text-generating LLM and real software systems—databases, internal APIs, calculators, and local Python code. Today we cover how tool schemas are defined with JSON Schema, how to parse and validate model-requested tool calls inside a clean `ToolRegistry`, and how to return observations in the exact message format the model expects.

## Table of Contents

- [Core Concepts & Why They Matter](#core-concepts--why-they-matter)
  - [The Model Proposes; Your Code Executes](#the-model-proposes-your-code-executes)
  - [Writing Good Tool Schemas](#writing-good-tool-schemas)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Code Walkthrough](#code-walkthrough)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [Exercises - Day 4](#-exercises---day-4)

---

## Core Concepts & Why They Matter

### The Model Proposes; Your Code Executes

A persistent misconception is that "tool calling" means the LLM runs Python functions on its own servers. It does not. Tool calling is simply **structured output (Day 3) specialized for function dispatch**:

1. You send the model a list of `tools`—each describing a function's `name`, `description`, and JSON Schema `parameters`.
2. Instead of answering in plain text, the model can return a `tool_calls` array containing one or more objects with a unique `id`, a function `name`, and a JSON string of `arguments`.
3. Your Python application looks up that tool in a registry, parses and validates the arguments, executes the local function, and appends a `{"role": "tool", "tool_call_id": ..., "content": ...}` message to `messages`.
4. You send the updated `messages` back to the model so it can read the tool output and either request another tool or write the final answer.

Because your code sits between the model and every external action, you control argument validation, permissions, timeouts, and error handling.

### Writing Good Tool Schemas

A model only sees your tool's `name`, `description`, and `parameters` schema—it cannot read your Python function body. Good schemas follow four practical rules:

- **Use specific verb-noun names:** `get_order_status` and `calculate_shipping` beat vague names like `process_order` or `do_math`.
- **Describe inputs and units explicitly:** Mention `"subtotal in USD"` or `"package weight in kg"` right in the description so the model does not pass pounds or cents by mistake.
- **Lock down values with `enum`, `minimum`, and `additionalProperties: false`:** Restricting `target_currency` to `["USD", "EUR", "GBP", "JPY"]` prevents invented currency codes.
- **Keep tool outputs compact and structured:** Return concise JSON dictionaries (`{"ok": true, "result": {...}}`) rather than raw 40 KB HTML pages that flood the context window.

---

## How It Works Under the Hood

Here is the exact message sequence across a tool-calling round-trip:

```text
Request 1:
  messages = [
    {"role": "system", "content": "Use tools for live data..."},
    {"role": "user",   "content": "Check ORD-1042 shipping & EUR total"}
  ]
  tools = [get_order_status, calculate_shipping, convert_currency]
        |
        v
Response 1 (from LLM):
  {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_order_001", "type": "function",
       "function": {"name": "get_order_status", "arguments": "{\"order_id\": \"ORD-1042\"}"}}
  ]}
        |
        v  (Your Python ToolRegistry validates args & runs get_order_status("ORD-1042"))
        |
Request 2 (append BOTH the assistant message AND the matching tool message):
  messages = [
    system,
    user,
    assistant (with tool_calls id="call_order_001"),
    {"role": "tool", "tool_call_id": "call_order_001", "name": "get_order_status",
     "content": "{\"ok\": true, \"result\": {\"subtotal_usd\": 120.0, ...}}"}
  ]
        |
        v
Response 2 (from LLM):
  Next tool_calls (can be multiple calls in parallel!) OR final assistant text
```

Two wire-level details trip up almost everyone the first time:
1. `function.arguments` arrives as a **JSON-encoded string**, not an already-parsed dictionary—you must call `json.loads()` on it defensively.
2. Every `tool_call` in the assistant message MUST be followed by a `role: "tool"` message with the exact matching `tool_call_id`. Omitting even one ID triggers a `400 Bad Request` on the next turn.

---

## Code Walkthrough

Open [`tool_calling_basics.py`](./tool_calling_basics.py). It builds a complete, self-contained tool-calling stack using only the Python standard library:

- `ToolSpec`: binds a Python callable (`handler`) to its `name`, `description`, and JSON Schema `parameters`, and exports OpenAI-compatible definitions via `to_openai_tool()`.
- `validate_tool_args()`: checks required keys, unexpected extra arguments, parameter types (including `bool` vs `int`/`float`), enums, and numeric minimums before invoking any Python function.
- `ToolRegistry`: registers tools, exposes `openai_schemas()`, and executes calls via `execute_tool_call()`, always returning a properly formatted `role: "tool"` observation dictionary—even when the tool name is unknown, arguments fail validation, or the handler raises an exception.
- Three local tools: `get_order_status`, `calculate_shipping`, and `convert_currency`.
- Live OpenAI-compatible client (`call_live_model`) plus a deterministic `--demo` mock (`call_mock_model`) demonstrating both single and parallel tool calls.

Run in offline demo mode:

```bash
python3 04_Day_Tool_and_Function_Calling_Basics/tool_calling_basics.py --demo
```

Sample output:

```text
=== Tool Calling Round-Trip [DEMO (offline mock)] ===
Registered tools: ['get_order_status', 'convert_currency', 'calculate_shipping']

User: Check order ORD-1042, estimate its shipping cost to its destination zone, and tell me the order subtotal in EUR.

Turn 1 [Model requested 1 tool call(s)]:
  -> Call call_order_001: get_order_status({"order_id": "ORD-1042"})
  <- Tool observation: {"ok": true, "result": {"found": true, "order_id": "ORD-1042", "status": "in_transit", "carrier": "FastShip", "subtotal_usd": 120.0, "weight_kg": 2.5, "destination_zone": "EU"}}

Turn 2 [Model requested 2 tool call(s)]:
  -> Call call_ship_002: calculate_shipping({"weight_kg": 2.5, "destination_zone": "EU"})
  <- Tool observation: {"ok": true, "result": {"weight_kg": 2.5, "destination_zone": "EU", "shipping_cost_usd": 26.25}}
  -> Call call_fx_003: convert_currency({"amount_usd": 120.0, "target_currency": "EUR"})
  <- Tool observation: {"ok": true, "result": {"amount_usd": 120.0, "target_currency": "EUR", "rate": 0.92, "converted_amount": 110.4}}

Turn 3 [Assistant Final Answer]:
  Order ORD-1042 is currently in_transit via FastShip. Its subtotal is $120.00 USD (110.40 EUR at rate 0.92), and estimated shipping to EU for 2.5 kg is $26.25 USD.

=== Defensive Argument Validation Check ===
  Tool 'convert_currency' -> {"ok": false, "error": "Argument validation failed", "details": ["Argument 'amount_usd' must be >= 0, got -10", "Argument 'target_currency' must be one of ['USD', 'EUR', 'GBP', 'JPY'], got 'BTC'"]}
  Tool 'delete_database' -> {"ok": false, "error": "Unknown tool 'delete_database'. Available: ['calculate_shipping', 'convert_currency', 'get_order_status']"}
```

To run against a live tool-capable model (such as `gpt-4o-mini` or local Ollama `llama3.2` / `qwen2.5:7b`), set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `MODEL_NAME` and run without `--demo`.

---

## Common Pitfalls & Bugs Developers Run Into

1. **Forgetting to append the assistant's `tool_calls` message before appending `role: "tool"` messages.** If you append only the `tool` observation without the `assistant` message that requested that `tool_call_id`, the API rejects the conversation as malformed.
2. **Assuming only one tool call per turn.** Models frequently return 2 or 3 parallel `tool_calls` in a single assistant message (as shown in Turn 2 above). Always loop over `assistant_msg.get("tool_calls") or []` and reply to every single `id`.
3. **Executing raw `**json.loads(arguments)` without validation.** A hallucinated extra keyword argument raises `TypeError: got an unexpected keyword argument`, and an unvalidated path or SQL string creates a security hole. Always validate arguments against the schema first.
4. **Crashing the whole process when a tool fails.** Raising an unhandled exception inside a tool kills the entire agent turn. Catching tool errors and returning `{"ok": false, "error": "..."}` as a tool observation lets the model recover or explain the failure clearly (we build full self-correction loops on Day 6).
5. **Registering 40+ overlapping tools at once.** Too many similar tools inflate every turn's input token count and confuse tool selection. Start with a small, distinct set of tools with crisp descriptions.

---

## 💻 Exercises - Day 4

### Exercise: Level 1

1. Run `tool_calling_basics.py --demo` and inspect the exact keys inside the `ToolSpec.to_openai_tool()` output by printing `json.dumps(registry.openai_schemas(), indent=2)` at startup.
2. Add a fourth tool `list_supported_currencies()` (taking no arguments: `required: []`, `properties: {}`) to `build_default_registry()` and verify `execute_tool_call` runs it cleanly when `arguments` is `"{}"`.
3. Add a third case to `run_demo_validation_check()` that passes malformed JSON (e.g., `'{"order_id": "ORD-1042'`) as `arguments` and confirm the registry returns a clean JSON error observation instead of crashing.

### Exercise: Level 2

1. Add a `--query` CLI flag to `tool_calling_basics.py` so you can test arbitrary user prompts against a live OpenAI or Ollama model (e.g., `"How much does shipping cost for ORD-2088 and what is its subtotal in GBP?"`).
2. Add a `max_content_chars: int = 2000` guard to `ToolRegistry._wrap_observation()` so an unexpectedly huge tool result is truncated cleanly with a `"truncated": true` flag before being appended to `messages`.
3. Add a `read_only: bool = True` metadata flag to `ToolSpec` and expose a `registry.list_tools(include_write_tools=False)` method so read-only assistants cannot accidentally receive write-capable tool schemas.

### Exercise: Level 3

1. Extend `ToolRegistry` with a `@registry.tool(...)` Python decorator that automatically registers a function and validates at registration time that every key in `parameters["properties"]` actually exists in the Python function's signature (using `inspect.signature`).
2. Add per-tool execution timing (`duration_ms`) and a simple audit list on `ToolRegistry` recording every invocation `(call_id, tool_name, ok, duration_ms)` without logging sensitive argument values.
3. Run `tool_calling_basics.py` against a live model with a prompt asking for a nonexistent order (`ORD-9999`). Inspect how the model handles `{"found": false, "error": "..."}` and verify it does not hallucinate fake order totals or shipping costs.

---

[<< Day 3](../03_Day_Structured_Outputs_and_Schema_Validation/README.md) | [Home](../README.md) | [Day 5 >>](../05_Day_Building_Your_First_Agent_Loop/README.md)
