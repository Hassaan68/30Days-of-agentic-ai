[<< Day 5](../05_Day_Building_Your_First_Agent_Loop/README.md) | [Home](../README.md) | [Day 7 >>](../07_Day_Short_Term_Memory_and_Conversation_State/README.md)

# Day 6: Error Handling and Self-Correction

Most agent loops break the first time a tool fails—not because the task is impossible, but because the runtime either crashes with an unhandled exception or passes a vague error string back to the model. Today we build a tool execution layer that turns recoverable failures into structured observations the model can fix, while cutting off fatal errors and infinite retry loops immediately.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `self_correcting_agent.py`](#walkthrough-self_correcting_agentpy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 6](#-exercises---day-6)

---

## Core Concept & Why It Matters

In traditional backend code, raising an exception and returning HTTP 400 makes sense because a human developer will inspect the stack trace and fix the caller. In an agent loop, the caller is an LLM generating tool arguments on the fly. Models routinely drop a closing brace in JSON, pass an enum string like `"broken"` instead of `"damaged_item"`, or guess an ID before calling a lookup tool.

If your runtime lets `json.JSONDecodeError` or `KeyError` crash the process, every small model mistake turns into a failed user request. On the other hand, if you swallow every error and let the model retry forever, a broken permission or dead endpoint will burn through your step budget and API credits while repeating the exact same call.

Reliable error handling in agent loops comes down to three habits:

1. **Return errors as structured tool observations.** Give the model `ok: false`, a machine-readable `error_type`, a `retryable` boolean, a concrete `message`, and an actionable `hint` (for example, valid enum values or which discovery tool to call next).
2. **Separate retryable errors from fatal errors.** A bad parameter or missing ID can be corrected by the model. A `permission_denied`, `invalid_state`, or invalid API key error cannot—retrying wastes tokens and risks tripping rate limits or audit alerts.
3. **Put hard caps around self-correction.** Track retries per tool, consecutive errors across turns, overall step budgets, and exact duplicate failing calls so the runtime—not the prompt—guarantees termination.

### Retryable vs. Fatal Error Taxonomy

| Error Category | Example | Retryable by Model? | Expected Runtime Behavior |
| :--- | :--- | :---: | :--- |
| Malformed JSON arguments | Missing `}` in tool args | Yes | Return `json_parse_error` with parse offset and hint |
| Schema / validation mismatch | `amount_usd = 60` when max is `48.50` | Yes | Return `validation_error` with allowed range or enum values |
| Resource not found | `ORD-9999` does not exist | Yes | Return `not_found` and hint to call `list_orders` |
| Transient upstream failure | HTTP 429 / 503 timeout | No (runtime handles) | Retry in Python with exponential backoff, then fail cleanly |
| Policy / permission denial | Refund exceeds agent limit | No | Set `retryable=false`; stop tool loop and explain blocker |
| Duplicate failing call | Exact same broken payload sent twice | No | Trip circuit breaker immediately |
| Unhandled tool bug | `ZeroDivisionError` inside tool | No | Catch, log full traceback privately, return safe fatal error |

Notice the distinction between **model retries** and **runtime retries**. If a database times out, asking the LLM to "try calling the tool again" adds an extra LLM round-trip, latency, and token cost for no reason. Handle transient network retries inside your Python wrapper with backoff, and reserve model self-correction for mistakes where the model actually needs to change its arguments or plan.

---

## How It Works Under the Hood

Every tool call from the model passes through a guarded execution wrapper before hitting your business logic:

```text
  LLM proposes tool_call(name, raw_args)
                   │
                   ▼
      ┌─────────────────────────┐
      │ 1. Tool exists?         │──No──► ok=false, unknown_tool (retryable=true)
      └────────────┬────────────┘
                   │ Yes
                   ▼
      ┌─────────────────────────┐
      │ 2. Duplicate failed     │──Yes─► ok=false, duplicate_failed_call
      │    call or cap reached? │        (retryable=false -> CIRCUIT BREAKER)
      └────────────┬────────────┘
                   │ No
                   ▼
      ┌─────────────────────────┐
      │ 3. Parse JSON & check   │──Err─► record failed hash, increment counter
      │    function signature   │        ok=false, json_parse_error (retryable=true)
      └────────────┬────────────┘
                   │ Valid
                   ▼
      ┌─────────────────────────┐
      │ 4. Execute tool         │──ToolError(retryable=true)──► feed hint to LLM
      │                         │──ToolError(retryable=false)─► stop & explain
      │                         │──Unexpected Exception───────► stop & escalate
      └────────────┬────────────┘
                   │ Success
                   ▼
         Reset consecutive_errors = 0
         Append {"ok": true, ...} as tool message
```

### Why Hash Failed Tool Calls?

Smaller and mid-sized models often get stuck repeating the exact same failing call even when you tell them it failed. Hashing `(tool_name, canonical_json_args)` into a `failed_signatures` set catches that behavior on the second attempt:

```python
def _canonical_call_signature(tool_name: str, raw_args: str) -> str:
    try:
        parsed = json.loads(raw_args)
        normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except json.JSONDecodeError:
        normalized = raw_args.strip()
    return hashlib.sha256(f"{tool_name}:{normalized}".encode("utf-8")).hexdigest()[:16]
```

Canonicalizing JSON keys (`sort_keys=True`) matters because `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` represent the exact same call.

### Structured Error Observation Shape

Instead of returning `"Error: invalid refund"`, return a compact JSON object:

```json
{
  "ok": false,
  "tool": "issue_refund",
  "error_type": "validation_error",
  "retryable": true,
  "message": "amount_usd=60.0 is out of range. Must be > 0 and <= 48.50.",
  "attempts_used": 1,
  "attempts_remaining": 2,
  "hint": "Set amount_usd to 48.50 for a full refund.",
  "details": {
    "max_refundable_usd": 48.5
  }
}
```

Giving the model `attempts_remaining` and concrete `details` cuts down guesswork and usually resolves validation errors in a single retry.

---

## Walkthrough: `self_correcting_agent.py`

Open [`self_correcting_agent.py`](./self_correcting_agent.py). It implements a self-contained order-and-refund agent using Python's standard library (`urllib.request` for live OpenAI-compatible endpoints, plus a deterministic `--demo` mode with three scripted scenarios).

Key pieces to inspect in the file:

- **`ToolError` dataclass**: Carries `error_type`, `message`, `retryable`, `hint`, and `details`, with `to_observation(...)` to format consistent JSON tool responses.
- **`RetryPolicy` dataclass**: Configures `max_steps=8`, `max_retries_per_tool=3`, and `max_consecutive_errors=3`.
- **`execute_tool_safely(...)`**: Catches unknown tools, duplicate failing signatures, per-tool retry caps, `json.JSONDecodeError`, `TypeError` (wrong/missing kwargs), `ToolError`, and unexpected Python exceptions. Nothing escapes unhandled.
- **`run_self_correcting_agent(...)`**: Drives the loop, resets `consecutive_errors` after any successful tool call, immediately halts on circuit-breaker conditions, and on policy/fatal errors lets the model generate one tool-free response (`tools=[]`) explaining the blocker to the user.

### Run the Offline Demo

From the repository root:

```bash
python 06_Day_Error_Handling_and_Self_Correction/self_correcting_agent.py --demo
```

Sample output:

```text
=== Scenario 1: Recoverable Error -> Self-Correction ===
Status: completed | Steps: 6 | Tool errors: 3
Trace:
  - Step 1: get_order({"order_id": "ORD-9999") -> ERROR [json_parse_error, retryable=True]: Malformed JSON arguments: Expecting ',' delimiter at position 23.
  - Step 2: get_order({"order_id": "ORD-9999"}) -> ERROR [not_found, retryable=True]: Order 'ORD-9999' does not exist.
  - Step 3: list_orders({"customer": "Maya"}) -> OK
  - Step 4: issue_refund({"order_id": "ORD-1042", "amount_usd": 60.0, "reason": "broken"}) -> ERROR [validation_error, retryable=True]: amount_usd=60.0 is out of range. Must be > 0 and <= 48.50.
  - Step 5: issue_refund({"order_id": "ORD-1042", "amount_usd": 48.5, "reason": "damaged_item"}) -> OK
  - Step 6: assistant produced final response.
Final answer: Refund REF-1042 for $48.50 on Maya Lin's order ORD-1042 has been processed with reason 'damaged_item'.

=== Scenario 2: Fatal Policy Error -> Stop Without Retrying ===
Status: stopped_fatal | Steps: 1 | Tool errors: 1
Trace:
  - Step 1: issue_refund({"order_id": "ORD-2099", "amount_usd": 1450.0, "reason": "damaged_item"}) -> ERROR [permission_denied, retryable=False]: Order ORD-2099 ($1450.00) exceeds the automated refund limit and requires manager approval.
  - Step 1: non-retryable tool error -> returned user explanation.
Final answer: I cannot process the $1,450.00 refund for order ORD-2099 automatically because it exceeds the automated refund threshold. A manager needs to approve it.

=== Scenario 3: Duplicate Failing Call -> Circuit Breaker ===
Status: stopped_circuit_breaker | Steps: 2 | Tool errors: 2
Trace:
  - Step 1: get_order({"order_id": "ORD-0000"}) -> ERROR [not_found, retryable=True]: Order 'ORD-0000' does not exist.
  - Step 2: get_order({"order_id": "ORD-0000"}) -> ERROR [duplicate_failed_call, retryable=False]: Identical tool call already failed earlier in this run.
  - Step 2: circuit breaker tripped (Identical tool call already failed earlier in this run.).
Final answer: Stopped automatically to prevent an infinite retry loop. Last error: Identical tool call already failed earlier in this run.
```

### Run Against a Live Model

You can test the same loop against OpenAI, Ollama, or any OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
python 06_Day_Error_Handling_and_Self_Correction/self_correcting_agent.py \
  --query "Please refund Maya Lin's damaged order for $60."
```

You can also run a single demo scenario with `--scenario recoverable`, `--scenario fatal`, or `--scenario loop_breaker`.

---

## Common Pitfalls & Bugs Developers Run Into

1. **Leaking raw stack traces into tool observations.** Dumping 40 lines of Python tracebacks into the `tool` message bloats context, occasionally exposes file paths or credentials, and gives the model less useful signal than a concise 2-line structured JSON error. Log the full traceback to your application logger; send only a clean summary to the model.
2. **Forgetting to strip tools on the fatal explanation turn.** When a fatal policy error fires, if you ask the model to summarize the failure while still passing `tools=TOOLS_SCHEMA`, some models ignore `retryable=false` and try calling the tool again anyway. Passing `tools=[]` enforces the stop in code rather than hoping the prompt holds.
3. **Treating `bool` as `int` in Python validation.** In Python, `isinstance(True, int)` evaluates to `True`. If a model passes `amount_usd: true`, a naive `isinstance(amount_usd, (int, float))` check accepts it as `1.0`. Always exclude `bool` explicitly (or validate with Pydantic).
4. **Blocking legitimate retries after state changes.** Duplicate-call hashing is great for read-only or unchanged-state calls, but be careful with stateful tools: if a tool fails because a file did not exist yet, and the agent creates the file in step 2, calling `read_file` with the same path in step 3 should succeed! In production, clear or scope `failed_signatures` whenever a mutating tool succeeds.
5. **Not reporting all validation errors at once.** In `tool_issue_refund`, if both `amount_usd` and `reason` are invalid and you raise on the first check, the model needs two round-trips to fix both fields. Collecting all field errors into a single `validation_errors` list lets the model repair everything in one step.

---

## 💻 Exercises - Day 6

### Exercise: Level 1

1. **Run each scenario individually.** Run `self_correcting_agent.py --demo --scenario recoverable`, `--scenario fatal`, and `--scenario loop_breaker`, and trace how `tool_error_counts`, `failed_signatures`, and `consecutive_errors` change at each step.
2. **Test the `invalid_state` fatal path.** Add a fourth demo scenario for `ORD-3011` (which is already refunded) and verify the agent returns `status="stopped_fatal"` in 1 step without retrying.
3. **Inspect missing-argument handling.** Modify Scenario 1 Step 1 so the model omits `order_id` completely (`{}`). Verify that `execute_tool_safely` catches the `TypeError` and returns `error_type="schema_mismatch"`.

### Exercise: Level 2

1. **Batch all validation errors in one observation.** Refactor `tool_issue_refund` so that when both `amount_usd` and `reason` are invalid (as in Scenario 1 Step 4), it returns a single `ToolError` containing a list of every failed field check and hint so the model fixes all issues in one turn.
2. **Add runtime exponential backoff for transient errors.** Add a simulated flaky tool (for example, `check_payment_gateway` that raises a `TransientToolError` twice before succeeding). Handle transient errors inside `execute_tool_safely` using Python retries with backoff (`0.1s`, `0.2s`, `0.4s`) instead of burning extra LLM turns.
3. **Clear duplicate-call cache after state mutations.** Add a `create_order` tool and mark tools as `read_only=True` or `mutates_state=True`. When a mutating tool succeeds, clear `failed_signatures` so a previously missing resource can now be queried again.

### Exercise: Level 3

1. **Build an automatic fallback strategy.** Extend `RetryPolicy` with a `fallback_tools` map (for example, if `primary_search` hits its retry cap, automatically hint or switch to `backup_search`) and verify it in a dedicated demo scenario.
2. **Separate public error messages from internal telemetry.** Add structured logging (`logging` module) that records full exception details, latency, and raw model arguments to a JSONL log file while ensuring the tool observation sent back to the model never exposes stack traces or internal secrets.
3. **Write a unit test suite for error boundaries.** Create a `test_self_correcting_agent.py` file using `unittest` that covers: malformed JSON, unknown tool names, `bool` passed for `amount_usd`, duplicate call detection with reordered JSON keys, per-tool retry cap enforcement, consecutive-error circuit breaking, and fatal error tool-stripping.

---

[<< Day 5](../05_Day_Building_Your_First_Agent_Loop/README.md) | [Home](../README.md) | [Day 7 >>](../07_Day_Short_Term_Memory_and_Conversation_State/README.md)
