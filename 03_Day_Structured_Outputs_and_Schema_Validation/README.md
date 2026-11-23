[<< Day 2](../02_Day_Prompts_System_Instructions_and_Context/README.md) | [Home](../README.md) | [Day 4 >>](../04_Day_Tool_and_Function_Calling_Basics/README.md)

# Day 3: Structured Outputs & Schema Validation

Free-form prose is fine when a human is reading a chat bubble, but it breaks immediately when Python code needs to route a support ticket, write to a database, or invoke a tool. Today we cover strict JSON schemas, Pydantic v2 validation, and how to handle malformed model outputs safely without guessing or silently corrupting data.

## Table of Contents

- [Core Concepts & Why They Matter](#core-concepts--why-they-matter)
  - [Prompt-Only JSON vs. Constrained Structured Outputs](#prompt-only-json-vs-constrained-structured-outputs)
  - [Repair Formatting, Reject Bad Semantics](#repair-formatting-reject-bad-semantics)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Code Walkthrough](#code-walkthrough)
  - [1. Zero-Dependency Extractor & Strict Validator](#1-zero-dependency-extractor--strict-validator)
  - [2. Idiomatic Pydantic v2 Validation & JSON Schema Export](#2-idiomatic-pydantic-v2-validation--json-schema-export)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [Exercises - Day 3](#-exercises---day-3)

---

## Core Concepts & Why They Matter

### Prompt-Only JSON vs. Constrained Structured Outputs

There are three ways teams ask an LLM for JSON, ordered from least reliable to most reliable:

| Approach | How It Works | Trade-offs |
| :--- | :--- | :--- |
| **1. Prompt-only JSON** | System prompt says *"Return raw JSON with keys X, Y."* | Works on every model, but models frequently add markdown fences (` ```json `), trailing commentary, wrong types, or invented keys. |
| **2. JSON Mode (`response_format: {"type": "json_object"}`)** | Provider guarantees syntactically valid JSON, but not your exact keys or types. | Eliminates syntax errors, yet the model can still omit required fields or return `"priority": "super_high"`. |
| **3. Strict Schema (`response_format: {"type": "json_schema", "strict": true}`)** | Provider uses **constrained decoding** (masking invalid tokens at generation time) so the output conforms to your JSON Schema. | Highest reliability; however, not every local/third-party model supports full JSON Schema features (`pattern`, `minimum`, etc.), and network truncation (`max_tokens`) can still leave incomplete JSON. |

The engineering rule: **use `response_format` when your provider supports it, and ALWAYS validate in Python after the response arrives.** Treat every LLM response like an untrusted external HTTP payload.

### Repair Formatting, Reject Bad Semantics

When a model returns messy output, draw a bright line between two categories of problems:

- **Formatting wrapper issues (safe to extract):** markdown code fences (` ```json ... ``` `) or polite lead-in text like *"Sure, here is the JSON:"* around an otherwise valid object. Extracting the balanced JSON object recovers these cleanly without an extra API call.
- **Schema / semantic violations (must reject or retry):** missing required fields, values outside allowed `enum` options, wrong types (`"yes"` instead of `true`), unexpected extra fields (`approved_refund_usd: 500`), or JSON cut off mid-string by `max_tokens`. Never invent defaults for missing critical fields silently; return a structured error so your caller can retry (Day 6) or fail safely.

---

## How It Works Under the Hood

A dependable structured-output pipeline separates transport, extraction, and strict validation:

```text
  [Pydantic Model or JSON Schema]
        |                     \
        | (export schema)      \ (validate decoded dict)
        v                       \
  [LLM API Call]                 \
  (response_format: json_schema)  \
        |                          \
        v                           v
  [Raw Model String] --> [1. Extract JSON Object] --> [2. Strict Schema Validation]
                           (strip fences/prose)         (types, required, enums,
                                                         no extra fields)
                                                              |
                                         +--------------------+--------------------+
                                         |                                         |
                                    Valid object                           ValidationError
                                  (typed Python data)                (reject / feed back to retry)
```

Notice why `additionalProperties: false` (or `ConfigDict(extra="forbid")` in Pydantic v2) matters so much for agents: if an injected prompt tricks the model into adding `"approved_refund_usd": 500` to its JSON, strict validation rejects the whole payload immediately instead of quietly carrying an unauthorized field downstream.

---

## Code Walkthrough

This folder contains two complementary scripts: a zero-dependency validator you can run immediately in any Python 3.11+ environment, and an idiomatic Pydantic v2 implementation used across production Python codebases.

### 1. Zero-Dependency Extractor & Strict Validator

Open [`schema_validator_stdlib.py`](./schema_validator_stdlib.py). It implements:
- `extract_json_object()`: strips markdown fences or scans for the first balanced `{...}` object while respecting quoted strings and escape characters.
- `validate_against_schema()`: enforces `type`, `required`, `properties`, `additionalProperties: False`, `enum`, `minLength`/`maxLength`, `pattern`, `minimum`/`maximum`, and nested `array` items.
- `_matches_type()`: explicitly separates `bool` from `int` (in Python, `isinstance(True, int)` is `True`, a classic bug in hand-written validators).

Run it:

```bash
python3 03_Day_Structured_Outputs_and_Schema_Validation/schema_validator_stdlib.py
```

Sample output:

```text
=== Zero-Dependency Strict Schema Validation Demo ===

1. Clean valid JSON
  Status: PASS
  Order : ORD-1042 | Category: late_delivery

2. Markdown fence + surrounding prose (safe to repair formatting)
  Status: PASS (formatting repaired)
  Order : ORD-4490 | Category: damaged_item

3. Missing required field + unexpected extra field (must reject)
  Status: REJECT
  Error : $.priority: missing required field
  Error : $.approved_refund_usd: unexpected extra field

4. Wrong types (bool passed for integer, string passed for boolean)
  Status: REJECT
  Error : $.refund_requested: expected type 'boolean', got 'str'
  Error : $.items[0].quantity: expected type 'integer', got 'bool'

5. Truncated output cut off by max_tokens (must reject)
  Status: REJECT
  Error : $: could not extract a valid JSON object from model output
```

### 2. Idiomatic Pydantic v2 Validation & JSON Schema Export

Open [`pydantic_structured_output.py`](./pydantic_structured_output.py). It defines `ExtractedItem` and `SupportTicketExtraction` using Pydantic v2:
- `model_config = ConfigDict(extra="forbid")` rejects unknown keys and emits `"additionalProperties": false` in the generated JSON Schema.
- `Field(..., pattern=..., ge=..., le=...)` enforces domain rules directly on the data model.
- `build_openai_response_format()` converts `model_json_schema()` into the exact `response_format` payload OpenAI-compatible APIs expect.
- `validate_ticket_json()` parses raw JSON via `SupportTicketExtraction.model_validate_json()` and turns `ValidationError` into compact, structured error dictionaries.

If `pydantic` is installed (`pip install pydantic` or `uv pip install -r requirements.txt`), run:

```bash
python3 03_Day_Structured_Outputs_and_Schema_Validation/pydantic_structured_output.py --demo
```

If `pydantic` is not installed yet in your active virtual environment, the script detects the missing package cleanly, prints install instructions, and exits without a traceback so you can install it when ready.

---

## Common Pitfalls & Bugs Developers Run Into

1. **Forgetting OpenAI strict-mode schema rules.** OpenAI's `strict: true` JSON Schema mode requires `additionalProperties: false` on every object *and* requires every property in `properties` to appear in `required`. Using Pydantic fields with default values (like `Field(default_factory=list)`) omits them from `required` and triggers a `400 Bad Request` on strict endpoints unless you make them required or nullable (`str | None = Field(...)`).
2. **Relying on `json.loads` without type checks.** `json.loads` succeeds on `{"quantity": "2"}`, `{"quantity": true}`, or `[]` (a top-level list instead of a dict). Downstream math or dict lookups then crash deep in your business logic.
3. **Naive regex extraction (`\{.*\}`).** Greedy regexes break when the model outputs two JSON objects or mentions curly braces inside prose. Use a balanced-brace scanner that tracks string quotes and escape sequences, like `extract_json_object()`.
4. **Silent type coercion in Pydantic.** By default, Pydantic v2 coerces `"42"` to `42` and `"true"` to `True`. That is convenient for web forms, but if you want strict type checks on LLM outputs, pass `strict=True` to `model_validate_json(raw, strict=True)` or set `ConfigDict(strict=True, extra="forbid")`.
5. **Truncated JSON from a low `max_tokens` limit.** When a model hits `max_tokens` mid-response, `finish_reason` becomes `"length"` instead of `"stop"`. Check `finish_reason` before even attempting JSON parsing so you log the real root cause (token limit) instead of blaming the model for "bad JSON."

---

## 💻 Exercises - Day 3

### Exercise: Level 1

1. Run `schema_validator_stdlib.py` and add two new test cases to `main()`:
   - A ticket whose `order_id` is `"ORD-12"` (too short for `^ORD-\d{4,}$`)—confirm it is rejected.
   - A ticket whose `items` list is empty `[]`—confirm it passes validation.
2. Install `pydantic` in your virtual environment (`pip install pydantic` or `uv pip install -r requirements.txt`) and run `pydantic_structured_output.py --demo`. Inspect the printed JSON Schema and verify both `SupportTicketExtraction` and `ExtractedItem` include `"additionalProperties": false`.
3. In `pydantic_structured_output.py`, test passing `"refund_requested": "yes"` with default Pydantic validation versus `SupportTicketExtraction.model_validate_json(raw, strict=True)` and note the difference.

### Exercise: Level 2

1. Extend `schema_validator_stdlib.py` to support **nullable types** (e.g., `"type": ["string", "null"]`) and add an optional `tracking_number` field to `TICKET_SCHEMA` that accepts either a tracking string or `null`.
2. Add a `maxItems` constraint check to `validate_against_schema()` in `schema_validator_stdlib.py` so a model cannot return 500 duplicate items in one ticket extraction.
3. Update `pydantic_structured_output.py` to strip markdown fences (reusing the extraction idea from `schema_validator_stdlib.py`) before calling `model_validate_json()`, and log a metric counter whenever formatting repair was needed.

### Exercise: Level 3

1. Build a small **single-retry validator wrapper** around `call_live_structured()` in `pydantic_structured_output.py`: if the first response fails `validate_ticket_json()`, send one follow-up turn containing the exact structured `errors` list and ask the model to fix only the invalid fields (we will generalize this pattern across tools on Day 6).
2. Add a check in `call_live_structured()` that inspects `data["choices"][0]["finish_reason"]`: if it equals `"length"`, raise a dedicated `TruncatedOutputError` instead of attempting to parse half-finished JSON.
3. Write a unit test script (using stdlib `unittest` or `pytest`) with at least 8 edge cases for `extract_json_object()`—including JSON strings that contain literal `{` and `}` characters inside string values (e.g., `{"summary": "User wrote {hello} in chat"}`)—and verify every case passes.

---

[<< Day 2](../02_Day_Prompts_System_Instructions_and_Context/README.md) | [Home](../README.md) | [Day 4 >>](../04_Day_Tool_and_Function_Calling_Basics/README.md)
