[<< Day 12](../12_Day_Planning_and_Task_Decomposition/README.md) | [Home](../README.md) | [Day 14 >>](../14_Day_Agentic_RAG/README.md)

# Day 13: Reflection and Self-Critique Loops

Today we build a **generator-critic (reflection) loop** that catches defects in a model's output before it ever reaches a user or downstream system. By combining fast deterministic checks (syntax, required fields, leak guards) with a structured LLM rubric and hard stopping rules, you turn rough first drafts into reliable outputs without falling into endless nitpicking.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `reflection_loop.py`](#walkthrough-reflection_looppy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 13](#-exercises---day-13)

---

## Core Concept & Why It Matters

Even strong models miss details on a first pass: a broken trailing comma in an example JSON payload, a forgotten header name, a leaked internal ticket ID, or a subtle edge case in generated code.

The simplest instinct is to ask the model *"Are you sure? Please improve your answer."* That usually fails in two opposite ways:

- **Rubber-stamping:** Without an explicit checklist, the model glances at its own text and says *"Looks good!"*
- **Endless nitpicking:** When told to "critique this," an LLM invents stylistic complaints round after round—burning tokens and sometimes rewriting a good answer into a worse one.

A production **Reflection Loop** fixes both problems with three rules:

1. **Separate the Generator from the Critic.** The critic gets its own system prompt, temperature `0.0`, and an explicit JSON rubric—not vague encouragement.
2. **Run deterministic checks first.** Anything a Python function can verify objectively (valid JSON, `ast.parse`, unit tests, forbidden substrings, exact dates, length bounds) should be checked in code. Deterministic checks are free, take milliseconds, and never hallucinate a pass.
3. **Define explicit stopping criteria and keep the best draft.** Stop immediately once all deterministic checks pass, blocking defects hit zero, and rubric scores clear a threshold—or stop at `max_rounds` and return the highest-ranked draft seen so far.

### When Reflection Is Worth the Extra Cost

| Task Type | Use Reflection? | Why |
| :--- | :--- | :--- |
| Customer-facing advisories, code generation, SQL queries, compliance reports | **Yes** | High cost of mistakes; objective checks (parsers, tests, leak filters) catch real defects cheaply. |
| Real-time voice / autocomplete / simple chat replies | **No** | Doubling or tripling latency breaks user experience. |
| Single-fact extraction with strict schema | **Usually no** | Schema validation + one targeted retry is cheaper than full rubric critique. |

---

## How It Works Under the Hood

Here is the loop implemented in [`reflection_loop.py`](./reflection_loop.py):

```text
  Source Notes / Task Spec
             │
             ▼
┌────────────────────────────┐
│  1. Generator              │◄─────────────────────────────┐
│  Draft (or revise using    │                              │
│  exact defects from last   │                              │
│  round)                    │                              │
└─────────────┬──────────────┘                              │
              │ draft                                       │
              ▼                                             │
┌────────────────────────────┐                              │
│  2. Deterministic Checks   │  (Python code, 0 LLM cost)   │
│  - required sections       │                              │
│  - exact factual tokens    │                              │
│  - no internal leaks       │                              │
│  - valid JSON block        │                              │
│  - word count bounds       │                              │
└─────────────┬──────────────┘                              │
              │ check results                               │
              ▼                                             │
┌────────────────────────────┐                              │
│  3. Structured Rubric      │                              │
│     Critic                 │                              │
│  Outputs: rubric_scores +  │                              │
│  typed defects (blocking / │                              │
│  minor) with fix guidance  │                              │
└─────────────┬──────────────┘                              │
              │                                             │
              ▼                                             │
   Update best_round if higher ranked                       │
              │                                             │
              ▼                                             │
   All deterministic PASS &                                 │
   0 blocking defects &                                     │
   rubric avg >= min_score? ── no (and round < max_rounds) ─┘
              │ yes (or max_rounds hit)
              ▼
   Return ReflectionResult(best_round, final_draft, history)
```

### Why `overall_rank()` Matters

Revisions do not always improve monotonically. On round 3, a generator might fix a phrasing suggestion while accidentally dropping a required heading. By ranking every round with a tuple:

```python
(
    1 if self.deterministic_passed else 0,
    -len(self.blocking_defects),
    self.average_rubric_score,
)
```

the controller always preserves the strongest draft seen across the entire run. If `max_rounds` expires after a regression on the final turn, the user still gets the earlier winning draft.

---

## Walkthrough: `reflection_loop.py`

[`reflection_loop.py`](./reflection_loop.py) drafts a customer-facing **Webhook API v2 Migration Advisory** from raw engineering notes that deliberately contain internal-only fields (`JIRA-4821` and `webhooks-v2.internal.corp`).

Five deterministic checks run on every draft:

1. `required_sections`: verifies `## Summary`, `## Required Action`, `## Example Payload`, and `## Timeline`.
2. `required_facts`: verifies `2026-09-15`, `X-Webhook-Signature-256`, and `HMAC-SHA256`.
3. `no_internal_leaks`: rejects any mention of `JIRA-`, `internal.corp`, or `webhooks-v2.internal`.
4. `valid_json_example`: extracts the fenced ```` ```json ```` block, parses it with `json.loads()`, and checks `"api_version": "2026-09"`.
5. `word_count_bounds`: keeps the advisory concise (`80–260` words).

Those check results are passed directly into the rubric critic (`accuracy`, `actionability`, `clarity_and_tone`), which produces typed `Defect` items (`blocking` vs `minor`) with concrete fix instructions.

### Running the Script

Run the offline 3-round demo:

```bash
python3 13_Day_Reflection_and_Self_Critique_Loops/reflection_loop.py --demo
```

Sample output:

```text
Mode: demo
========================================================================

[Round 1] Deterministic: 0/5 passed | Rubric Avg: 2.50/5.00 | Blocking Defects: 4
  - [FAIL] required_sections: Missing section(s): ## Timeline
  - [FAIL] required_facts: Missing required fact(s): X-Webhook-Signature-256
  - [FAIL] no_internal_leaks: Leaked internal identifier(s): JIRA-, internal.corp, webhooks-v2.internal
  - [FAIL] valid_json_example: Invalid JSON syntax in example block: Illegal trailing comma before end of object (line 4).
  - [FAIL] word_count_bounds: Word count is 56 (allowed: 80-260).
...
[Round 2] Deterministic: 4/5 passed | Rubric Avg: 3.90/5.00 | Blocking Defects: 1
  - [PASS] required_sections: All 4 required sections present.
  - [FAIL] required_facts: Missing required fact(s): X-Webhook-Signature-256
  - [PASS] no_internal_leaks: No internal hostnames or ticket IDs leaked.
  - [PASS] valid_json_example: Valid JSON block with "api_version": "2026-09".
  - [PASS] word_count_bounds: Word count is 97 (allowed: 80-260).
...
[Round 3] Deterministic: 5/5 passed | Rubric Avg: 4.70/5.00 | Blocking Defects: 0
  - [PASS] required_sections: All 4 required sections present.
  - [PASS] required_facts: Exact cutoff date and signature header preserved.
  - [PASS] no_internal_leaks: No internal hostnames or ticket IDs leaked.
  - [PASS] valid_json_example: Valid JSON block with "api_version": "2026-09".
  - [PASS] word_count_bounds: Word count is 126 (allowed: 80-260).
  - Rubric (accuracy): 5.0/5.0 — Matches all source facts with exact cutoff date and header.
  - Rubric (actionability): 4.6/5.0 — Clear numbered migration checklist plus valid JSON example.
  - Rubric (clarity_and_tone): 4.5/5.0 — Concise, direct customer language with no internal jargon.

========================================================================
STOP REASON: Accepted on round 3: all deterministic checks passed, 0 blocking defects, rubric avg 4.70 >= 4.20.
SELECTED DRAFT: Round 3
```

Test hitting `max_rounds` before acceptance (to see best-draft fallback):

```bash
python3 13_Day_Reflection_and_Self_Critique_Loops/reflection_loop.py --demo --max-rounds 2
```

Run with a live OpenAI-compatible model:

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
python3 13_Day_Reflection_and_Self_Critique_Loops/reflection_loop.py --max-rounds 3 --min-score 4.2
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Vague critiques ("make it clearer and more thorough"):** Without a concrete defect location and fix instruction, the generator rewrites the entire document from scratch and breaks things that were already working.
2. **Same-model blind spots:** Using the exact same model and temperature to write and review factual claims often misses subtle reasoning bugs. Even when using one model family, use a separate system prompt, temperature `0.0`, and deterministic validators for anything checkable in code.
3. **Paying for an LLM critic when deterministic checks already failed:** If a generated Python script fails `ast.parse` or a JSON payload fails `json.loads()`, you can skip the LLM critic call entirely on that round and send the exact compiler/parser error straight back to the generator.
4. **No plateau detection:** If round 2 scores `3.8` and round 3 scores `3.8` with the exact same blocking defect, running rounds 4 and 5 rarely helps. Stop early when scores stop improving for 2 consecutive rounds.
5. **Passing the entire conversation history of failed drafts:** Stuffing 4 old broken drafts into the prompt confuses the generator and inflates token cost. Pass only the source specification, the **most recent draft**, and its **structured defect list**.

---

## 💻 Exercises - Day 13

### Exercise: Level 1

1. Run `reflection_loop.py --demo --max-rounds 2` and verify that the stop reason explains why it stopped early and still selects Round 2 over Round 1.
2. Add a 6th deterministic check `http_410_mentioned` that verifies the advisory explains what happens after the cutoff (`410` or `410 Gone`). Run the demo and confirm Round 1 fails it while Rounds 2 and 3 pass.
3. Write a small test script that feeds a deliberately regressed Round 3 draft into `CritiqueReport.overall_rank()` and asserts that Round 2 outranks Round 3.

### Exercise: Level 2

1. Add a **short-circuit optimization** in `run_reflection_loop()`: add a `--skip-llm-on-det-fail` flag so when deterministic checks fail, the loop skips calling `backend.critique()` and feeds the failed deterministic check messages straight back to `backend.generate()`. Count and print how many critic LLM calls are saved.
2. Add **plateau detection**: if the rubric average fails to improve by at least `0.2` points compared to the previous round (and deterministic pass count did not increase), stop early with a clear `stop_reason` instead of spending remaining rounds.
3. Extend `OpenAIReflectionBackend` so the Generator and Critic can use *different* models via environment variables (`GENERATOR_MODEL` and `CRITIC_MODEL`, e.g., a fast model generating and a stronger model critiquing).

### Exercise: Level 3

1. Build a **code-generation reflection loop** in a new script `code_reflection.py`: ask a model to write a Python function (for example, `parse_cron_field(expr: str, min_val: int, max_val: int) -> list[int]`), run deterministic checks (`ast.parse`, type annotation check, and a suite of `pytest`/assertion test cases in a subprocess with a 2-second timeout), then feed failing test tracebacks back to the generator for up to 3 rounds.
2. Log every reflection run as a JSONL record (`round_number`, `deterministic_pass_rate`, `rubric_scores`, `token_estimate`, `accepted`) so you can measure whether Round 2 and Round 3 actually justify their latency/cost on a 10-prompt benchmark.
3. Add a **diff view** (`difflib.unified_diff`) printed between Round `N-1` and Round `N` so reviewers can audit exactly what changed in each revision rather than reading full drafts side-by-side.

---

[<< Day 12](../12_Day_Planning_and_Task_Decomposition/README.md) | [Home](../README.md) | [Day 14 >>](../14_Day_Agentic_RAG/README.md)
