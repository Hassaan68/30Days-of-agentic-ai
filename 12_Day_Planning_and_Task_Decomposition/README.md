[<< Day 11](../11_Day_ReAct_and_Chain_of_Thought_Reasoning/README.md) | [Home](../README.md) | [Day 13 >>](../13_Day_Reflection_and_Self_Critique_Loops/README.md)

# Day 12: Planning and Task Decomposition

Today we build a **Plan-and-Execute** agent that turns a broad goal into an explicit, typed step plan with dependencies—and revises that plan mid-flight when a tool observation contradicts its original assumptions. Separating planning from step execution keeps long-horizon tasks organized, auditable, and resilient to surprises.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `plan_and_execute.py`](#walkthrough-plan_and_executepy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 12](#-exercises---day-12)

---

## Core Concept & Why It Matters

ReAct (Day 11) decides **one step at a time**. For 2–4 step lookups, that is fast and flexible. Once a workflow grows to 5–15 interdependent steps (release readiness checks, incident triage, multi-source research, code migrations), pure step-by-step ReAct develops predictable flaws:

- **Goal drift:** After several verbose tool outputs fill the message history, the agent forgets one of the original requirements.
- **No visibility up front:** Users and operators cannot see what the agent intends to do before it starts calling tools.
- **Hard to resume or parallelize:** Because there is no explicit graph of steps, you cannot inspect which subgoals finished, which failed, or which independent steps can run concurrently.

**Plan-and-Execute** fixes this by splitting the agent into three distinct roles:

1. **Planner:** Takes the user's goal and outputs a structured `Plan` containing typed `PlanStep` objects (ID, title, tool name, arguments, and `depends_on` edges).
2. **Executor:** Picks the next `pending` step whose dependencies are all `completed`, invokes the tool, and records its structured result or error.
3. **Replanner:** After each observation, checks whether new facts invalidate the remaining plan. If so, it inserts, modifies, or skips **pending** steps while keeping already `completed` work intact.

### Trade-Offs to Keep in Mind

| Approach | Strengths | Weaknesses |
| :--- | :--- | :--- |
| **Pure ReAct** | Low upfront latency; natural for open-ended exploration | Drifts on long tasks; no explicit progress state |
| **Static Plan-and-Execute** | Predictable, inspectable, easy to parallelize independent steps | Breaks when an early observation uncovers unexpected work |
| **Dynamic Plan-and-Execute (with Replanner)** | Combines structured progress tracking with mid-flight adaptation | Extra LLM calls for replanning checkpoints; needs a `max_replans` cap |

You do not always need an LLM call at every replanning checkpoint. In many production systems, deterministic rules trigger the replanner only when a step returns a warning, an error, or an unexpected field—keeping normal runs cheap and fast.

---

## How It Works Under the Hood

Here is the execution lifecycle in [`plan_and_execute.py`](./plan_and_execute.py):

```text
  Goal
   │
   ▼
┌────────────────────────────┐
│  1. Planner                │
│  Creates Plan v1           │
│  Validates unique IDs &    │
│  checks for DAG cycles     │
└─────────────┬──────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  2. Scheduler: plan.next_ready_step()   │◄──────────────────────┐
│  Finds first pending step whose         │                       │
│  depends_on steps are all completed     │                       │
└─────────────┬───────────────────────────┘                       │
              │ step                                              │
              ▼                                                   │
┌────────────────────────────┐                                    │
│  3. Executor               │                                    │
│  Runs tool(step.tool_args) │                                    │
│  Marks completed / failed  │                                    │
└─────────────┬──────────────┘                                    │
              │ observation                                       │
              ▼                                                   │
┌────────────────────────────┐   yes (if replans < max_replans)   │
│  4. Replanner              ├────────────────────────────────────┤
│  Does observation require  │  Inserts new steps, wires deps,    │
│  changing remaining steps? │  re-validates DAG, increments v+1  │
└─────────────┬──────────────┘                                    │
              │ no                                                │
              ▼                                                   │
       All steps done? ──── no ───────────────────────────────────┘
              │ yes
              ▼
    Final Synthesized Report
```

### Why Typed Plans and DAG Validation Matter

Representing the plan as plain markdown bullet points ("1. Check metadata\n2. Check staging...") is brittle. When the model revises a markdown list, it often renumbers items, drops finished steps, or forgets which step depends on which.

Using explicit dataclasses (`Plan` and `PlanStep`) gives you real software guarantees:

- **`plan.validate()`** runs a 3-color depth-first search (DFS) to reject missing dependency IDs, duplicate IDs, and cycles *before* executing anything—and again right after every replan.
- **Explicit status transitions** (`pending` -> `in_progress` -> `completed` / `failed` / `skipped`) make it straightforward to persist plans to SQLite/Postgres and resume interrupted runs.
- **Preserved work:** The replanner only appends/inserts pending steps and updates dependency edges on pending steps. Completed observations stay untouched.

---

## Walkthrough: `plan_and_execute.py`

[`plan_and_execute.py`](./plan_and_execute.py) models a production release readiness check for `checkout-api v2.14.0`:

1. **Plan v1** schedules four steps:
   - `S1`: `fetch_release_metadata`
   - `S2`: `check_staging_health` (depends on `S1`)
   - `S3`: `check_rollback_artifact` (depends on `S1`)
   - `S4`: `synthesize_report` (depends on `S2`, `S3`)
2. When `S2` runs, `check_staging_health` reports healthy HTTP 5xx and latency—**but** flags an unvalidated pending database migration (`2026_05_add_idempotency_idx.sql`) with `staging_db_lock_warning: true`.
3. The **Replanner** catches that observation and upgrades the plan to **v2**:
   - Keeps `S1` and `S2` marked `COMPLETED`.
   - Inserts `S2b` (`inspect_migration`, depends on `S2`) before `S4`.
   - Adds `S2b` to `S4.depends_on` so the final report cannot run until the migration check finishes.
4. `S2b` discovers `CREATE INDEX` without `CONCURRENTLY` on a 48-million-row `orders` table (`blocking_lock_risk: HIGH`).
5. `S4` synthesizes a grounded **NO-GO (HOLD RELEASE)** decision with the exact SQL fix.

### Running the Script

Run the offline deterministic demo:

```bash
python3 12_Day_Planning_and_Task_Decomposition/plan_and_execute.py --demo
```

Sample output (abbreviated around the replan event):

```text
Mode: demo
========================================================================
Plan v1 — Goal: Assess production release readiness for checkout-api v2.14.0.
  [PENDING    ] S1: Fetch release metadata and CI status -> tool=fetch_release_metadata
  [PENDING    ] S2: Check staging service health and DB state -> tool=check_staging_health (depends on: S1)
  [PENDING    ] S3: Verify rollback artifact for previous stable release -> tool=check_rollback_artifact (depends on: S1)
  [PENDING    ] S4: Compile final Go/No-Go release readiness report -> tool=synthesize_report (depends on: S2, S3)
------------------------------------------------------------------------

>>> Executing S1: Fetch release metadata and CI status (tool=fetch_release_metadata)
...
>>> Executing S2: Check staging service health and DB state (tool=check_staging_health)
    Observation: {"service": "checkout-api", "http_5xx_rate_pct": 0.04, "p95_latency_ms": 142, "pending_migrations": ["2026_05_add_idempotency_idx.sql"], "staging_db_lock_warning": true, "warning_detail": "Migration 2026_05_add_idempotency_idx.sql has not been validated for table-locking DDL."}

!!! REPLAN TRIGGERED (#1): Step S2 reported pending migration '2026_05_add_idempotency_idx.sql' with a DB lock warning. Inserted step S2b before final synthesis.
Plan v2 — Goal: Assess production release readiness for checkout-api v2.14.0.
  [COMPLETED  ] S1: Fetch release metadata and CI status -> tool=fetch_release_metadata
  [COMPLETED  ] S2: Check staging service health and DB state -> tool=check_staging_health (depends on: S1)
  [PENDING    ] S3: Verify rollback artifact for previous stable release -> tool=check_rollback_artifact (depends on: S1)
  [PENDING    ] S2b: Inspect pending migration '2026_05_add_idempotency_idx.sql' for table-lock risk -> tool=inspect_migration (depends on: S2)
  [PENDING    ] S4: Compile final Go/No-Go release readiness report -> tool=synthesize_report (depends on: S2, S3, S2b)
...
========================================================================
RELEASE READINESS DECISION: NO-GO (HOLD RELEASE)
Service / Version: checkout-api v2.14.0 (commit 8f3c21a)
...
Blocking Finding (uncovered via dynamic replanning):
- Migration `2026_05_add_idempotency_idx.sql` targets `orders` (~48,000,000 rows).
- Lock risk: HIGH (`uses_concurrently=False`).
- Required fix before deploy: Rewrite as CREATE INDEX CONCURRENTLY idx_orders_idempotency_key ON orders (idempotency_key);
```

Run with a live OpenAI-compatible model:

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
python3 12_Day_Planning_and_Task_Decomposition/plan_and_execute.py --goal "Assess production release readiness for checkout-api v2.14.0."
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Over-granular plans:** Asking a planner to generate 20 tiny steps ("1. Parse service name, 2. Format request...") bloats context and causes brittle failures. Keep each `PlanStep` mapped to a concrete tool call or milestone.
2. **Discarding completed work during replanning:** If your replanner regenerates the entire plan from scratch on every surprise, the executor re-runs `S1` and `S2` over and over. Patch only `pending` steps and preserve completed results.
3. **Unbounded replanning thrash:** A nervous replanner can keep adding "one more verification step" indefinitely. Always enforce both `max_replans` and `max_iterations`, and reject duplicate `(tool, tool_args)` insertions.
4. **Silent dependency deadlocks:** If a replanner inserts a step depending on a misspelled ID or creates a cycle (`S2b -> S4 -> S2b`), `next_ready_step()` returns `None` while steps are still pending. Always call `plan.validate()` immediately after every plan mutation and roll back invalid mutations.
5. **Passing stale arguments to downstream steps:** When `S2` discovers the exact filename `2026_05_add_idempotency_idx.sql`, the step inserted by the replanner (`S2b`) needs that concrete value in `tool_args`. Avoid hardcoding guessed arguments at initial plan time when earlier steps are meant to discover them.

---

## 💻 Exercises - Day 12

### Exercise: Level 1

1. Add a unit check (using `assert`) that constructs a `Plan` with a circular dependency (`S1 -> S2 -> S1`) and verifies `plan.validate()` raises a `ValueError`.
2. Modify `check_staging_health` temporarily to return `"pending_migrations": []` and `"staging_db_lock_warning": False`. Re-run with `--demo` and confirm no replan triggers and the final decision becomes `GO`.
3. Add a helper method `plan.to_json()` that serializes the full `Plan` (including step statuses, results, and `revision_notes`) to formatted JSON so an operator could inspect or save it to disk.

### Exercise: Level 2

1. Notice that after `S1` completes, both `S2` and `S3` have all dependencies satisfied. Write a `ready_steps()` method returning *all* ready pending steps, and execute ready batches concurrently with `concurrent.futures.ThreadPoolExecutor`.
2. Add a failure-recovery path: simulate `check_rollback_artifact` throwing a transient `TimeoutError` on its first call, track `attempts` on `PlanStep`, and retry up to 2 times before marking the step `failed` and notifying the replanner.
3. Add a fifth tool `check_oncall_freeze_window` (returns whether a deployment freeze is active) and extend the demo scenario so a second replan inserts a freeze check whenever `staging_db_lock_warning` is false but `pr_count >= 5`.

### Exercise: Level 3

1. Implement **transactional replanning**: deep-copy the `Plan` before calling `backend.maybe_replan()`. If the replanner produces a cycle, references an unknown tool, or proposes a duplicate `(tool, tool_args)` step, discard the bad mutation, log the validation error, and continue executing the last valid plan.
2. Add checkpoint persistence: write the `Plan` state to a JSON file after every completed step and add a `--resume <path>` CLI flag that loads a partially completed plan and resumes from the first `pending` step without re-running `completed` steps.
3. Add an **approval gate** on `PlanStep` (`requires_approval: bool = False`): pause execution before any destructive or high-impact tool step, print the step's planned `tool_args`, and require explicit CLI confirmation (`y/N`) before continuing.

---

[<< Day 11](../11_Day_ReAct_and_Chain_of_Thought_Reasoning/README.md) | [Home](../README.md) | [Day 13 >>](../13_Day_Reflection_and_Self_Critique_Loops/README.md)
