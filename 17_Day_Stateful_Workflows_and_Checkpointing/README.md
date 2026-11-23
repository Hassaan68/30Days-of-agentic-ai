[<< Day 16](../16_Day_Building_Custom_MCP_Servers/README.md) | [Home](../README.md) | [Day 18 >>](../18_Day_Multi_Agent_Handoffs_and_Specialization/README.md)

# Day 17: Stateful Workflows & Checkpointing

In-memory agent loops work fine while a request lasts three seconds on your laptop. Once a workflow runs five LLM calls, fetches external data, or pauses waiting for a human to approve a change, an unhandled exception or container restart wipes out every intermediate result. Today you build a SQLite-backed checkpointer that saves immutable, versioned state snapshots after every step—enabling crash recovery, clean pause/resume, and time-travel replay from any earlier checkpoint.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
  - [What Needs to Be Persisted](#what-needs-to-be-persisted)
  - [Trade-offs: Checkpointing Overhead vs. Recovery Value](#trade-offs-checkpointing-overhead-vs-recovery-value)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
  - [Crash Recovery & Resume Flow](#crash-recovery--resume-flow)
  - [Append-Only Snapshots & Time-Travel Forking](#append-only-snapshots--time-travel-forking)
  - [Schema Versioning & State Migrations](#schema-versioning--state-migrations)
- [Walkthrough of the Python Example](#walkthrough-of-the-python-example)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 17](#-exercises---day-17)

---

## Core Concept & Why It Matters

Every multi-step agent pipeline eventually hits one of four real-world interruptions:
- **Transient infrastructure failures**: rate limits, network resets, worker OOM kills, or rolling deployments mid-run.
- **Human-in-the-loop pauses**: pausing before an irreversible write until an engineer approves it two hours later (which you will expand on Day 23).
- **Costly upstream steps**: if step 1 spent $0.18 scraping documents and building a research summary, restarting from scratch after step 3 times out wastes money and time.
- **Post-incident debugging**: when a workflow outputs a bad draft on step 4, you want to inspect the exact state produced at step 1 and step 2—and test a prompt fix from step 2 without re-running step 1.

Checkpointing solves all four by treating workflow execution as a sequence of **durable state transitions** keyed by a `run_id` (sometimes called a `thread_id`).

### What Needs to Be Persisted

A minimal durable checkpointer stores two tables:
1. **`workflow_runs`**: tracks top-level run metadata (`run_id`, `workflow_name`, `status` such as `running`/`paused`/`failed`/`completed`, `last_completed_step`, `error_message`, and optional fork lineage `parent_run_id` + `forked_from_step`).
2. **`checkpoints`**: an **append-only** log of `(run_id, step_index, step_name, state_version, state_hash, state_json, created_at)` written inside a database transaction immediately after each step finishes.

### Trade-offs: Checkpointing Overhead vs. Recovery Value

Writing state to SQLite (or Postgres in multi-worker deployments) after every step adds a few milliseconds of I/O and requires keeping your workflow state strictly JSON-serializable. You cannot stuff live HTTP client handles, open file descriptors, or Python lambdas inside `WorkflowState`.

In exchange, you gain deterministic resume semantics, full run inspection, and zero lost progress on crashes. For single-shot model calls, checkpointing is unnecessary; for any pipeline with 3+ steps, expensive tool calls, or pause/resume requirements, it is foundational.

---

## How It Works Under the Hood

### Crash Recovery & Resume Flow

```text
  Worker A (run_id="run-001")
  ===========================
  Step 0 (init)         --> [SAVE checkpoint step=0]
  Step 1 (gather_facts) --> [SAVE checkpoint step=1]
  Step 2 (draft_brief)  --> [CRASH! SimulatedCrashError]
                            (run status marked 'failed'; step 1 snapshot stays intact)

  -------- Process dies; fresh Worker B boots with same run_id="run-001" --------

  Worker B (run_id="run-001")
  ===========================
  Load latest checkpoint --> step=1 (gather_facts)
  Step 1 (gather_facts) --> [SKIP] (step.index <= 1)
  Step 2 (draft_brief)  --> [RUN ] --> [SAVE checkpoint step=2]
  Step 3 (verify_claims)--> [RUN ] --> [SAVE checkpoint step=3]
  Step 4 (publish)      --> [RUN ] --> [SAVE checkpoint step=4, status='completed']
```

Notice the critical invariant: **a checkpoint is written only after its step succeeds**. If the worker crashes inside step 2, step 1's snapshot remains the latest committed record, so the restarted worker knows step 1 is complete and step 2 must be attempted again.

### Append-Only Snapshots & Time-Travel Forking

Because we insert a new row per step instead of overwriting a single `current_state` column, every historical snapshot remains intact:

```text
  run-001:        [step 0: init] -> [step 1: gather_facts] -> [step 2] -> [step 3] -> [step 4]
                                             \
                                              \  fork_run(from_step=1, audience="security reviewers")
                                               v
  run-001-fork:                     [step 1: fork:gather_facts] -> [step 2'] -> [step 3'] -> [step 4']
```

Forking into a brand-new `run_id` gives you safe time-travel debugging: you reuse expensive step 1 outputs, change a parameter or prompt, and replay steps 2–4 while keeping the original run untouched for comparison.

### Schema Versioning & State Migrations

Long-running workflows sometimes stay paused across code deployments. If you add a new required field to `WorkflowState` on Tuesday, resuming a run that paused on Monday will fail unless you version your snapshots.

In `checkpoint_workflow.py`, every snapshot stores `schema_version` (`STATE_SCHEMA_VERSION = 1`), and `WorkflowState.from_json()` routes older payloads through `migrate_state_dict()` to backfill defaults before constructing the dataclass.

---

## Walkthrough of the Python Example

[checkpoint_workflow.py](./checkpoint_workflow.py) implements the full SQLite checkpointer (using `PRAGMA journal_mode=WAL` and foreign keys), a 4-step `CheckpointedWorkflow`, crash injection, clean resume, checkpoint inspection, and time-travel forking—all using Python's standard library.

Run the offline demo (uses a clean temporary SQLite file automatically):

```bash
python3 17_Day_Stateful_Workflows_and_Checkpointing/checkpoint_workflow.py --demo
```

Or save the SQLite file locally so you can query it directly afterward:

```bash
python3 17_Day_Stateful_Workflows_and_Checkpointing/checkpoint_workflow.py --db ./day17_checkpoints.db
sqlite3 ./day17_checkpoints.db "SELECT run_id, step_index, step_name, state_hash FROM checkpoints;"
```

Sample output from `--demo`:

```text
Phase 1: Starting fresh workflow instance (will crash on step 2)...
  [RUN ] Step 1 (gather_facts)...
  [SAVE] Checkpoint #2: step=1 (gather_facts) state_hash=d35eba2b0181 status=running
  [RUN ] Step 2 (draft_brief)...
[CRASH] Simulated process crash during step 2 (draft_brief)
        Worker A executed steps before crash: ['gather_facts']

Phase 2: Restarting in a BRAND-NEW workflow instance with the same run_id...
[RESUME] Loaded run='run-checkout-latency-001' from checkpoint step=1 (gather_facts, hash=d35eba2b0181)
  [SKIP] Step 1 (gather_facts) already completed in checkpoint.
  [RUN ] Step 2 (draft_brief)...
  [SAVE] Checkpoint #3: step=2 (draft_brief) state_hash=0a4a4b02a9af status=running
  [RUN ] Step 3 (verify_claims)...
  [SAVE] Checkpoint #4: step=3 (verify_claims) state_hash=f7823f0d5a12 status=running
  [RUN ] Step 4 (publish_report)...
  [SAVE] Checkpoint #5: step=4 (publish_report) state_hash=e52bc5667c0e status=completed
        Worker B executed ONLY remaining steps: ['draft_brief', 'verify_claims', 'publish_report']

--- Checkpoint History for run_id='run-checkout-latency-001' (status=completed, last_step=4) ---
  cp_id=1  | step=0 (init              ) | v1 | hash=8d266f54ed3e | facts=0 | draft_chars=0
  cp_id=2  | step=1 (gather_facts      ) | v1 | hash=d35eba2b0181 | facts=4 | draft_chars=0
  cp_id=3  | step=2 (draft_brief       ) | v1 | hash=0a4a4b02a9af | facts=4 | draft_chars=339
  cp_id=4  | step=3 (verify_claims     ) | v1 | hash=f7823f0d5a12 | facts=4 | draft_chars=339
  cp_id=5  | step=4 (publish_report    ) | v1 | hash=e52bc5667c0e | facts=4 | draft_chars=339

Phase 3: Time-travel replay — forking from step 1 into a new run_id with audience='security & compliance reviewers'...
[RESUME] Loaded run='run-checkout-latency-001-security-fork' from checkpoint step=1 (fork:gather_facts, hash=60338ce7bb00)
  [SKIP] Step 1 (gather_facts) already completed in checkpoint.
  [RUN ] Step 2 (draft_brief)...
  [RUN ] Step 3 (verify_claims)...
  [RUN ] Step 4 (publish_report)...
```

You can also pass `--live` with `OPENAI_API_KEY` (or local Ollama) configured so step 2 generates real model drafts while steps 1, 3, and 4 remain deterministic and checkpointed.

---

## Common Pitfalls & Bugs Developers Run Into

1. **Duplicate side effects on crash boundaries.** Suppose step 4 sends an email or charges a credit card, and the process crashes *after* the external API succeeds but *before* `save_checkpoint()` commits. On resume, step 4 runs again and sends a duplicate email. Fix this by keeping external writes at the end of a step and passing a deterministic idempotency key derived from `(run_id, step_index)` (covered in depth on Day 28).
2. **Storing non-serializable objects in state.** Putting database connections, `datetime` instances, or custom class objects inside `WorkflowState` causes `json.dumps()` to crash right when you try to checkpoint. Keep state strictly to plain primitives, dataclasses, lists, and dicts, and inject clients/connections into step handlers at runtime.
3. **Overwriting checkpoints in place.** Updating a single `state_json` column instead of appending per-step rows destroys your ability to inspect intermediate states or fork from step 1 when debugging a step 3 regression.
4. **Concurrent workers resuming the same `run_id`.** If two workers pick up the same paused `run_id` at the same time, both execute step 2 and race on the `UNIQUE(run_id, step_index)` constraint. Production runners use a lease/lock column (`locked_by`, `lease_expires_at`) so only one worker owns a run at a time.
5. **Unbounded state growth.** Copying giant 500 KB raw tool outputs into every step snapshot quickly bloats the checkpoint database. Store large artifacts in blob storage (or a separate `artifacts` table) and keep only lightweight artifact IDs and hashes inside `WorkflowState`.

---

## 💻 Exercises - Day 17

### Exercise: Level 1

1. Run `checkpoint_workflow.py --db ./local_test.db` and inspect both `workflow_runs` and `checkpoints` using the `sqlite3` CLI or a short Python snippet. Verify that `parent_run_id` and `forked_from_step` are populated for the forked run.
2. Add a Phase 4 to `run_demo()` that exercises `pause_after_step=2` on a fresh `run_id`, confirms `get_run_info(run_id)["status"] == "paused"`, and then resumes it in a new `CheckpointedWorkflow` instance to completion.
3. Write a helper function `diff_checkpoints(cp_a, cp_b)` that compares two `CheckpointRecord` objects and prints only the fields in `WorkflowState` that changed between them.

### Exercise: Level 2

1. Bump `STATE_SCHEMA_VERSION` from `1` to `2` and add a `tags: list[str]` field to `WorkflowState`. Save a v1 JSON blob directly into SQLite, update `migrate_state_dict()` to migrate v1 -> v2 cleanly, and verify that resuming the v1 run succeeds without errors.
2. Add a `step_durations_ms: dict[str, int]` column or state field that measures wall-clock duration for each executed step (and leaves skipped steps untouched on resume).
3. Add a CLI subcommand mode (`--resume-run <run_id> --db ./runs.db`) so you can run a workflow with `--fail-on-step 3` in one terminal command, exit with non-zero status, and resume the exact same run in a second terminal command.
4. Refactor step 4 (`publish_report`) to accept a deterministic `idempotency_key = f"{run_id}:step-{step.index}"` and simulate a mock publishing API that safely deduplicates repeated calls with the same key.

### Exercise: Level 3

1. Add optimistic locking / worker leases (`locked_by`, `lease_expires_at`) to `workflow_runs` so two concurrent `CheckpointedWorkflow` instances attempting to resume the same `run_id` cannot execute the same step simultaneously.
2. Separate large payload storage from checkpoints: add an `artifacts(artifact_id TEXT PRIMARY KEY, sha256 TEXT, content TEXT)` table, store `draft` and `final_brief` as content-addressed artifact references, and resolve them automatically when loading a checkpoint.
3. Add support for **looping / conditional steps** (where `verify_claims` can send execution back to `draft_brief` up to 2 times if a check fails) by changing the checkpoint primary key from `(run_id, step_index)` to `(run_id, sequence_num)` so every retry attempt is preserved in the timeline.

---

[<< Day 16](../16_Day_Building_Custom_MCP_Servers/README.md) | [Home](../README.md) | [Day 18 >>](../18_Day_Multi_Agent_Handoffs_and_Specialization/README.md)
