[<< Day 18](../18_Day_Multi_Agent_Handoffs_and_Specialization/README.md) | [Home](../README.md) | [Day 20 >>](../20_Day_Code_Execution_and_Sandboxing/README.md)

# Day 19: Supervisor & Orchestrator Patterns

Yesterday you built peer-to-peer handoffs where specialists pass a conversation sideways until one of them answers the user. That works well for support routing, but breaks down on complex deliverables where research, quantitative analysis, and final writing must be combined and checked against clear standards. Today you build a central **Supervisor (Orchestrator)** that decomposes a goal into explicit subtasks, invokes specialized workers with bounded context, validates every output against acceptance criteria, retries failures with targeted feedback, and synthesizes the final response.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
  - [Peer Handoffs (Day 18) vs. Central Supervisor (Day 19)](#peer-handoffs-day-18-vs-central-supervisor-day-19)
  - [Why Bounded Worker Context Improves Quality & Cost](#why-bounded-worker-context-improves-quality--cost)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
  - [Orchestration, Validation & Retry Loop](#orchestration-validation--retry-loop)
  - [Acceptance Criteria & Deterministic Verification First](#acceptance-criteria--deterministic-verification-first)
- [Walkthrough of the Python Example](#walkthrough-of-the-python-example)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 19](#-exercises---day-19)

---

## Core Concept & Why It Matters

### Peer Handoffs (Day 18) vs. Central Supervisor (Day 19)

Both patterns use multiple specialized agents, but control flows differently:

| Dimension | Peer-to-Peer Handoffs (Day 18) | Central Supervisor / Orchestrator (Day 19) |
| :--- | :--- | :--- |
| **Who owns the goal?** | Whoever currently holds the turn | The Supervisor from start to finish |
| **Who talks to the user?** | Whichever specialist finishes | Always the Supervisor (after synthesis) |
| **Best suited for** | Customer routing, domain triage, conversational ownership shifts | Multi-part reports, code+test+review pipelines, research-to-brief workflows |
| **Quality control** | Trusting the final specialist | Explicit verification gate after every subtask |
| **Main trade-off** | Risk of ping-pong loops if domains overlap | Higher orchestration overhead and extra latency/tokens per subtask |

### Why Bounded Worker Context Improves Quality & Cost

A common mistake in multi-agent code is passing the supervisor's entire transcript to every worker. That creates two problems:
1. **Token bloat**: If the researcher pulled 6,000 tokens of raw documents, passing all of them plus the full chat history into the writer wastes tokens and distracts the model.
2. **Scope creep**: Giving the writer access to search tools tempts it to invent new queries instead of synthesizing the already-verified research and math.

In today's pattern, each worker receives **only the exact inputs declared in its `depends_on` list** and **only the tools required for its role**:
- `researcher`: gets the search tool (`search_internal_docs`) and returns cited facts.
- `analyst`: gets the verified research artifact + deterministic calculator (`compute_cost_and_slo`), returning verified numbers.
- `writer`: gets **no external tools**—only the verified research and analysis artifacts—and drafts the memo.

---

## How It Works Under the Hood

### Orchestration, Validation & Retry Loop

```text
                         User Goal
                             |
                             v
                 +-----------------------+
                 | SupervisorOrchestrator|
                 |  1. Build subtask DAG |
                 +-----------------------+
                     /       |        \
          +---------+        |         +-----------+
          v                  v                     v
   [task_research]    [task_analysis]        [task_writing]
   worker: researcher worker: analyst        worker: writer
   deps: none         deps: research         deps: research, analysis
          |                  |                     |
          v                  v                     v
   +-------------------------------------------------------+
   | validate_artifact(subtask, artifact)                  |
   |   PASS -> store verified artifact & advance           |
   |   FAIL -> attach failure reasons -> retry (up to N)   |
   +-------------------------------------------------------+
                             |
                             v
               Synthesize Final Report + Scorecard
```

### Acceptance Criteria & Deterministic Verification First

Every `Subtask` declares concrete `acceptance_criteria` before execution begins. When a worker returns a `WorkerArtifact`, the supervisor validates it **before** passing it downstream:
- **Deterministic checks first**: Verify required document citations exist in the real corpus, required calculated figures appear, required sections are present, and word-count budgets are respected. These checks run in microseconds at zero token cost.
- **Targeted feedback on retry**: When `analyst` attempt 1 omits the annual cost delta (`$37,800/year`) and percentage increase (`+75.0%`), the supervisor does not just say "try again." It sends the exact list of failed assertions back into attempt 2 so the worker knows precisely what to fix.
- **Bounded retries**: Each subtask carries `max_retries=2`. If a worker still fails after 3 total attempts, the supervisor halts with an explicit error instead of burning API credits indefinitely.

---

## Walkthrough of the Python Example

[supervisor_orchestrator.py](./supervisor_orchestrator.py) implements the full supervisor, three bounded workers (`researcher`, `analyst`, `writer`), deterministic validation gates, targeted retries, and final report synthesis.

Run the offline demo (in `--demo` mode, `analyst` intentionally submits an incomplete first draft so you can watch the supervisor reject it, send targeted feedback, and accept attempt 2):

```bash
python3 19_Day_Supervisor_and_Orchestrator_Patterns/supervisor_orchestrator.py --demo
```

Sample output:

```text
[SUPERVISOR] Created 3-step delegation plan:
  - task_research -> worker='researcher' (depends_on: none)
  - task_analysis -> worker='analyst' (depends_on: task_research)
  - task_writing -> worker='writer' (depends_on: task_research, task_analysis)

[SUPERVISOR] Delegating 'task_research' to worker 'researcher'...
  [PASS] researcher attempt 1/3 passed validation (citations=['doc-arch-01', 'doc-cost-02', 'doc-slo-03']).

[SUPERVISOR] Delegating 'task_analysis' to worker 'analyst'...
  [FAIL] analyst attempt 1/3 rejected:
         * Must report exact annual cost delta ($37,800/year).
         * Must report percentage cost increase (+75.0%).
         * Must report RTO percentage reduction (88.9%).
  [RETRY] Sending targeted feedback back to 'analyst'...
  [PASS] analyst attempt 2/3 passed validation (citations=['doc-arch-01', 'doc-cost-02', 'doc-slo-03']).

[SUPERVISOR] Delegating 'task_writing' to worker 'writer'...
  [PASS] writer attempt 1/3 passed validation (citations=['doc-arch-01', 'doc-cost-02', 'doc-slo-03']).

[SUPERVISOR] All subtasks verified. Synthesizing final report...
...
Supervisor Verification Scorecard:
  - task_research (researcher): PASSED on attempt 1
  - task_analysis (analyst): PASSED on attempt 2
  - task_writing (writer): PASSED on attempt 1
```

To run all three workers against a real OpenAI-compatible API (OpenAI or local Ollama):

```bash
OPENAI_API_KEY=your-key python3 19_Day_Supervisor_and_Orchestrator_Patterns/supervisor_orchestrator.py
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Sequential bottlenecks when subtasks are independent.** In our 3-step pipeline, `analysis` depends on `research`, so sequential execution makes sense. If your plan has three independent research tasks (pricing, compliance, benchmarks), running them serially triples wall-clock latency—execute independent DAG nodes concurrently with `asyncio.gather` (Day 10).
2. **Rubber-stamp validation.** Asking the exact same LLM prompt "does this look good?" without concrete criteria lets subtle omissions slip through. Check structure, citations, numbers, and length deterministically first, then reserve LLM-based grading for subjective qualities like clarity or tone.
3. **Unbounded retry loops.** Without `max_retries` and a total run-level token/call cap, a strict validator paired with a confused worker can burn 20 retries on a single subtask. Always bound both per-subtask retries and total supervisor calls.
4. **Over-engineering simple tasks.** A supervisor with 3 workers and 1 retry makes 4+ LLM calls. If a single well-structured prompt with two tools solves the problem with 95% accuracy, keep the single agent until task complexity genuinely requires delegation and intermediate verification.
5. **Dropping error context on retries.** Retrying a worker with the exact same prompt and no explanation of *why* attempt 1 failed usually produces the same mistake again. Always pass the concrete list of failed criteria back into the retry prompt.

---

## 💻 Exercises - Day 19

### Exercise: Level 1

1. Add a fourth document (`doc-ops-04`) to `DOC_CORPUS` describing quarterly failover drill staffing requirements (2 on-call engineers, 2 hours/quarter), and update `validate_artifact` so the final `writer` memo must mention quarterly failover drills.
2. Inspect what happens when a subtask exhausts `max_retries`: temporarily force `analyst` to always return the incomplete attempt 1 text in `--demo` mode, run the script, and catch the `RuntimeError` cleanly in `main()` so it prints a partial failure report instead of a raw traceback.
3. Add elapsed wall-clock timing (`time.perf_counter()`) per subtask attempt and include each subtask's total milliseconds in the `Supervisor Verification Scorecard`.

### Exercise: Level 2

1. Add a fourth worker `reviewer` (after `writer`) that checks for unsupported numeric claims by extracting every dollar figure and percentage in the writer's memo and asserting each one appears in either `researcher`'s documents or `analyst`'s `computed_metrics`.
2. Turn the linear plan into a real DAG runner: add a second independent research subtask `task_compliance` (checking EU data residency constraints in eu-central-1 vs eu-west-1) that has `depends_on=[]`, and run `task_research` and `task_compliance` concurrently using `concurrent.futures.ThreadPoolExecutor` or `asyncio`.
3. Add dynamic LLM-generated planning in live mode: let the supervisor generate the `list[Subtask]` plan as strict JSON (validating worker names and checking for dependency cycles via topological sort) before dispatching workers.

### Exercise: Level 3

1. Combine Day 17 and Day 19: persist every verified `WorkerArtifact` into SQLite after it passes validation so if `task_writing` fails or the process crashes after `task_analysis`, re-running with the same `run_id` resumes straight at `task_writing`.
2. Add a hybrid verification gate for live mode: run deterministic checks first (fast, free), and only if deterministic checks pass, call a lightweight LLM judge returning structured JSON `{"passed": bool, "issues": list[str]}` to verify tone and logical coherence.
3. Implement a per-run budget controller inside `SupervisorOrchestrator` (`max_total_worker_calls=6`, `max_wall_clock_seconds=60`) that gracefully degrades to a partial executive report marked `[INCOMPLETE]` if the budget is exhausted mid-plan.

---

[<< Day 18](../18_Day_Multi_Agent_Handoffs_and_Specialization/README.md) | [Home](../README.md) | [Day 20 >>](../20_Day_Code_Execution_and_Sandboxing/README.md)
