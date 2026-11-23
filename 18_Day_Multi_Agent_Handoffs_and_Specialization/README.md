[<< Day 17](../17_Day_Stateful_Workflows_and_Checkpointing/README.md) | [Home](../README.md) | [Day 19 >>](../19_Day_Supervisor_and_Orchestrator_Patterns/README.md)

# Day 18: Multi-Agent Handoffs & Specialization

Splitting one general-purpose agent into specialized agents sounds appealing, yet doing it too early often slows everything down and doubles your token bill. Today you learn when specialization actually earns its keep, how to enforce strict tool boundaries per role, how to transfer state cleanly using explicit handoff packets, and how to stop specialists from bouncing tickets back and forth in infinite ping-pong loops.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
  - [Single Agent vs. Multiple Specialized Agents](#single-agent-vs-multiple-specialized-agents)
  - [Strict Role Boundaries & Scoped Tools](#strict-role-boundaries--scoped-tools)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
  - [Peer-to-Peer Handoff Flow](#peer-to-peer-handoff-flow)
  - [Explicit `HandoffPacket` Design](#explicit-handoffpacket-design)
  - [Preventing Ping-Pong Loops & Runaway Hops](#preventing-ping-pong-loops--runaway-hops)
- [Walkthrough of the Python Example](#walkthrough-of-the-python-example)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 18](#-exercises---day-18)

---

## Core Concept & Why It Matters

### Single Agent vs. Multiple Specialized Agents

Start with **one well-prompted agent and clean tools** until you hit a measurable failure mode. Adding agents introduces extra LLM round-trips, handoff latency, and state-transfer bugs.

Splitting into specialized agents pays off when you see at least one of these real constraints:
- **Tool overload**: Once a single agent carries 15–30 overlapping tools across unrelated domains, tool-selection accuracy drops and prompt token cost rises on every turn.
- **Conflicting instructions**: A billing agent needs conservative, compliance-heavy refund rules, while a technical support agent needs exploratory log-debugging instructions. Packing both policies into one giant system prompt creates contradictions.
- **Least-privilege isolation**: A front-door triage agent should never even have `issue_refund` or `rotate_api_key` in its tool registry. Separating agents lets the runtime enforce strict per-role permissions.
- **Different model tiers per role**: Triage can run on a small, fast model while complex root-cause diagnosis uses a stronger reasoning model.

### Strict Role Boundaries & Scoped Tools

Never rely on prompt text alone ("please don't call `issue_refund`") to enforce boundaries. Enforce tool scoping in **two places**:
1. **Schema exposure**: Only send `active_agent.allowed_tools` (plus `handoff_to_agent`) in the model's `tools` parameter.
2. **Runtime authorization**: Even if a model hallucinates a tool name belonging to another specialist, the runtime checks `if tc.name not in agent.allowed_tools:` and returns a structured `permission_denied` error instead of executing it.

| Specialist | Allowed Tools | Allowed Handoff Targets | Cannot Touch |
| :--- | :--- | :--- | :--- |
| `triage` | `lookup_account` | `tech_support`, `billing` | Error logs, API keys, refunds |
| `tech_support` | `check_service_status`, `get_error_logs`, `rotate_api_key` | `billing` | Invoices, refunds |
| `billing` | `get_invoices`, `issue_refund` | `tech_support` | Service incidents, error logs, API keys |

---

## How It Works Under the Hood

### Peer-to-Peer Handoff Flow

In a handoff architecture, **one specialist owns the conversation at a time** and explicitly transfers control when the ticket crosses its boundary:

```text
  User Query ("504 errors + duplicate $199 charge on cust-42")
       |
       v
  +-----------+    lookup_account("cust-42")
  |  triage   | ----------------------------------------+
  +-----------+                                         |
       | HandoffPacket (account verified, needs logs)   |
       v                                                |
  +--------------+  check_service_status + get_error_logs
  | tech_support | -------------------------------------+
  +--------------+                                      |
       | HandoffPacket (INC-408 confirmed inv-9002 dup) |
       v                                                |
  +-----------+    get_invoices + issue_refund("inv-9002")
  |  billing  | ----------------------------------------+
  +-----------+
       |
       v
  Final Customer Response (direct from billing specialist)
```

Notice what happens to the message context at each boundary: the receiving specialist gets **its own system prompt**, the **original user request**, and the **structured `HandoffPacket`**. It does *not* inherit the previous specialist's system prompt or raw tool-call noise. That keeps token usage tight and prevents persona bleed.

### Explicit `HandoffPacket` Design

Passing a vague string like `"Billing please take a look"` forces the receiving specialist to repeat work or guess what was verified. Require an explicit structured schema on `handoff_to_agent`:

```python
@dataclass
class HandoffPacket:
    from_agent: str
    to_agent: str
    reason: str
    summary: str
    customer_id: str
    verified_facts: list[str]
    requested_action: str
```

Separating `verified_facts` (observations backed by tool calls) from `requested_action` makes handoffs auditable and dramatically reduces redundant tool calls.

### Preventing Ping-Pong Loops & Runaway Hops

When `tech_support` can hand off to `billing` and `billing` can hand off to `tech_support`, an ambiguous ticket can easily oscillate forever (`tech_support -> billing -> tech_support -> billing`).

`HandoffRuntime` prevents runaway loops with three deterministic checks:
1. **Allowed target graph**: Only transitions declared in `agent.allowed_handoff_targets` are accepted.
2. **Visited-agent cycle guard**: If `target in visited_path`, the runtime blocks the cycle immediately and escalates to a human queue instead of letting the two agents argue.
3. **Hard `max_handoffs` budget**: Caps total transfers per request (default `3`), guaranteeing bounded latency and cost even in larger graphs.

---

## Walkthrough of the Python Example

[multi_agent_handoffs.py](./multi_agent_handoffs.py) runs two complete scenarios in `--demo` mode (no API key needed) and also works against any OpenAI-compatible API (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_NAME`).

Run the offline demo:

```bash
python3 18_Day_Multi_Agent_Handoffs_and_Specialization/multi_agent_handoffs.py --demo
```

Sample output:

```text
--- Scenario 1: Multi-Hop Handoff (triage -> tech_support -> billing) ---

>>> Active Specialist: [triage] (allowed tools: ['lookup_account'])
  [TOOL ] triage.lookup_account({"customer_id": "cust-42"})
  [HANDOFF #1] triage -> tech_support | reason='Technical 504 errors need verification before billing refund.'

>>> Active Specialist: [tech_support] (allowed tools: ['check_service_status', 'get_error_logs', 'rotate_api_key'])
  [TOOL ] tech_support.check_service_status({"service_name": "webhooks-api"})
  [TOOL ] tech_support.get_error_logs({"customer_id": "cust-42"})
  [HANDOFF #2] tech_support -> billing | reason='Confirmed duplicate invoice caused by INC-408 504 retry.'

>>> Active Specialist: [billing] (allowed tools: ['get_invoices', 'issue_refund'])
  [TOOL ] billing.get_invoices({"customer_id": "cust-42"})
  [TOOL ] billing.issue_refund({"invoice_id": "inv-9002", "reason": "Duplicate charge triggered by INC-408 504 retry"})

Path: triage -> tech_support -> billing | Escalated: False
Final Agent: billing

--- Scenario 2: Ping-Pong Loop Guard (triage -> tech_support -> billing -> tech_support) ---
  [HANDOFF #1] triage -> tech_support | reason='Possible API billing mismatch'
  [HANDOFF #2] tech_support -> billing | reason='Looks like a billing question'
  [LOOP GUARD] Blocked ping-pong handoff billing -> tech_support (already visited: ['triage', 'tech_support', 'billing']). Escalating ticket to human support queue.
```

Run against a live model:

```bash
OPENAI_API_KEY=your-key python3 18_Day_Multi_Agent_Handoffs_and_Specialization/multi_agent_handoffs.py \
  --query "Customer cust-42 here. Please check my invoices and refund inv-9002 if it was a duplicate."
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Passing the entire unfiltered conversation across handoffs.** Dumping every previous system message and raw log blob into the next agent causes prompt confusion (the billing agent starts acting like triage) and inflates token costs linearly with each hop. Pass the user's request plus a concise `HandoffPacket`.
2. **Allowing handoffs without verified evidence.** In Scenario 2 above, notice `tech_support` handed off with `verified_facts=[]` without running a single diagnostic tool. In real systems, reject empty handoffs or require at least one domain observation before handing off when facts are missing.
3. **Over-splitting trivial workflows.** If `billing` and `tech_support` must talk back and forth three times on 80% of tickets, they are tightly coupled and should either be merged into one agent or coordinated by a central supervisor (Day 19).
4. **Silent handoff failures.** If a specialist calls `handoff_to_agent` alongside another tool call in the same parallel batch, decide explicitly whether the side-effecting tool runs *before* the transfer or is rejected so you do not drop tool executions mid-handoff.
5. **Forgetting a human escalation exit.** Blocking a ping-pong loop is only half the fix—you must return a clean escalation payload containing the `visited_path` and `handoffs` history so a human support engineer sees what already ran.

---

## 💻 Exercises - Day 18

### Exercise: Level 1

1. Add a fourth specialist `security` to `AGENTS` with strictly scoped tool `rotate_api_key` (moving `rotate_api_key` out of `tech_support`), and update `triage` and `tech_support` so compromised-key reports route cleanly to `security`.
2. Write a short test scenario in `DemoScenarioLLM` where `triage` attempts to call `issue_refund` directly and verify that `HandoffRuntime` prints `[DENY ]` and never modifies `invoices_db`.
3. Extend `HandoffRunResult` to print a compact handoff audit table at the end of every run showing `from_agent`, `to_agent`, `reason`, and count of `verified_facts`.

### Exercise: Level 2

1. Refine the ping-pong guard: instead of forbidding *any* revisit to an agent, allow at most one revisit *only if* the handing-off agent executed at least one domain tool and added at least one new entry to `verified_facts`. Otherwise block and escalate.
2. Add a `HandoffPacket` validator that rejects handoffs where `reason` is shorter than 10 characters or `customer_id` is empty, returning a structured tool error so the active agent fills out the missing fields before transferring.
3. Configure per-agent model routing: add a `model: str` field to `AgentSpec` so `triage` uses a cheaper/faster model (e.g., `gpt-4o-mini`) while `tech_support` uses a stronger model when running in live mode.

### Exercise: Level 3

1. Persist every `HandoffPacket` and specialist step into a SQLite table (combining Day 17's checkpointing idea) so a multi-hop ticket that crashes inside `billing` resumes directly at `billing` with `incoming_packet` intact instead of re-running `triage` and `tech_support`.
2. Support **tool-call ordering safety**: if an agent returns both a domain tool call (`issue_refund`) and `handoff_to_agent` in the same assistant turn, execute all allowed domain tools first and append their results to the handoff packet before switching agents—or explicitly reject mixed turns with an error message.
3. Build a 5-case evaluation harness (testing pure billing, pure tech support, multi-hop incident+refund, unauthorized tool call attempt, and ping-pong loop) and assert expected `visited_path`, `escalated` status, and final invoice state for each case.

---

[<< Day 17](../17_Day_Stateful_Workflows_and_Checkpointing/README.md) | [Home](../README.md) | [Day 19 >>](../19_Day_Supervisor_and_Orchestrator_Patterns/README.md)
