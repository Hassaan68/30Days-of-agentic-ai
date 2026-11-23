[<< Day 21](../21_Day_Input_and_Output_Guardrails/README.md) | [Home](../README.md) | [Day 23 >>](../23_Day_Human_in_the_Loop_Approvals/README.md)

# Day 22: Tool Execution Policies and Least Privilege

Prompt instructions like *"never issue refunds above $50"* or *"only read files inside `/workspace`"* are suggestions, not security controls. Today we build a deterministic policy engine and guarded tool executor that evaluate every tool call in plain Python code before any side effect can happen.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
  - [The Three-Way Decision Model: `AUTO`, `CONFIRM`, `DENY`](#the-three-way-decision-model-auto-confirm-deny)
  - [Static vs. Dynamic Argument-Aware Policies](#static-vs-dynamic-argument-aware-policies)
  - [Least-Privilege Capability Scopes](#least-privilege-capability-scopes)
  - [Sliding-Window Rate Limits](#sliding-window-rate-limits)
  - [Separating Policy Evaluation from Tool Execution](#separating-policy-evaluation-from-tool-execution)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Code Walkthrough: `policy_engine.py`](#code-walkthrough-policy_enginepy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 22](#-exercises---day-22)

---

## Core Concept & Why It Matters

In Day 21 we saw that prompt injection cannot be stopped 100% of the time at the text layer. That means you have to design your tool layer assuming the LLM itself may occasionally propose a dangerous or out-of-bounds tool call—either because of an indirect injection hidden in a fetched document, or simply because the model hallucinated bad arguments.

Treat the LLM as an **untrusted caller**. It can *request* a tool invocation, and deterministic code decides whether that request is permitted under the session's least-privilege scope.

### The Three-Way Decision Model: `AUTO`, `CONFIRM`, `DENY`

Binary allow/deny rules are rarely enough in real operations:

- **`AUTO`**: Low-risk, bounded actions that can run without slowing the user down (for example, reading an order record or issuing a small `$25` refund).
- **`CONFIRM`**: Medium-to-high-impact actions that are legitimate in some contexts but need a human to sign off first (for example, issuing a `$180` refund, emailing an external customer, or exporting 500 database rows). We build the full approval gate for `CONFIRM` decisions on Day 23.
- **`DENY`**: Hard boundaries the agent is never allowed to cross (unknown tools, `drop_table`, refunds over `$500`, `../` path traversal, or SSRF to `169.254.169.254`).

Always pair this with a **default-deny posture**: if a tool name is not explicitly registered in the policy engine, the default decision is `DENY`.

### Static vs. Dynamic Argument-Aware Policies

- **Static policies** decide solely on tool name (`get_order -> AUTO`, `drop_table -> DENY`). They are easy to audit and cover read-only lookups and banned operations.
- **Dynamic policies** inspect the concrete arguments against the session's `CapabilityScope`:
  - `issue_refund`: `amount_usd <= 50` -> `AUTO`; `50 < amount_usd <= 500` -> `CONFIRM`; `amount_usd > 500` (or `<= 0` / `NaN`) -> `DENY`.
  - `query_orders_db`: must be a single `SELECT` without write keywords; `limit <= 100` -> `AUTO`; `101..1000` -> `CONFIRM`; `> 1000` -> `DENY`.
  - `send_email`: internal recipients (`@acme.internal`) -> `AUTO`; external recipients -> `CONFIRM`.

### Least-Privilege Capability Scopes

Give each agent session only the exact resources it needs for its job:

- **Filesystem containment:** Resolve the candidate path with `Path.resolve()` and check `resolved.is_relative_to(allowed_root)`. This defeats relative `../../etc/passwd` traversal and symlinks pointing outside the workspace. Layer a blocked-filename list (`.env`, `id_rsa`, `credentials.json`) on top.
- **Network domain allowlists:** Parse URLs with `urllib.parse.urlparse`, require `https`, reject raw IP literals and `localhost` (preventing SSRF against cloud metadata services like `169.254.169.254`), and match the hostname strictly (`host == allowed or host.endswith("." + allowed)`). A naive substring check like `"docs.python.org" in url` would mistakenly allow `https://docs.python.org.attacker.example/`.

### Sliding-Window Rate Limits

Even an `AUTO`-approved tool becomes dangerous if a stuck loop calls it 500 times in a minute (spamming emails, burning third-party API quotas, or issuing 50 separate `$49` refunds to bypass a `$50` per-call cap).

A sliding-window limiter stores call timestamps per `(session_id, tool_name)` in a `deque`, evicts entries older than `now - window_seconds`, and denies new calls once `len(timestamps) >= max_calls`, returning a clear `retry_after_seconds` hint.

### Separating Policy Evaluation from Tool Execution

Never put policy checks inside a prompt or expect the agent loop to "remember" to check policy before calling a tool. Wrap all raw tool functions inside a **Policy Enforcement Point** (`GuardedToolExecutor`) that holds the only references to the real tool callables. The LLM can only pass `(tool_name, arguments)` to `executor.execute(...)`, which unconditionally calls `policy_engine.evaluate(...)` first.

---

## How It Works Under the Hood

```text
  LLM Tool Call Proposal: (session_id, tool_name, arguments)
                        │
                        ▼
  ┌───────────────────────────────────────────────────────────┐
  │ Policy Enforcement Point: GuardedToolExecutor.execute()   │
  │                                                           │
  │   1. Call PolicyEngine.evaluate(session, tool, args)      │
  │        ├─ Unknown tool? ─────────────────────► DENY       │
  │        ├─ Dynamic / static rule check ───────► AUTO /     │
  │        │   (dollar cap, path resolve,          CONFIRM /  │
  │        │    HTTPS host allowlist, SQL check)   DENY       │
  │        └─ Sliding-window rate limit check ───► DENY if    │
  │                                                exceeded   │
  │                                                           │
  │   2. Branch on PolicyDecision:                            │
  │        ├─ DENY    ──► return PolicyResult (not executed)  │
  │        ├─ CONFIRM ──► pause for HITL approval (Day 23)    │
  │        └─ AUTO    ──► invoke private tool callable        │
  └───────────────────────────────────────────────────────────┘
```

Notice two subtle design choices in the flow above:
1. **Denied calls do not consume rate-limit budget.** Otherwise an attacker or buggy retry with bad arguments could lock out legitimate calls. (If you want to throttle repeated denials, track a separate "abuse/denial" counter.)
2. **Injectable clock in `SlidingWindowRateLimiter`:** Passing `clock: Callable[[], float] = time.monotonic` keeps production timing monotonic (immune to wall-clock jumps) while letting tests advance simulated time instantaneously without `time.sleep()`.

---

## Code Walkthrough: `policy_engine.py`

Open [`policy_engine.py`](./policy_engine.py). It uses only the Python standard library and runs directly without any external API keys.

Key classes and functions:
- `PolicyDecision` & `PolicyResult`: structured `AUTO` / `CONFIRM` / `DENY` decisions with `rule_name`, human-readable `reason`, and optional `retry_after_seconds`.
- `CapabilityScope`: immutable dataclass bundling workspace root, allowed HTTPS domains, blocked filenames, refund caps, and DB row limits for a session.
- `SlidingWindowRateLimiter`: per-`(session_id, tool_name)` sliding window backed by `collections.deque`.
- `PolicyEngine`: registers static and dynamic evaluators, enforces default-deny, and applies rate limits.
- Concrete dynamic evaluators: `evaluate_refund_policy`, `evaluate_file_read_policy`, `evaluate_http_fetch_policy`, `evaluate_db_query_policy`, `evaluate_send_email_policy`.
- `GuardedToolExecutor`: private tool registry that only invokes a tool when `PolicyEngine.evaluate()` returns `AUTO`.

### Running the Script

Run from the repository root:

```bash
python3 22_Day_Tool_Execution_Policies_and_Least_Privilege/policy_engine.py
```

Sample output (abbreviated):

```text
=== Day 22: Tool Execution Policy Engine Demo ===

1. Unknown tool (default-deny)
   Decision: DENY    | Executed: False | Rule: default_deny
   Reason:   Tool 'shell_exec' is not allowed by policy (default-deny)

...

4. Small refund ($25 <= $50)
   Decision: AUTO    | Executed: True  | Rule: refund:auto_tier
   Reason:   Refund $25.00 is within auto-approval limit ($50.00)
   Output:   {'order_id': 'ORD-1001', 'refunded_usd': 25.0}

5. Medium refund ($180 in $50-$500)
   Decision: CONFIRM | Executed: False | Rule: refund:confirm_tier
   Reason:   Refund $180.00 exceeds auto limit ($50.00) and requires human approval (up to $500.00)

6. Large refund ($1,200 > $500)
   Decision: DENY    | Executed: False | Rule: refund:hard_cap
   Reason:   Refund $1200.00 exceeds maximum permitted cap ($500.00)

...

9. File path traversal (../)
   Decision: DENY    | Executed: False | Rule: file:path_traversal_blocked
   Reason:   Path '../../etc/passwd' escapes allowed workspace '/var/folders/...'

12. Spoofed subdomain attack
   Decision: DENY    | Executed: False | Rule: http:domain_denied
   Reason:   Domain 'docs.python.org.attacker.example' is not in allowlist ('docs.python.org', 'status.acme.example')

13. Cloud metadata SSRF attempt
   Decision: DENY    | Executed: False | Rule: http:scheme_denied
   Reason:   Only HTTPS URLs are permitted (got scheme 'http')

--- 17. Sliding-Window Rate Limit Check (send_email: 3 calls / 60s) ---
   Call 2: AUTO (executed=True)
   Call 3: AUTO (executed=True)
   Call 4: DENY | Rate limit exceeded for 'send_email' (3 calls per 60s) | retry_after=60.0s
   Call 5 (+61s): AUTO (executed=True)

All 17 policy and rate-limit checks passed.
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **String-prefix path checks instead of `is_relative_to()`:** Writing `str(resolved).startswith(str(root))` has a classic bug: `/tmp/workspace_secret/keys.txt` starts with `/tmp/workspace` and passes! Always use `Path.resolve()` plus `Path.is_relative_to(root)`.
2. **Allowing `bool` where `int`/`float` is expected in Python:** In Python, `isinstance(True, int)` is `True` (so `True` looks like `1`). Always explicitly guard `isinstance(val, bool)` when validating numeric thresholds like dollar amounts or row limits.
3. **Splitting attacks ("smurfing") across multiple small calls:** A per-call `$50` auto-refund cap does not stop an agent from issuing ten `$45` refunds in one session (`$450` total). Pair per-call thresholds with cumulative session budgets (see Level 2 Exercise 1).
4. **DNS rebinding and HTTP redirects in `http_fetch`:** Checking the initial URL hostname stops naive SSRF, but if `https://allowed.example/redirect` returns a `302` to `http://169.254.169.254/`, your HTTP client will follow it unless you disable auto-redirects or re-run the URL policy on every redirect hop.
5. **Passing unvalidated raw arguments instead of canonicalized values:** If `evaluate_file_read_policy` resolves a path safely, make sure the tool uses the exact resolved path (or re-checks `O_NOFOLLOW`) so a concurrent symlink swap cannot redirect the open call.

---

## 💻 Exercises - Day 22

### Exercise: Level 1

1. Run `policy_engine.py` and trace why Case 16 (`send_email` to an external address) counted toward the sliding-window rate limit in Case 17. Discuss whether `CONFIRM` decisions should consume rate-limit quota or only count once approved.
2. Add a test case passing `amount_usd=float("nan")` and `amount_usd=True` to `issue_refund` and verify both return `PolicyDecision.DENY`.
3. Add a symlink test inside `run_demo()` where a symlink inside `workspace` points to `/etc/hosts` outside `workspace`, and verify `evaluate_file_read_policy` blocks it via `Path.resolve()`.
4. Add a new static tool `get_shipping_status` with decision `AUTO` and a rate limit of `2` calls per `10` seconds, and verify the 3rd call is denied.

### Exercise: Level 2

1. Implement a **cumulative session budget** tracker for `issue_refund`: even if each individual refund is `<= $50`, require `CONFIRM` once the session's cumulative refunds exceed `$100` in 24 hours, and `DENY` once they exceed `$500`.
2. Add a `write_workspace_file` dynamic policy that:
   - allows writes only under a `drafts/` subdirectory inside `workspace`,
   - enforces a maximum content size (e.g., 16 KB),
   - denies overwriting existing files unless `overwrite=True` (in which case it returns `CONFIRM`).
3. Support **policy-as-data**: load static rules, allowed domains, thresholds, and rate limits from a JSON config file into `CapabilityScope` and `PolicyEngine` instead of hardcoding them in Python.
4. Add a redirect-safe check helper `evaluate_redirect_chain(urls, scope)` that verifies every hop in an HTTP redirect chain passes `evaluate_http_fetch_policy`.

### Exercise: Level 3

1. Add **role-based capability profiles** (`support_tier1`, `support_tier2`, `readonly_analyst`) so `PolicyEngine.evaluate()` accepts a `SessionContext(session_id, user_role, tenant_id)` and applies different thresholds and allowed tools per role. Include multi-tenant isolation checks (e.g., `order_id` or file paths must belong to `tenant_id`).
2. Connect `GuardedToolExecutor` to a real agent loop (using Ollama or OpenAI, or a mock model) where a `DENY` result is returned as a structured JSON tool message so the model can recover gracefully without crashing, and a `CONFIRM` result pauses the loop cleanly and returns a pending-action payload.
3. Build a **dry-run policy simulator** CLI that reads a JSONL file of recorded historical tool calls and outputs a diff report showing how many historical calls would change between `AUTO`, `CONFIRM`, and `DENY` under a proposed new `CapabilityScope` before deploying the policy change to production.

---

[<< Day 21](../21_Day_Input_and_Output_Guardrails/README.md) | [Home](../README.md) | [Day 23 >>](../23_Day_Human_in_the_Loop_Approvals/README.md)
