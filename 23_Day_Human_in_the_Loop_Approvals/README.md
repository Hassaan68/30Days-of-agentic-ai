[<< Day 22](../22_Day_Tool_Execution_Policies_and_Least_Privilege/README.md) | [Home](../README.md) | [Day 24 >>](../24_Day_Audit_Logs_and_Tamper_Evident_Receipts/README.md)

# Day 23: Human-in-the-Loop Approvals

When Day 22's policy engine returns `CONFIRM`, the agent must pause and get explicit sign-off before running an irreversible action like wiring money, deleting cloud resources, or emailing an external customer. Today we build an approval store and execution gate with expiration, single-use tokens, and SHA-256 payload binding to block Time-of-Check to Time-of-Use (TOCTOU) tampering.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
  - [Why `input("Approve? [y/n]")` Breaks in Production](#why-inputapprove-yn-breaks-in-production)
  - [Clear Human-Readable Approval Requests](#clear-human-readable-approval-requests)
  - [Expiration (TTL) and Single-Use Tokens](#expiration-ttl-and-single-use-tokens)
  - [Defeating TOCTOU Tampering with Canonical Payload Hashes](#defeating-toctou-tampering-with-canonical-payload-hashes)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Code Walkthrough: `hitl_approvals.py`](#code-walkthrough-hitl_approvalspy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 23](#-exercises---day-23)

---

## Core Concept & Why It Matters

Human-in-the-loop (HITL) approvals keep humans accountable for high-impact decisions while letting the agent automate the tedious research and preparation steps leading up to the action.

### Why `input("Approve? [y/n]")` Breaks in Production

In a local CLI prototype, pausing with `input()` works. In a real web app, Slack bot, or background worker, a human might take 30 seconds or 20 minutes to review a request, and the process that proposed the action is often not the same process or HTTP request that executes it later.

Instead of blocking a thread, model approvals as explicit state transitions:

`PENDING` -> `APPROVED` (or `REJECTED` / `EXPIRED`) -> `CONSUMED`

The agent serializes the pending request, returns control to the caller (or pauses its run state), and only resumes tool execution once a verified approval grant arrives.

### Clear Human-Readable Approval Requests

Approval fatigue is a real security risk. If an approval UI just says *"Allow agent to call `transfer_funds`?"*, reviewers start clicking Approve on autopilot.

Every approval request should present:
- **Session / requester identity** and unique `approval_id`.
- **Exact tool name and full structured arguments** (recipient, amount, currency, memo—never hidden behind a vague summary).
- **Plain-language impact summary** ("Pay vendor invoice #882 for $150.00").
- **Expiration timestamp** and remaining TTL.
- **Payload SHA-256 fingerprint** so reviewers and downstream audit logs (Day 24) reference the exact payload.

### Expiration (TTL) and Single-Use Tokens

Two bugs show up quickly in naive approval systems:

1. **Stale approvals:** A manager approves a database migration or stock trade 4 hours later when the underlying state has completely changed. Every approval needs a strict **TTL** (`expires_at`) checked both at approval time and again at execution time.
2. **Replay attacks / duplicate retries:** A network timeout causes the worker to retry `execute_with_approval(approval_id, token, ...)` twice, sending two $150 transfers using the same approval. Every approval token must be **single-use** and atomically marked `CONSUMED` before invoking the underlying tool.

### Defeating TOCTOU Tampering with Canonical Payload Hashes

**Time-of-Check to Time-of-Use (TOCTOU)** is one of the most overlooked vulnerabilities in agent approval flows. Here is how it happens:

1. **Time of Check:** The agent proposes `transfer_funds(to_account="acct_vendor_42", amount_usd=150.00)`. The human inspects those arguments and clicks **Approve**, minting `approval_id=apr_8f2a1c`.
2. **Mutation in between:** Before execution, a prompt injection on a follow-up turn, a buggy retry that mutates a shared dictionary, or a compromised worker swaps the arguments to `transfer_funds(to_account="acct_attacker_99", amount_usd=15000.00)` while passing the already-approved `approval_id` and `token`.
3. **Time of Use:** If the execution gate only checks *"is `apr_8f2a1c` approved?"*, the $15,000 transfer goes through!

The fix is straightforward and cryptographic:

1. Serialize `{"tool_name": tool_name, "arguments": arguments}` into **canonical JSON** (recursively sorted keys, compact separators `(",", ":")`, normalized numbers, no `NaN`/`Infinity`).
2. Compute `payload_hash = sha256(canonical_json)`. Store that digest on the `ApprovalRequest` at creation time (and deep-copy the arguments so external dict mutations cannot alter the record).
3. Right before executing the tool in `ApprovalGate`, recompute `sha256(canonical_json({"tool_name": tool_name, "arguments": arguments}))` over the *exact runtime payload* and compare with `hmac.compare_digest(req.payload_hash, actual_hash)`.

If even a single cent (`150.00` -> `150.01`) or one character of the recipient changes—or if the tool name itself is swapped—the SHA-256 digest changes completely and execution is blocked immediately.

---

## How It Works Under the Hood

```text
  1. Time of Check (Create Request)
     Agent proposes: ("transfer_funds", {"to_account": "acct_vendor_42", "amount_usd": 150.0})
          │
          ▼
     canonical_json() ──► SHA-256 ──► payload_hash = "9c4a1d..."
          │
          ▼
     ApprovalStore stores PENDING request (deep-copied args, payload_hash, expires_at)
          │
  2. Human Review
     Reviewer inspects exact card ──► Approves ──► single-use token minted
                                  └── Rejects  ──► status = REJECTED (no token)
          │
  3. Time of Use (ApprovalGate.execute_with_approval)
     Caller supplies: (approval_id, token, runtime_tool, runtime_args)
          │
          ├─ Exists & status == APPROVED? ──► No  ──► BLOCK (rejected / pending / replay)
          ├─ now <= expires_at?           ──► No  ──► BLOCK (expired)
          ├─ hmac.compare_digest(token)?  ──► No  ──► BLOCK (invalid_token)
          ├─ Recompute SHA-256(runtime_tool, runtime_args)
          │   matches stored payload_hash?──► No  ──► BLOCK (TOCTOU payload_hash_mismatch!)
          │
          ▼ All checks passed
     Mark status = CONSUMED (single-use)
          │
          ▼
     Invoke tool_fn(**runtime_args)
```

---

## Code Walkthrough: `hitl_approvals.py`

Open [`hitl_approvals.py`](./hitl_approvals.py). It uses only the Python standard library (`hashlib`, `hmac`, `secrets`, `json`, `copy`, `time`).

Key building blocks:
- `_normalize_for_canonical()` & `canonical_json()`: recursively sorts dictionary keys, rejects `NaN`/`Infinity`, and normalizes exact whole-number floats (`150.0` -> `150`) so equivalent JSON payloads hash identically regardless of dict insertion order or float/int representation.
- `compute_payload_hash(tool_name, arguments)`: binds both the tool name and its normalized arguments into one SHA-256 hex digest.
- `format_approval_card(req, now)`: renders a readable review box showing the exact arguments, TTL countdown, and SHA-256 fingerprint.
- `ApprovalStore`: stores requests with deep-copied arguments, mints `secrets.token_urlsafe(24)` tokens on approval, checks TTL via an injectable clock, and verifies + atomically consumes approvals in `verify_and_consume()`.
- `ApprovalGate`: holds the registered tool callables privately and enforces `verify_and_consume()` on every call.

### Running the Script

Run from the repository root:

```bash
python3 23_Day_Human_in_the_Loop_Approvals/hitl_approvals.py
```

Sample output:

```text
=== Day 23: Human-in-the-Loop Approvals & TOCTOU Protection ===

--- Scenario 1: Normal Approved Execution & Single-Use Token ---
┌──────────────────────────────────────────────────────────────┐
│ HUMAN APPROVAL REQUIRED  [apr_4b91e0c72a11]
├──────────────────────────────────────────────────────────────┤
│ Session:      sess_fin_01
│ Action/Tool:  transfer_funds
│ Summary:      Pay vendor invoice #882 for $150.00
│ Expires:      2025-10-09 08:55:20 UTC (120s remaining)
│ SHA-256 Hash: 7d94a1f0b82c31e6d2148a9b...
│ Exact Arguments:
│   {
│     "amount_usd": 150.0,
│     "memo": "Invoice #882",
│     "to_account": "acct_vendor_42"
│   }
└──────────────────────────────────────────────────────────────┘
Execution result: executed=True | output={'tx_id': 'tx_0001', 'to_account': 'acct_vendor_42', 'amount_usd': 150.0, 'memo': 'Invoice #882'}
Replay attempt:   executed=False | code=replay_detected | Approval 'apr_4b91e0c72a11' was already consumed (replay blocked)

--- Scenario 2: Rejected Action ---
Execution result: executed=False | code=rejected | Action was rejected by maya@acme.internal: Unverified recipient

--- Scenario 3: Expired Approval Token ---
Execution result (+75s): executed=False | code=expired | Approval 'apr_...' expired before execution

--- Scenario 4: TOCTOU Tampering Attempt (Caught by SHA-256 Hash) ---
Tamper attempt A (recipient + $15,000): executed=False | code=payload_hash_mismatch
  Reason: TOCTOU payload mismatch: execution arguments do not match approved payload (approved=..., attempted=...)
Tamper attempt B (+$0.01 change):       executed=False | code=payload_hash_mismatch
  Reason: TOCTOU payload mismatch: execution arguments do not match approved payload (approved=..., attempted=...)

Total transfers executed: 1 (expected 1). All checks passed.
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Hashing only `arguments` without `tool_name`:** If two tools share similar argument shapes (say `preview_transfer(to_account, amount_usd)` and `execute_transfer(to_account, amount_usd)`), hashing only the arguments lets an approval for `preview_transfer` be reused on `execute_transfer`. Always include `tool_name` (and ideally `session_id` or `tenant_id`) inside the hashed envelope.
2. **Storing a shallow reference to `arguments`:** If the caller keeps a reference to the `args` dict passed into `create_request()` and mutates `args["amount_usd"] = 15000` later, a naive store that didn't `copy.deepcopy(arguments)` would display mutated arguments in logs. Always deep-copy on creation and bind the hash immediately.
3. **Marking `CONSUMED` *after* the tool runs instead of *before*:** Under concurrency (two threads or async tasks retrying at the same millisecond), checking `status == APPROVED`, awaiting a slow HTTP transfer, and *then* setting `CONSUMED` creates a double-spend race condition. In production, do an atomic database `UPDATE approvals SET status='consumed' WHERE id=? AND status='approved' AND expires_at > NOW() AND payload_hash=?` and execute the tool only if `rowcount == 1`.
4. **Using `==` instead of `hmac.compare_digest()`:** Direct string equality on bearer tokens leaks timing information. Using `hmac.compare_digest()` is one line and eliminates timing side channels.
5. **Non-deterministic JSON serialization:** In Python, `json.dumps({"a": 1, "b": 2})` and `json.dumps({"b": 2, "a": 1})` produce different byte strings unless `sort_keys=True` is set. Without canonicalization, harmless key reordering causes false hash mismatches.

---

## 💻 Exercises - Day 23

### Exercise: Level 1

1. Run `hitl_approvals.py` and verify that passing the same keys in a different order (`{"memo": "Invoice #882", "amount_usd": 150, "to_account": "acct_vendor_42"}`) produces the exact same `compute_payload_hash()` output as `args1`.
2. Add a 5th scenario in `run_demo()` that swaps `tool_name` from `"transfer_funds"` to `"wire_international"` while keeping `approved_args` identical, and assert that `payload_hash_mismatch` blocks it.
3. Add a check in `verify_and_consume()` that also binds `session_id` so an approval minted for `sess_fin_01` cannot be redeemed by `sess_fin_02`.
4. Add a `list_pending()` method to `ApprovalStore` that automatically flips any overdue `PENDING` requests to `EXPIRED` and returns only active pending requests.

### Exercise: Level 2

1. Persist `ApprovalStore` to SQLite (`sqlite3` in the Python standard library) and implement atomic single-use consumption using a conditional `UPDATE ... WHERE status = 'approved' AND payload_hash = ?` checking `cursor.rowcount == 1`. Test concurrent consumption from two threads (`threading.Thread`) to prove only one thread executes the tool.
2. Add **two-person approval (four-eyes rule)** for actions above a high threshold (for example, `required_approvals=2` for transfers over `$1,000`): the same `approver` cannot approve twice, and the token is minted only after two distinct approvers sign off.
3. Add a **dry-run diff preview** field to `ApprovalRequest` (e.g., `"Before: balance $1,200 -> After: balance $1,050"`) and display it cleanly inside `format_approval_card()`.
4. Combine Day 22's `PolicyEngine` with Day 23's `ApprovalGate` in a single script: when `PolicyEngine` returns `AUTO`, execute directly; when it returns `CONFIRM`, create an `ApprovalRequest`, prompt the terminal user `[a]pprove / [r]eject`, and run via `ApprovalGate`.

### Exercise: Level 3

1. Add **asymmetric or HMAC-signed approval receipts**: when a human approves a request, sign `(approval_id, payload_hash, approver, expires_at)` with a server secret key (`hmac.new(..., hashlib.sha256)`), so a stateless worker can verify the approval signature and payload hash without needing direct write access to the approval UI's memory.
2. Build a **revocation + automatic lockout** safeguard: if `ApprovalGate` catches a `payload_hash_mismatch` (TOCTOU attempt) on an approval token, immediately revoke that approval (`status = REVOKED`) so even a subsequent attempt with the original untampered arguments fails, and emit a high-severity security event.
3. Implement an **asynchronous webhook-style approval flow** using `asyncio`: an agent coroutine pauses on an `asyncio.Event` when a tool requires confirmation, a simulated Slack/UI handler coroutine approves or rejects it after a short delay (or times out past TTL), and the agent resumes cleanly with either the tool result or a structured rejection message.

---

[<< Day 22](../22_Day_Tool_Execution_Policies_and_Least_Privilege/README.md) | [Home](../README.md) | [Day 24 >>](../24_Day_Audit_Logs_and_Tamper_Evident_Receipts/README.md)
