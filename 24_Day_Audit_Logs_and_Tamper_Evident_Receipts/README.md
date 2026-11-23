[<< Day 23](../23_Day_Human_in_the_Loop_Approvals/README.md) | [Home](../README.md) | [Day 25 >>](../25_Day_Observability_Tracing_and_Telemetry/README.md)

# Day 24: Audit Logs and Tamper-Evident Receipts

When an autonomous agent issues refunds, sends emails, or queries customer records, someone will eventually ask: *"Who approved this action, what exact arguments ran, and did the agent try anything that was blocked?"* Today we replace ad-hoc `print()` debugging with structured execution receipts stored in a SHA-256 hash-chained JSONL audit log.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
  - [Why Plain `print()` and Unstructured Text Logs Fail](#why-plain-print-and-unstructured-text-logs-fail)
  - [Recording Every Outcome Equally](#recording-every-outcome-equally)
  - [Designing a Structured Execution Receipt](#designing-a-structured-execution-receipt)
  - [SHA-256 Hash-Chained Append-Only Logs (and Their Threat Model)](#sha-256-hash-chained-append-only-logs-and-their-threat-model)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Code Walkthrough: `audit_log.py`](#code-walkthrough-audit_logpy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 24](#-exercises---day-24)

---

## Core Concept & Why It Matters

Observability traces (Day 25) help engineers debug performance and prompt quality. **Audit logs** serve a different master: incident forensics, customer disputes, and compliance reviews (SOC 2, HIPAA, PCI-DSS, financial audits). They must be structured, complete, and verifiable months later.

### Why Plain `print()` and Unstructured Text Logs Fail

Lines like `INFO: refunded order 8821` fall apart the moment an incident occurs:

1. **No stable schema:** Regex-parsing free-form log strings breaks every time a developer tweaks a print statement.
2. **Missing correlation IDs:** Without `trace_id`, `session_id`, and `approval_id`, you cannot tie a tool execution back to the exact prompt turn and human sign-off.
3. **Silent omissions:** Teams naturally log successes and forget to log policy denials or human rejections.
4. **Zero integrity proof:** Anyone with access to a text log file can edit `150.0` to `15.0` or delete an embarrassing line, leaving no evidence anything changed.

### Recording Every Outcome Equally

A common mistake is writing audit records only when a tool actually runs. In practice, all five lifecycle outcomes deserve identical structured receipts:

- **`approved`**: Records *who* signed off, *when*, and the `approval_id` + argument digest they approved.
- **`executed`**: Records the actual tool run, latency, and which policy rule or approval authorized it.
- **`denied`**: Records every action blocked by `PolicyEngine` (Day 22). Repeated `DENIED` receipts—like an agent trying `../../etc/passwd` three times in a row—are often your earliest indicator of an active indirect prompt injection.
- **`rejected`**: Records actions a human reviewer explicitly turned down.
- **`failed`**: Records permitted actions that crashed or timed out downstream, so responders can distinguish a blocked action from a partial failure.

### Designing a Structured Execution Receipt

Each JSONL line in `audit_log.py` captures:
- `seq`: strictly increasing 1-based sequence number.
- `timestamp_utc`: ISO-8601 UTC timestamp with millisecond precision.
- `trace_id` & `session_id`: links the receipt to Day 25's execution trace and user session.
- `actor`: distinguishes automated actions (`agent:support-bot`) from human decisions (`human:maya@acme.internal`).
- `tool_name`, `args_sha256`, and `args_summary`: hashes the full canonical arguments for cryptographic proof while storing only a **redacted summary** (`customer_email: "[REDACTED]"`) in plaintext. That way the audit log does not turn into a PII/secret liability.
- `outcome`, `policy_rule`, `reason`, `approval_id`, `duration_ms`, `error`.
- `prev_hash` and `entry_hash`: the SHA-256 chain links.

### SHA-256 Hash-Chained Append-Only Logs (and Their Threat Model)

Every receipt links cryptographically to the receipt before it:

- Entry 1 sets `prev_hash = "0" * 64` (the genesis hash).
- `entry_hash = SHA-256(canonical_json(entry without entry_hash))`.
- Entry *N* sets `prev_hash = entry[N-1].entry_hash`.

If someone edits a single field on line 3, deletes line 4, or swaps lines 2 and 3, `verify_chain()` immediately spots the exact broken sequence number and line number.

**Be honest about the threat model:** A local hash chain provides *tamper evidence*, not *tamper prevention*. If an attacker gains write access to the log disk, they could edit line 3 **and** recompute `prev_hash`/`entry_hash` for lines 3 through *N* so the file looks internally consistent again (Check 5 in our demo proves this!). To defeat full-chain rewrites, periodically publish the latest `(seq, tip_hash)` checkpoint to a separate append-only destination—such as S3 Object Lock (WORM storage), a separate write-only log collector, or a KMS-signed checkpoint table—and pass that anchored `expected_tip_hash` to `verify_chain()`.

---

## How It Works Under the Hood

```text
  seq=1 (EXECUTED: get_order)
  ┌────────────────────────────────────────────┐
  │ prev_hash:  0000000000... (genesis)        │
  │ payload:    {seq:1, tool:"get_order", ...} │
  │ entry_hash: a91c4e7f2b...  ◄───────────────────┐
  └────────────────────────────────────────────┘   │
                                                   │ linked
  seq=2 (APPROVED: issue_refund)                   │
  ┌────────────────────────────────────────────┐   │
  │ prev_hash:  a91c4e7f2b...  ────────────────────┘
  │ payload:    {seq:2, outcome:"approved"...} │
  │ entry_hash: 5d02b8c311...  ◄───────────────────┐
  └────────────────────────────────────────────┘   │
                                                   │ linked
  seq=3 (EXECUTED: issue_refund)                   │
  ┌────────────────────────────────────────────┐   │
  │ prev_hash:  5d02b8c311...  ────────────────────┘
  │ payload:    {seq:3, amount:150.0, ...}     │
  │ entry_hash: e481f904aa...  ──► Anchor tip_hash externally
  └────────────────────────────────────────────┘

  Tamper Detection:
  - Edit seq=3 amount 150 -> 15  ──► entry_hash mismatch at seq=3
  - Delete seq=4 (DENIED entry)  ──► sequence gap + prev_hash break at line 4
  - Swap seq=2 and seq=3         ──► sequence discontinuity at line 2
  - Re-hash seq=3..N on disk     ──► tip_hash mismatch vs anchored checkpoint
```

---

## Code Walkthrough: `audit_log.py`

Open [`audit_log.py`](./audit_log.py). It runs with the Python standard library alone (`hashlib`, `hmac`, `json`, `os`, `tempfile`).

Key components:
- `AuditReceipt` & `ReceiptOutcome`: typed schema covering `approved`, `executed`, `denied`, `rejected`, and `failed`.
- `HashChainedAuditLog`: append-only JSONL writer that resumes `(last_seq, last_hash)` on reopen, computes canonical SHA-256 `entry_hash`, writes one line, and calls `flush()` + `os.fsync()` so receipts survive sudden process crashes.
- `verify_chain(log_path, expected_tip_hash=None)`: standalone verifier that validates JSON syntax, monotonic `seq`, `prev_hash` linkage, recomputed `entry_hash`, and optional externally anchored `expected_tip_hash`.
- `run_demo()`: logs 6 receipts covering all five outcomes, verifies the intact log, and then runs four deliberate tamper tests (in-place edit, silent deletion, line reordering, and full-chain rewrite detected via checkpoint).

### Running the Script

Run from the repository root:

```bash
python3 24_Day_Audit_Logs_and_Tamper_Evident_Receipts/audit_log.py
```

To keep the generated JSONL file and inspect it with `jq` or a text editor:

```bash
python3 24_Day_Audit_Logs_and_Tamper_Evident_Receipts/audit_log.py --log-file /tmp/day24_audit.jsonl
```

Sample output:

```text
=== Day 24: Tamper-Evident Hash-Chained Audit Log Demo ===

  seq=1 | EXECUTED | get_order            | prev=0000000000.. -> hash=9c14f2a8b1..
  seq=2 | APPROVED | issue_refund         | prev=9c14f2a8b1.. -> hash=3e70d91a4c..
  seq=3 | EXECUTED | issue_refund         | prev=3e70d91a4c.. -> hash=71b8c04e2a..
  seq=4 | DENIED   | read_workspace_file  | prev=71b8c04e2a.. -> hash=d4502f9b11..
  seq=5 | REJECTED | transfer_funds       | prev=d4502f9b11.. -> hash=8f2a910c3e..
  seq=6 | FAILED   | send_email           | prev=8f2a910c3e.. -> hash=1a6d4b7c92..

Anchored external checkpoint (tip hash): 1a6d4b7c92e804f1a3b2c5d6...

[Check 1] Intact chain: valid=True | entries=6
[Check 2] In-place edit of seq=3:   valid=False | seq=3 | Tampered payload at seq=3 (line 3): stored hash ... != recomputed ...
[Check 3] Deletion of seq=4:        valid=False | seq=4 | Sequence discontinuity at line 4: expected seq=4, found seq=5 (possible deletion, insertion, or reordering)
[Check 4] Swapping seq=2 and seq=3: valid=False | seq=2 | Sequence discontinuity at line 2: expected seq=2, found seq=3 (possible deletion, insertion, or reordering)
[Check 5] Re-hashed rewrite from seq=3: without checkpoint valid=True, WITH checkpoint valid=False
          Checkpoint error: Tip hash does not match externally anchored checkpoint (anchored=..., current=...)

All 5 audit log integrity checks passed.
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Dumping raw tool arguments straight into the audit log:** If a tool accepts passwords, session cookies, SSNs, or API keys, writing raw arguments creates a compliance violation inside your audit log itself. Store `args_sha256` (so you can still prove exact inputs) alongside a scrubbed `args_summary`.
2. **Concurrent writers corrupting the hash chain:** If two worker processes append to the same JSONL file simultaneously without a lock (`fcntl.flock` on POSIX, or a single log-writer queue/database sequence), they will read the same `prev_hash`, fork the chain, and fail `verify_chain()`. Serialize appends through a single writer or file lock.
3. **Log rotation breaking verification:** When rotating logs daily (`audit-2025-10-09.jsonl` -> `audit-2025-10-10.jsonl`), carry the final `tip_hash` of the previous file into the genesis header of the next file so auditors can verify across day boundaries.
4. **Skipping `fsync()` or buffering in memory:** If the process crashes right after executing a wire transfer while the receipt is still sitting in Python's user-space buffer, you lose the most important record. Flush and `fsync` audit writes (or write to a durable WAL-backed store).
5. **Forgetting to verify the chain automatically:** A hash chain nobody ever verifies is theater. Run `verify_chain()` on a schedule (e.g., hourly cron or before archiving each rotated file) and alert immediately on any failure.

---

## 💻 Exercises - Day 24

### Exercise: Level 1

1. Run `audit_log.py --log-file /tmp/day24_audit.jsonl`, inspect one raw JSONL line, and manually verify its `entry_hash` in a Python REPL using `hashlib.sha256`.
2. Reopen the same `/tmp/day24_audit.jsonl` file in a second `HashChainedAuditLog` instance, append a 7th receipt (`seq=7`), and confirm `verify_chain()` passes across the restart.
3. Add a tamper test that inserts a forged extra line between `seq=2` and `seq=3` and verify `verify_chain()` catches it.
4. Write a small helper function `summarize_outcomes(log_path)` that reads the JSONL file and prints counts grouped by `outcome` (`executed`, `approved`, `denied`, `rejected`, `failed`).

### Exercise: Level 2

1. Add a **CLI verifier mode** (`--verify <path> [--checkpoint <hash>]`) to `audit_log.py` that exits with status code `0` on a valid chain and non-zero (`1`) on any corruption so it can run inside CI or a cron job.
2. Add **HMAC-signed checkpoints**: every *K* entries (e.g., every 100 entries or on daily rotation), write a separate `checkpoints.jsonl` containing `{seq, tip_hash, timestamp_utc, hmac_sha256}` signed with a secret key, and verify the main log against the latest valid checkpoint.
3. Make `HashChainedAuditLog.append()` safe for multiple threads (via `threading.Lock`) and multiple local processes (via `fcntl.flock`), then test concurrent appends from 4 threads to prove `verify_chain()` still passes with zero forks or interleaved bytes.
4. Implement **chain-preserving log rotation**: when a log file exceeds `max_entries`, rotate to `audit_0001.jsonl`, `audit_0002.jsonl`, etc., carrying the previous segment's `tip_hash` as the next segment's `prev_hash`, and write `verify_rotated_segments(paths)` to verify all files end-to-end.

### Exercise: Level 3

1. Wire **Days 22 + 23 + 24** together into one unified `AuditedSafeExecutor`:
   - `PolicyEngine` evaluates every call (`AUTO`, `CONFIRM`, `DENY`).
   - Every `DENY` writes a `DENIED` receipt immediately.
   - Every `CONFIRM` creates an `ApprovalRequest`; approving writes an `APPROVED` receipt and rejecting writes a `REJECTED` receipt.
   - Every tool run writes an `EXECUTED` or `FAILED` receipt (and catches TOCTOU tampering by logging a `DENIED` receipt with `policy_rule="hitl:payload_hash_mismatch"`).
2. Implement a **Merkle tree batch root** over each day's `entry_hash` values so an auditor can prove a single specific receipt (say `seq=42`) existed in the day's log using an *O(log N)* inclusion proof without downloading the full multi-gigabyte log file.
3. Add a **redaction-enforced receipt builder** that integrates Day 21's `OutputGuardrail` on `args_summary` and `error` messages so stack traces containing accidental API keys or customer emails are automatically redacted before hitting the append-only audit log.

---

[<< Day 23](../23_Day_Human_in_the_Loop_Approvals/README.md) | [Home](../README.md) | [Day 25 >>](../25_Day_Observability_Tracing_and_Telemetry/README.md)
