"""Day 24: Audit Logs and Tamper-Evident Execution Receipts.

Implements structured execution receipts and an append-only SHA-256
hash-chained JSONL audit log:
- Logs approved, executed, denied, rejected, and failed actions equally.
- Stores SHA-256 hashes of full arguments alongside redacted summaries so logs
  remain verifiable without leaking raw secrets.
- Chains every record with `prev_hash` and `entry_hash` over canonical JSON.
- Provides an independent `verify_chain()` verifier that pinpoints the exact
  sequence number and line number of any edit, deletion, or reordering.

Run directly from the repository root:
    python3 24_Day_Audit_Logs_and_Tamper_Evident_Receipts/audit_log.py
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


GENESIS_HASH = "0" * 64


def canonical_json(obj: Any) -> str:
    """Serialize an object to deterministic compact JSON for hashing."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def hash_arguments(arguments: dict[str, Any]) -> str:
    """Compute SHA-256 digest of canonicalized tool arguments."""
    return hashlib.sha256(canonical_json(arguments).encode("utf-8")).hexdigest()


def compute_entry_hash(entry: dict[str, Any]) -> str:
    """Hash all receipt fields except `entry_hash` itself."""
    payload = {k: v for k, v in entry.items() if k != "entry_hash"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ReceiptOutcome(str, Enum):
    """Every lifecycle outcome that must leave an auditable receipt."""

    APPROVED = "approved"
    EXECUTED = "executed"
    DENIED = "denied"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class AuditReceipt:
    """Structured execution receipt stored as one line in a JSONL audit log."""

    seq: int
    timestamp_utc: str
    trace_id: str
    session_id: str
    actor: str
    tool_name: str
    args_sha256: str
    args_summary: dict[str, Any]
    outcome: str
    policy_rule: str
    reason: str
    approval_id: str | None = None
    duration_ms: float | None = None
    error: str | None = None
    prev_hash: str = GENESIS_HASH
    entry_hash: str = ""


@dataclass(frozen=True)
class ChainVerificationResult:
    """Structured output of `verify_chain()`."""

    valid: bool
    total_entries: int
    tip_hash: str
    first_corrupted_seq: int | None = None
    first_corrupted_line: int | None = None
    error: str | None = None


class HashChainedAuditLog:
    """Append-only JSONL audit log where each line cryptographically links to the previous line."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_seq = 0
        self._last_hash = GENESIS_HASH
        self._load_tail_state()

    def _load_tail_state(self) -> None:
        """Resume sequence number and tip hash if opening an existing log file."""
        if not self.log_path.exists():
            return

        lines = [line for line in self.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return

        last_entry = json.loads(lines[-1])
        self._last_seq = int(last_entry["seq"])
        self._last_hash = str(last_entry["entry_hash"])

    @property
    def tip_hash(self) -> str:
        return self._last_hash

    @property
    def count(self) -> int:
        return self._last_seq

    def append(
        self,
        *,
        trace_id: str,
        session_id: str,
        actor: str,
        tool_name: str,
        arguments: dict[str, Any],
        args_summary: dict[str, Any],
        outcome: ReceiptOutcome,
        policy_rule: str,
        reason: str,
        approval_id: str | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> AuditReceipt:
        """Create, hash-chain, and fsync a new structured execution receipt."""
        next_seq = self._last_seq + 1
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        receipt = AuditReceipt(
            seq=next_seq,
            timestamp_utc=timestamp,
            trace_id=trace_id,
            session_id=session_id,
            actor=actor,
            tool_name=tool_name,
            args_sha256=hash_arguments(arguments),
            args_summary=args_summary,
            outcome=outcome.value,
            policy_rule=policy_rule,
            reason=reason,
            approval_id=approval_id,
            duration_ms=duration_ms,
            error=error,
            prev_hash=self._last_hash,
            entry_hash="",
        )

        entry_dict = asdict(receipt)
        entry_hash = compute_entry_hash(entry_dict)
        receipt.entry_hash = entry_hash
        entry_dict["entry_hash"] = entry_hash

        line = canonical_json(entry_dict) + "\n"
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

        self._last_seq = next_seq
        self._last_hash = entry_hash
        return receipt


def verify_chain(
    log_path: Path,
    expected_tip_hash: str | None = None,
) -> ChainVerificationResult:
    """Independently verify sequence continuity and SHA-256 hash links for a JSONL audit log.

    Optionally checks the final `tip_hash` against an externally anchored
    checkpoint to detect full-file rewrites.
    """
    path = Path(log_path)
    if not path.exists():
        return ChainVerificationResult(
            valid=False,
            total_entries=0,
            tip_hash=GENESIS_HASH,
            error=f"Audit log file not found: {path}",
        )

    expected_seq = 1
    expected_prev_hash = GENESIS_HASH
    entries_checked = 0

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                return ChainVerificationResult(
                    valid=False,
                    total_entries=entries_checked,
                    tip_hash=expected_prev_hash,
                    first_corrupted_seq=expected_seq,
                    first_corrupted_line=line_no,
                    error=f"Invalid JSON on line {line_no}: {exc}",
                )

            if not isinstance(entry, dict):
                return ChainVerificationResult(
                    valid=False,
                    total_entries=entries_checked,
                    tip_hash=expected_prev_hash,
                    first_corrupted_seq=expected_seq,
                    first_corrupted_line=line_no,
                    error=f"Line {line_no} is not a JSON object",
                )

            seq = entry.get("seq")
            if seq != expected_seq:
                return ChainVerificationResult(
                    valid=False,
                    total_entries=entries_checked,
                    tip_hash=expected_prev_hash,
                    first_corrupted_seq=expected_seq,
                    first_corrupted_line=line_no,
                    error=(
                        f"Sequence discontinuity at line {line_no}: "
                        f"expected seq={expected_seq}, found seq={seq} "
                        "(possible deletion, insertion, or reordering)"
                    ),
                )

            stored_prev = entry.get("prev_hash")
            if not isinstance(stored_prev, str) or not hmac.compare_digest(stored_prev, expected_prev_hash):
                return ChainVerificationResult(
                    valid=False,
                    total_entries=entries_checked,
                    tip_hash=expected_prev_hash,
                    first_corrupted_seq=seq,
                    first_corrupted_line=line_no,
                    error=(
                        f"Broken prev_hash link at seq={seq} (line {line_no}): "
                        f"expected {expected_prev_hash[:16]}..., found {str(stored_prev)[:16]}..."
                    ),
                )

            stored_hash = entry.get("entry_hash")
            recomputed = compute_entry_hash(entry)
            if not isinstance(stored_hash, str) or not hmac.compare_digest(stored_hash, recomputed):
                return ChainVerificationResult(
                    valid=False,
                    total_entries=entries_checked,
                    tip_hash=expected_prev_hash,
                    first_corrupted_seq=seq,
                    first_corrupted_line=line_no,
                    error=(
                        f"Tampered payload at seq={seq} (line {line_no}): "
                        f"stored hash {str(stored_hash)[:16]}... != recomputed {recomputed[:16]}..."
                    ),
                )

            expected_prev_hash = stored_hash
            expected_seq += 1
            entries_checked += 1

    if expected_tip_hash is not None and not hmac.compare_digest(expected_prev_hash, expected_tip_hash):
        return ChainVerificationResult(
            valid=False,
            total_entries=entries_checked,
            tip_hash=expected_prev_hash,
            first_corrupted_seq=entries_checked,
            first_corrupted_line=entries_checked,
            error=(
                "Tip hash does not match externally anchored checkpoint "
                f"(anchored={expected_tip_hash[:16]}..., current={expected_prev_hash[:16]}...)"
            ),
        )

    return ChainVerificationResult(
        valid=True,
        total_entries=entries_checked,
        tip_hash=expected_prev_hash,
    )


def _tamper_edit_line(src: Path, dst: Path, line_index_zero_based: int) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[line_index_zero_based])
    # Silently alter the logged refund amount from 150.0 to 15.0 without updating hashes
    entry["args_summary"]["amount_usd"] = 15.0
    lines[line_index_zero_based] = canonical_json(entry)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tamper_delete_line(src: Path, dst: Path, line_index_zero_based: int) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    del lines[line_index_zero_based]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tamper_swap_lines(src: Path, dst: Path, idx_a: int, idx_b: int) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    lines[idx_a], lines[idx_b] = lines[idx_b], lines[idx_a]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tamper_rewrite_chain_from(src: Path, dst: Path, start_idx: int) -> None:
    """Simulate an attacker with disk access editing an entry and re-hashing subsequent lines."""
    lines = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines[start_idx]["args_summary"]["amount_usd"] = 15.0
    prev = GENESIS_HASH if start_idx == 0 else lines[start_idx - 1]["entry_hash"]
    for idx in range(start_idx, len(lines)):
        lines[idx]["prev_hash"] = prev
        lines[idx]["entry_hash"] = compute_entry_hash(lines[idx])
        prev = lines[idx]["entry_hash"]
    dst.write_text("\n".join(canonical_json(e) for e in lines) + "\n", encoding="utf-8")


def run_demo(custom_log_path: Path | None = None) -> None:
    """Log approved/executed/denied/rejected/failed receipts, then test tamper detection."""
    print("=== Day 24: Tamper-Evident Hash-Chained Audit Log Demo ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        log_path = custom_log_path if custom_log_path is not None else (base_dir / "audit_receipts.jsonl")
        if log_path.exists():
            log_path.unlink()

        audit = HashChainedAuditLog(log_path)

        # 1. Auto-approved read action (executed)
        r1 = audit.append(
            trace_id="tr_9001",
            session_id="sess_ops_07",
            actor="agent:support-bot",
            tool_name="get_order",
            arguments={"order_id": "ORD-8821"},
            args_summary={"order_id": "ORD-8821"},
            outcome=ReceiptOutcome.EXECUTED,
            policy_rule="static:get_order",
            reason="Auto-allowed read-only lookup",
            duration_ms=18.4,
        )

        # 2. Human approval recorded before executing a $150 refund
        r2 = audit.append(
            trace_id="tr_9001",
            session_id="sess_ops_07",
            actor="human:maya@acme.internal",
            tool_name="issue_refund",
            arguments={"order_id": "ORD-8821", "amount_usd": 150.0, "customer_email": "dana@example.com"},
            args_summary={"order_id": "ORD-8821", "amount_usd": 150.0, "customer_email": "[REDACTED]"},
            outcome=ReceiptOutcome.APPROVED,
            policy_rule="refund:confirm_tier",
            reason="Approved against verified return receipt",
            approval_id="apr_7a3c91",
        )

        # 3. Approved refund executed
        r3 = audit.append(
            trace_id="tr_9001",
            session_id="sess_ops_07",
            actor="agent:support-bot",
            tool_name="issue_refund",
            arguments={"order_id": "ORD-8821", "amount_usd": 150.0, "customer_email": "dana@example.com"},
            args_summary={"order_id": "ORD-8821", "amount_usd": 150.0, "customer_email": "[REDACTED]"},
            outcome=ReceiptOutcome.EXECUTED,
            policy_rule="refund:confirm_tier",
            reason="Executed with verified single-use approval token",
            approval_id="apr_7a3c91",
            duration_ms=142.7,
        )

        # 4. Policy denial logged equally (crucial for detecting prompt injection probing)
        r4 = audit.append(
            trace_id="tr_9002",
            session_id="sess_ops_07",
            actor="agent:support-bot",
            tool_name="read_workspace_file",
            arguments={"path": "../../etc/passwd"},
            args_summary={"path": "../../etc/passwd"},
            outcome=ReceiptOutcome.DENIED,
            policy_rule="file:path_traversal_blocked",
            reason="Path escapes allowed workspace",
        )

        # 5. Human rejection logged equally
        r5 = audit.append(
            trace_id="tr_9003",
            session_id="sess_ops_08",
            actor="human:maya@acme.internal",
            tool_name="transfer_funds",
            arguments={"to_account": "acct_unknown_77", "amount_usd": 420.0},
            args_summary={"to_account": "acct_unknown_77", "amount_usd": 420.0},
            outcome=ReceiptOutcome.REJECTED,
            policy_rule="transfer:confirm_tier",
            reason="Recipient account failed manual verification",
            approval_id="apr_9f0e11",
        )

        # 6. Downstream tool failure logged equally
        r6 = audit.append(
            trace_id="tr_9004",
            session_id="sess_ops_08",
            actor="agent:support-bot",
            tool_name="send_email",
            arguments={"to": "ops@acme.internal", "subject": "Daily reconciliation"},
            args_summary={"to": "ops@acme.internal", "subject": "Daily reconciliation"},
            outcome=ReceiptOutcome.FAILED,
            policy_rule="email:internal_auto",
            reason="SMTP gateway timed out after 5000ms",
            duration_ms=5002.1,
            error="TimeoutError: SMTP connect timed out",
        )

        for r in (r1, r2, r3, r4, r5, r6):
            print(
                f"  seq={r.seq} | {r.outcome.upper():8s} | {r.tool_name:20s} | "
                f"prev={r.prev_hash[:10]}.. -> hash={r.entry_hash[:10]}.."
            )

        anchored_checkpoint = audit.tip_hash
        print(f"\nAnchored external checkpoint (tip hash): {anchored_checkpoint[:24]}...")

        # Verify intact chain
        ok_res = verify_chain(log_path, expected_tip_hash=anchored_checkpoint)
        print(f"\n[Check 1] Intact chain: valid={ok_res.valid} | entries={ok_res.total_entries}")
        assert ok_res.valid and ok_res.total_entries == 6

        # Tamper Test A: In-place mutation of seq=3 (line 3)
        edited_path = base_dir / "tampered_edit.jsonl"
        _tamper_edit_line(log_path, edited_path, line_index_zero_based=2)
        res_edit = verify_chain(edited_path)
        print(f"[Check 2] In-place edit of seq=3:   valid={res_edit.valid} | seq={res_edit.first_corrupted_seq} | {res_edit.error}")
        assert not res_edit.valid and res_edit.first_corrupted_seq == 3

        # Tamper Test B: Silent deletion of seq=4 (the denied path-traversal attempt)
        deleted_path = base_dir / "tampered_delete.jsonl"
        _tamper_delete_line(log_path, deleted_path, line_index_zero_based=3)
        res_del = verify_chain(deleted_path)
        print(f"[Check 3] Deletion of seq=4:        valid={res_del.valid} | seq={res_del.first_corrupted_seq} | {res_del.error}")
        assert not res_del.valid and res_del.first_corrupted_seq == 4

        # Tamper Test C: Reordering seq=2 and seq=3
        swapped_path = base_dir / "tampered_swap.jsonl"
        _tamper_swap_lines(log_path, swapped_path, 1, 2)
        res_swap = verify_chain(swapped_path)
        print(f"[Check 4] Swapping seq=2 and seq=3: valid={res_swap.valid} | seq={res_swap.first_corrupted_seq} | {res_swap.error}")
        assert not res_swap.valid and res_swap.first_corrupted_seq == 2

        # Tamper Test D: Full-chain rewrite from seq=3 onward caught by external checkpoint
        rewritten_path = base_dir / "tampered_rewrite.jsonl"
        _tamper_rewrite_chain_from(log_path, rewritten_path, start_idx=2)
        res_internal = verify_chain(rewritten_path)
        res_anchored = verify_chain(rewritten_path, expected_tip_hash=anchored_checkpoint)
        print(
            f"[Check 5] Re-hashed rewrite from seq=3: "
            f"without checkpoint valid={res_internal.valid}, "
            f"WITH checkpoint valid={res_anchored.valid}"
        )
        print(f"          Checkpoint error: {res_anchored.error}")
        assert res_internal.valid and not res_anchored.valid

        # Keep clean copy if user requested --log-file
        if custom_log_path is not None:
            print(f"\nSaved intact JSONL audit log to: {custom_log_path}")

    print("\nAll 5 audit log integrity checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 24 Tamper-Evident Hash-Chained Audit Log")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to keep the generated JSONL audit log for inspection.",
    )
    args = parser.parse_args()
    run_demo(custom_log_path=args.log_file)


if __name__ == "__main__":
    main()
