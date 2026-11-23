"""Day 23: Human-in-the-Loop (HITL) Approvals with TOCTOU Protection.

Implements a production-style approval workflow for high-impact agent actions:
- Canonical JSON serialization + SHA-256 payload hashing over (tool_name, arguments).
- Human-readable approval request cards with explicit impact details and TTL.
- Single-use high-entropy approval tokens with expiration.
- Execution gate that recomputes the SHA-256 payload hash right before running
  the tool, defeating Time-of-Check to Time-of-Use (TOCTOU) argument tampering.

Run directly from the repository root:
    python3 23_Day_Human_in_the_Loop_Approvals/hitl_approvals.py
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


def _normalize_for_canonical(value: Any) -> Any:
    """Recursively normalize Python values for deterministic JSON hashing.

    Rejects NaN/Infinity and normalizes whole-number floats (e.g., 150.0 -> 150)
    so equivalent JSON numbers hash identically across serializers while any
    real numeric change (150.00 -> 150.01) changes the digest.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float cannot be canonicalized: {value}")
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, dict):
        return {str(k): _normalize_for_canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_canonical(item) for item in value]
    raise TypeError(f"Unsupported type in tool arguments: {type(value).__name__}")


def canonical_json(obj: Any) -> str:
    """Produce canonical compact JSON with sorted keys and no non-finite floats."""
    normalized = _normalize_for_canonical(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def compute_payload_hash(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return hex SHA-256 digest binding the exact tool name and normalized arguments."""
    envelope = {
        "tool_name": tool_name,
        "arguments": arguments,
    }
    serialized = canonical_json(envelope).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class ApprovalStatus(str, Enum):
    """Lifecycle states for a human approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONSUMED = "consumed"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """Immutable-payload approval record stored by ApprovalStore."""

    approval_id: str
    session_id: str
    tool_name: str
    arguments: dict[str, Any]
    payload_hash: str
    summary: str
    created_at: float
    expires_at: float
    status: ApprovalStatus = ApprovalStatus.PENDING
    token: str | None = None
    decided_by: str | None = None
    decision_note: str | None = None
    decided_at: float | None = None
    consumed_at: float | None = None


@dataclass(frozen=True)
class GateResult:
    """Outcome returned by ApprovalGate.execute_with_approval()."""

    executed: bool
    reason: str
    error_code: str | None = None
    output: Any = None


def format_approval_card(req: ApprovalRequest, now: float) -> str:
    """Render a clear human-readable approval prompt for review UIs or CLI."""
    ttl_remaining = max(0, int(req.expires_at - now))
    expires_iso = datetime.fromtimestamp(req.expires_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pretty_args = json.dumps(req.arguments, indent=2, sort_keys=True)
    return (
        "┌──────────────────────────────────────────────────────────────┐\n"
        f"│ HUMAN APPROVAL REQUIRED  [{req.approval_id}]\n"
        "├──────────────────────────────────────────────────────────────┤\n"
        f"│ Session:      {req.session_id}\n"
        f"│ Action/Tool:  {req.tool_name}\n"
        f"│ Summary:      {req.summary}\n"
        f"│ Expires:      {expires_iso} ({ttl_remaining}s remaining)\n"
        f"│ SHA-256 Hash: {req.payload_hash[:24]}...\n"
        "│ Exact Arguments:\n"
        + "\n".join(f"│   {line}" for line in pretty_args.splitlines())
        + "\n└──────────────────────────────────────────────────────────────┘"
    )


class ApprovalStore:
    """Manages approval requests, single-use tokens, TTL expiry, and hash verification."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._requests: dict[str, ApprovalRequest] = {}

    def create_request(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        summary: str,
        ttl_seconds: float = 300.0,
    ) -> ApprovalRequest:
        """Create a pending approval bound to the canonical SHA-256 hash of tool + args."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        now = self._clock()
        # Deep-copy arguments so later external mutation cannot silently alter the stored record.
        frozen_args = copy.deepcopy(arguments)
        payload_hash = compute_payload_hash(tool_name, frozen_args)
        approval_id = f"apr_{secrets.token_hex(6)}"

        req = ApprovalRequest(
            approval_id=approval_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=frozen_args,
            payload_hash=payload_hash,
            summary=summary,
            created_at=now,
            expires_at=now + ttl_seconds,
        )
        self._requests[approval_id] = req
        return req

    def approve(self, approval_id: str, approver: str, note: str = "") -> str:
        """Approve a pending request and issue a single-use approval token."""
        req = self._requests.get(approval_id)
        if req is None:
            raise KeyError(f"Unknown approval_id: {approval_id}")

        now = self._clock()
        if now > req.expires_at:
            req.status = ApprovalStatus.EXPIRED
            raise ValueError(f"Approval request {approval_id} has already expired")

        if req.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot approve request in status '{req.status.value}'")

        token = secrets.token_urlsafe(24)
        req.status = ApprovalStatus.APPROVED
        req.token = token
        req.decided_by = approver
        req.decision_note = note
        req.decided_at = now
        return token

    def reject(self, approval_id: str, approver: str, reason: str = "") -> None:
        """Explicitly reject a pending approval request."""
        req = self._requests.get(approval_id)
        if req is None:
            raise KeyError(f"Unknown approval_id: {approval_id}")

        if req.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot reject request in status '{req.status.value}'")

        req.status = ApprovalStatus.REJECTED
        req.decided_by = approver
        req.decision_note = reason
        req.decided_at = self._clock()

    def verify_and_consume(
        self,
        approval_id: str,
        token: str | None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, str, str | None]:
        """Verify token, expiry, single-use state, and SHA-256 payload hash.

        Returns (ok, reason, error_code). Marks the grant CONSUMED on success
        so it cannot be replayed.
        """
        req = self._requests.get(approval_id)
        if req is None:
            return False, f"Unknown approval_id '{approval_id}'", "unknown_approval"

        now = self._clock()

        if req.status == ApprovalStatus.REJECTED:
            return False, f"Action was rejected by {req.decided_by}: {req.decision_note}", "rejected"

        if req.status == ApprovalStatus.CONSUMED:
            return False, f"Approval '{approval_id}' was already consumed (replay blocked)", "replay_detected"

        if req.status == ApprovalStatus.PENDING:
            return False, f"Approval '{approval_id}' is still pending human review", "not_approved"

        if req.status == ApprovalStatus.EXPIRED or now > req.expires_at:
            req.status = ApprovalStatus.EXPIRED
            return False, f"Approval '{approval_id}' expired before execution", "expired"

        if not token or not req.token or not hmac.compare_digest(req.token, token):
            return False, "Invalid approval token", "invalid_token"

        # TOCTOU Defense: recompute canonical SHA-256 right before tool execution.
        actual_hash = compute_payload_hash(tool_name, arguments)
        if not hmac.compare_digest(req.payload_hash, actual_hash):
            return (
                False,
                (
                    "TOCTOU payload mismatch: execution arguments do not match approved payload "
                    f"(approved={req.payload_hash[:16]}..., attempted={actual_hash[:16]}...)"
                ),
                "payload_hash_mismatch",
            )

        # Consume atomically before tool runs to guarantee single-use semantics.
        req.status = ApprovalStatus.CONSUMED
        req.consumed_at = now
        return True, f"Verified approval '{approval_id}' granted by {req.decided_by}", None


class ApprovalGate:
    """Execution gate requiring a verified, hash-bound approval before invoking tools."""

    def __init__(self, store: ApprovalStore) -> None:
        self._store = store
        self._tools: dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    def execute_with_approval(
        self,
        approval_id: str,
        token: str | None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> GateResult:
        """Verify approval grant against exact runtime payload, then execute."""
        ok, reason, error_code = self._store.verify_and_consume(
            approval_id=approval_id,
            token=token,
            tool_name=tool_name,
            arguments=arguments,
        )
        if not ok:
            return GateResult(executed=False, reason=reason, error_code=error_code)

        tool_fn = self._tools.get(tool_name)
        if tool_fn is None:
            return GateResult(
                executed=False,
                reason=f"Tool '{tool_name}' is not registered in ApprovalGate",
                error_code="missing_tool",
            )

        output = tool_fn(**arguments)
        return GateResult(executed=True, reason=reason, output=output)


def run_demo() -> None:
    """Run all 4 core HITL approval scenarios plus replay prevention."""
    simulated_time = 1_760_000_000.0

    def fake_clock() -> float:
        return simulated_time

    store = ApprovalStore(clock=fake_clock)
    gate = ApprovalGate(store)

    executed_transfers: list[dict[str, Any]] = []

    def transfer_funds(to_account: str, amount_usd: float, memo: str) -> dict[str, Any]:
        record = {"tx_id": f"tx_{len(executed_transfers) + 1:04d}", "to_account": to_account, "amount_usd": amount_usd, "memo": memo}
        executed_transfers.append(record)
        return record

    gate.register_tool("transfer_funds", transfer_funds)

    print("=== Day 23: Human-in-the-Loop Approvals & TOCTOU Protection ===\n")

    # ------------------------------------------------------------------
    # Scenario 1: Normal approved execution (+ single-use replay block)
    # ------------------------------------------------------------------
    print("--- Scenario 1: Normal Approved Execution & Single-Use Token ---")
    args1 = {"to_account": "acct_vendor_42", "amount_usd": 150.00, "memo": "Invoice #882"}
    req1 = store.create_request(
        session_id="sess_fin_01",
        tool_name="transfer_funds",
        arguments=args1,
        summary="Pay vendor invoice #882 for $150.00",
        ttl_seconds=120,
    )
    print(format_approval_card(req1, simulated_time))

    token1 = store.approve(req1.approval_id, approver="maya@acme.internal", note="Matched PO-882")
    res1 = gate.execute_with_approval(req1.approval_id, token1, "transfer_funds", args1)
    print(f"Execution result: executed={res1.executed} | output={res1.output}")
    assert res1.executed and res1.output["amount_usd"] == 150.0

    # Attempt to replay the same approval token a second time
    replay_res = gate.execute_with_approval(req1.approval_id, token1, "transfer_funds", args1)
    print(f"Replay attempt:   executed={replay_res.executed} | code={replay_res.error_code} | {replay_res.reason}\n")
    assert not replay_res.executed and replay_res.error_code == "replay_detected"

    # ------------------------------------------------------------------
    # Scenario 2: Explicitly rejected action
    # ------------------------------------------------------------------
    print("--- Scenario 2: Rejected Action ---")
    args2 = {"to_account": "acct_unknown_77", "amount_usd": 420.00, "memo": "Urgent request"}
    req2 = store.create_request(
        session_id="sess_fin_01",
        tool_name="transfer_funds",
        arguments=args2,
        summary="Transfer $420.00 to unverified account",
        ttl_seconds=120,
    )
    store.reject(req2.approval_id, approver="maya@acme.internal", reason="Unverified recipient")
    res2 = gate.execute_with_approval(req2.approval_id, None, "transfer_funds", args2)
    print(f"Execution result: executed={res2.executed} | code={res2.error_code} | {res2.reason}\n")
    assert not res2.executed and res2.error_code == "rejected"

    # ------------------------------------------------------------------
    # Scenario 3: Expired approval rejected
    # ------------------------------------------------------------------
    print("--- Scenario 3: Expired Approval Token ---")
    args3 = {"to_account": "acct_vendor_42", "amount_usd": 200.00, "memo": "Invoice #889"}
    req3 = store.create_request(
        session_id="sess_fin_02",
        tool_name="transfer_funds",
        arguments=args3,
        summary="Pay vendor invoice #889 for $200.00",
        ttl_seconds=60,
    )
    token3 = store.approve(req3.approval_id, approver="alex@acme.internal")
    # Advance simulated clock past the 60s TTL
    simulated_time += 75.0
    res3 = gate.execute_with_approval(req3.approval_id, token3, "transfer_funds", args3)
    print(f"Execution result (+75s): executed={res3.executed} | code={res3.error_code} | {res3.reason}\n")
    assert not res3.executed and res3.error_code == "expired"

    # ------------------------------------------------------------------
    # Scenario 4: TOCTOU tampering caught immediately by SHA-256 mismatch
    # ------------------------------------------------------------------
    print("--- Scenario 4: TOCTOU Tampering Attempt (Caught by SHA-256 Hash) ---")
    approved_args = {"to_account": "acct_vendor_42", "amount_usd": 150.00, "memo": "Invoice #890"}
    req4 = store.create_request(
        session_id="sess_fin_03",
        tool_name="transfer_funds",
        arguments=approved_args,
        summary="Pay vendor invoice #890 for $150.00",
        ttl_seconds=120,
    )
    token4 = store.approve(req4.approval_id, approver="maya@acme.internal")

    # Attack A: Mutate recipient and amount dramatically after approval
    tampered_args = {"to_account": "acct_attacker_99", "amount_usd": 15000.00, "memo": "Invoice #890"}
    res4a = gate.execute_with_approval(req4.approval_id, token4, "transfer_funds", tampered_args)
    print(f"Tamper attempt A (recipient + $15,000): executed={res4a.executed} | code={res4a.error_code}")
    print(f"  Reason: {res4a.reason}")
    assert not res4a.executed and res4a.error_code == "payload_hash_mismatch"

    # Attack B: Even a single-cent mutation fails the hash check
    subtle_tamper = {"to_account": "acct_vendor_42", "amount_usd": 150.01, "memo": "Invoice #890"}
    res4b = gate.execute_with_approval(req4.approval_id, token4, "transfer_funds", subtle_tamper)
    print(f"Tamper attempt B (+$0.01 change):       executed={res4b.executed} | code={res4b.error_code}")
    print(f"  Reason: {res4b.reason}\n")
    assert not res4b.executed and res4b.error_code == "payload_hash_mismatch"

    # Confirm only Scenario 1 actually moved funds
    assert len(executed_transfers) == 1
    print(f"Total transfers executed: {len(executed_transfers)} (expected 1). All checks passed.")


if __name__ == "__main__":
    run_demo()
