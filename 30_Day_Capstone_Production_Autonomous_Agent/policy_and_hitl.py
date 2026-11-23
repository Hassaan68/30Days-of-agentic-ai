"""Static & dynamic tool policies plus HITL approvals with TOCTOU protection."""

from __future__ import annotations

import hmac
import time
import uuid
from typing import Any, Callable

from models import (
    ApprovalGrant,
    ApprovalRequest,
    PolicyAction,
    PolicyDecision,
    ToolCall,
    canonical_payload_hash,
)

ApproverFn = Callable[[ApprovalRequest], ApprovalGrant]


class TOCTOUTamperingError(RuntimeError):
    """Raised if tool arguments/scope differ from the hash approved by a human."""


class ApprovalValidationError(RuntimeError):
    """Raised on expired, denied, or replayed approval grants."""


class ToolPolicyEngine:
    """Evaluates static + argument-aware dynamic execution policies per tool call."""

    STATIC_RULES: dict[str, tuple[PolicyAction, str, str]] = {
        "get_service_health": (PolicyAction.AUTO, "Read-only health lookup", "static_read_auto"),
        "query_error_logs": (PolicyAction.AUTO, "Read-only log query", "static_read_auto"),
        "search_runbooks": (PolicyAction.AUTO, "Read-only runbook search", "static_read_auto"),
        "restart_service": (
            PolicyAction.CONFIRM,
            "Restarting a service affects live traffic",
            "static_mutation_confirm",
        ),
        "rollback_deployment": (
            PolicyAction.CONFIRM,
            "Rolling back a production release requires operator approval",
            "static_mutation_confirm",
        ),
        "purge_database": (
            PolicyAction.DENY,
            "Destructive database purge is strictly forbidden for autonomous agents",
            "static_catastrophic_deny",
        ),
    }

    def __init__(self, max_calls_per_run: int = 12) -> None:
        self.max_calls_per_run = max_calls_per_run
        self._call_counts: dict[str, int] = {}

    def evaluate(self, run_id: str, call: ToolCall) -> PolicyDecision:
        count = self._call_counts.get(run_id, 0) + 1
        self._call_counts[run_id] = count
        if count > self.max_calls_per_run:
            return PolicyDecision(
                PolicyAction.DENY,
                f"Per-run tool rate limit exceeded ({count} > {self.max_calls_per_run})",
                "dynamic_rate_limit",
            )

        # Dynamic argument-aware policy for incident tickets.
        if call.name == "create_incident_ticket":
            severity = str(call.arguments.get("severity", "sev2")).lower()
            if severity == "sev1":
                return PolicyDecision(
                    PolicyAction.CONFIRM,
                    "SEV1 pages executive on-call; requires human confirmation",
                    "dynamic_sev1_confirm",
                )
            return PolicyDecision(
                PolicyAction.AUTO,
                f"Standard {severity.upper()} ticket creation allowed automatically",
                "dynamic_ticket_auto",
            )

        # Dynamic scope escalation: any unknown tool defaults to DENY (least privilege).
        rule = self.STATIC_RULES.get(call.name)
        if rule is None:
            return PolicyDecision(
                PolicyAction.DENY,
                f"Unknown tool '{call.name}' denied under default-deny policy",
                "default_deny",
            )

        action, reason, rule_id = rule
        # Dynamic safeguard: refuse non-staging restarts if force=True is requested.
        if call.name == "restart_service" and call.arguments.get("force") is True:
            return PolicyDecision(
                PolicyAction.DENY,
                "Forced ungraceful restart (force=True) is denied",
                "dynamic_no_force_restart",
            )

        return PolicyDecision(action, reason, rule_id)


class HITLApprovalGate:
    """Issues hash-bound approval requests and verifies grants right before execution."""

    def __init__(
        self,
        approver: ApproverFn,
        ttl_seconds: float = 300.0,
    ) -> None:
        self.approver = approver
        self.ttl_seconds = ttl_seconds
        self._consumed_nonces: set[str] = set()

    def request_approval(
        self,
        run_id: str,
        call: ToolCall,
        reason: str,
    ) -> tuple[ApprovalRequest, ApprovalGrant]:
        payload_hash = canonical_payload_hash(call.name, call.arguments, call.scope)
        req = ApprovalRequest(
            approval_id=f"apr_{uuid.uuid4().hex[:10]}",
            run_id=run_id,
            tool_name=call.name,
            arguments=dict(call.arguments),
            scope=call.scope,
            payload_hash=payload_hash,
            reason=reason,
            expires_at_epoch=time.time() + self.ttl_seconds,
            nonce=uuid.uuid4().hex,
        )
        grant = self.approver(req)
        return req, grant

    def verify_grant_before_execution(
        self,
        grant: ApprovalGrant,
        call: ToolCall,
    ) -> None:
        """Re-hash the exact tool payload right before execution to block TOCTOU attacks."""
        if not grant.approved:
            raise ApprovalValidationError(
                f"Approval {grant.approval_id} was denied by {grant.approver}"
            )
        if time.time() > grant.expires_at_epoch:
            raise ApprovalValidationError(
                f"Approval {grant.approval_id} has expired"
            )
        if grant.nonce in self._consumed_nonces:
            raise ApprovalValidationError(
                f"Approval nonce {grant.nonce} was already consumed (replay blocked)"
            )

        live_hash = canonical_payload_hash(call.name, call.arguments, call.scope)
        if not hmac.compare_digest(live_hash, grant.payload_hash):
            raise TOCTOUTamperingError(
                f"TOCTOU mismatch for {call.name}: approved hash={grant.payload_hash[:12]} "
                f"!= execution hash={live_hash[:12]}"
            )

        self._consumed_nonces.add(grant.nonce)


# ---------------------------------------------------------------------------
# Built-in Approver Callbacks (Demo / Automated / Interactive CLI)
# ---------------------------------------------------------------------------


def make_policy_approver(
    approve_tools: set[str] | None = None,
    operator_name: str = "oncall-sre@example.com",
) -> ApproverFn:
    """Deterministic approver for demos and automated tests."""

    def _approve(req: ApprovalRequest) -> ApprovalGrant:
        allowed = True if approve_tools is None else req.tool_name in approve_tools
        return ApprovalGrant(
            approval_id=req.approval_id,
            approved=allowed,
            approver=operator_name,
            payload_hash=req.payload_hash,
            expires_at_epoch=req.expires_at_epoch,
            nonce=req.nonce,
            comment="approved via demo operator policy" if allowed else "denied by operator",
        )

    return _approve


def cli_interactive_approver(req: ApprovalRequest) -> ApprovalGrant:
    """Interactive terminal prompt for live HITL approval."""
    print("\n--- HUMAN APPROVAL REQUIRED ---")
    print(f"  Approval ID : {req.approval_id}")
    print(f"  Tool        : {req.tool_name} (scope={req.scope})")
    print(f"  Arguments   : {req.arguments}")
    print(f"  Payload Hash: {req.payload_hash}")
    print(f"  Reason      : {req.reason}")
    answer = input("Approve this exact action? [y/N]: ").strip().lower()
    approved = answer in ("y", "yes")
    return ApprovalGrant(
        approval_id=req.approval_id,
        approved=approved,
        approver="interactive-cli-operator",
        payload_hash=req.payload_hash,
        expires_at_epoch=req.expires_at_epoch,
        nonce=req.nonce,
        comment="approved via CLI" if approved else "denied via CLI",
    )
