"""Core data models and canonical hashing helpers for the Day 30 Capstone Agent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyAction(str, Enum):
    """Execution policy outcome for a requested tool call."""

    AUTO = "auto"
    CONFIRM = "confirm"
    DENY = "deny"


def canonical_json(data: Any) -> str:
    """Serialize data deterministically for cryptographic hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_payload_hash(
    tool_name: str,
    arguments: dict[str, Any],
    scope: str = "prod",
) -> str:
    """Compute a SHA-256 digest binding tool name, arguments, and scope."""
    payload = canonical_json(
        {"tool_name": tool_name, "arguments": arguments, "scope": scope}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PlanStep:
    """One decomposed step in an incident investigation plan."""

    step_id: int
    description: str
    target_tool: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """MCP-compatible tool metadata and JSON Schema definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: str  # "read", "write", "high_impact", "catastrophic"


@dataclass
class ToolCall:
    """Structured tool invocation requested by the agent."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    scope: str = "prod"


@dataclass
class ToolResult:
    """Structured result returned after tool execution or policy block."""

    call_id: str
    tool_name: str
    ok: bool
    output: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class PolicyDecision:
    """Decision from the tool policy engine."""

    action: PolicyAction
    reason: str
    rule_id: str


@dataclass(frozen=True)
class ApprovalRequest:
    """Human-in-the-loop approval request bound to a canonical payload hash."""

    approval_id: str
    run_id: str
    tool_name: str
    arguments: dict[str, Any]
    scope: str
    payload_hash: str
    reason: str
    expires_at_epoch: float
    nonce: str


@dataclass(frozen=True)
class ApprovalGrant:
    """Signed/bound approval decision returned by a human approver."""

    approval_id: str
    approved: bool
    approver: str
    payload_hash: str
    expires_at_epoch: float
    nonce: str
    comment: str = ""


@dataclass
class AuditReceipt:
    """Single link in the tamper-evident SHA-256 hash-chained audit ledger."""

    seq: int
    timestamp: str
    run_id: str
    event_type: str  # approved, executed, denied, blocked_guardrail, toctou_rejected, failed
    tool_name: str | None
    arguments_hash: str | None
    actor: str
    outcome: str
    details: dict[str, Any]
    prev_hash: str
    receipt_hash: str


@dataclass
class Span:
    """One timed span inside a nested execution trace."""

    span_id: str
    parent_id: str | None
    name: str
    start_offset_ms: float
    duration_ms: float
    status: str = "ok"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunResult:
    """Complete structured output from a single IncidentOpsAgent run."""

    run_id: str
    user_request: str
    status: str  # "completed", "blocked", "halted"
    final_response: str
    plan: list[PlanStep] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    recalled_memories: list[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
