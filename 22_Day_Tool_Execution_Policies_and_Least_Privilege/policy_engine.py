"""Day 22: Tool Execution Policies and Least Privilege.

Implements a deterministic PolicyEngine and GuardedToolExecutor that sit
between an agent's proposed tool calls and actual tool execution:
- Default-deny posture for unknown tools.
- Three-way policy decisions: AUTO, CONFIRM, DENY.
- Dynamic argument-aware policies (refund dollar thresholds, DB row limits).
- Least-privilege capability scopes (filesystem path traversal prevention,
  strict HTTPS domain allowlists blocking SSRF and subdomain spoofing).
- Sliding-window per-(session, tool) rate limiting.

Run directly from the repository root:
    python3 22_Day_Tool_Execution_Policies_and_Least_Privilege/policy_engine.py
"""

from __future__ import annotations

import ipaddress
import math
import re
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


class PolicyDecision(str, Enum):
    """Outcome returned by the PolicyEngine for a proposed tool call."""

    AUTO = "auto"        # Safe to execute immediately without human intervention
    CONFIRM = "confirm"  # Pause and require explicit human approval (Day 23)
    DENY = "deny"        # Reject immediately; do not execute


@dataclass(frozen=True)
class PolicyResult:
    """Structured explanation of a policy evaluation."""

    decision: PolicyDecision
    reason: str
    rule_name: str
    retry_after_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RateLimitRule:
    """Sliding-window limit: at most `max_calls` within `window_seconds`."""

    max_calls: int
    window_seconds: float


@dataclass(frozen=True)
class CapabilityScope:
    """Least-privilege boundaries granted to an agent session."""

    allowed_workspace_dir: Path
    allowed_http_domains: tuple[str, ...]
    internal_email_domains: tuple[str, ...] = ("acme.internal",)
    blocked_filenames: tuple[str, ...] = (".env", "id_rsa", "credentials.json")
    max_auto_refund_usd: float = 50.0
    max_confirm_refund_usd: float = 500.0
    max_auto_db_rows: int = 100
    max_confirm_db_rows: int = 1000


class SlidingWindowRateLimiter:
    """Tracks tool call timestamps per (session_id, tool_name) in a sliding window."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._windows: dict[tuple[str, str], deque[float]] = {}

    def check_and_record(
        self,
        session_id: str,
        tool_name: str,
        rule: RateLimitRule,
    ) -> tuple[bool, float | None]:
        """Return (allowed, retry_after_seconds). Records timestamp only if allowed."""
        now = self._clock()
        key = (session_id, tool_name)
        timestamps = self._windows.setdefault(key, deque())

        cutoff = now - rule.window_seconds
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        if len(timestamps) >= rule.max_calls:
            oldest = timestamps[0]
            retry_after = max(0.0, (oldest + rule.window_seconds) - now)
            return False, round(retry_after, 2)

        timestamps.append(now)
        return True, None


DynamicPolicyFn = Callable[[dict[str, Any], CapabilityScope], PolicyResult]


class PolicyEngine:
    """Evaluates proposed tool calls against static rules, dynamic policies, and rate limits.

    Follows a default-deny posture: any tool not explicitly registered in
    static_policies or dynamic_policies is denied.
    """

    def __init__(
        self,
        scope: CapabilityScope,
        rate_limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self.scope = scope
        self.rate_limiter = rate_limiter or SlidingWindowRateLimiter()
        self._static_policies: dict[str, PolicyDecision] = {}
        self._dynamic_policies: dict[str, DynamicPolicyFn] = {}
        self._rate_limits: dict[str, RateLimitRule] = {}

    def register_static(
        self,
        tool_name: str,
        decision: PolicyDecision,
        rate_limit: RateLimitRule | None = None,
    ) -> None:
        """Register a tool with a fixed static policy."""
        self._static_policies[tool_name] = decision
        if rate_limit is not None:
            self._rate_limits[tool_name] = rate_limit

    def register_dynamic(
        self,
        tool_name: str,
        evaluator: DynamicPolicyFn,
        rate_limit: RateLimitRule | None = None,
    ) -> None:
        """Register a tool whose decision depends on its arguments and capability scope."""
        self._dynamic_policies[tool_name] = evaluator
        if rate_limit is not None:
            self._rate_limits[tool_name] = rate_limit

    def evaluate(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PolicyResult:
        """Evaluate whether a tool call should be AUTO, CONFIRM, or DENY."""
        if not isinstance(arguments, dict):
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="Tool arguments must be a JSON object/dict",
                rule_name="invalid_arguments",
            )

        # 1. Policy evaluation (dynamic overrides static; unknown tools default-deny).
        if tool_name in self._dynamic_policies:
            result = self._dynamic_policies[tool_name](arguments, self.scope)
        elif tool_name in self._static_policies:
            decision = self._static_policies[tool_name]
            result = PolicyResult(
                decision=decision,
                reason=f"Static policy for '{tool_name}' is {decision.value}",
                rule_name=f"static:{tool_name}",
            )
        else:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Tool '{tool_name}' is not allowed by policy (default-deny)",
                rule_name="default_deny",
            )

        # Denied actions do not consume rate-limit budget.
        if result.decision == PolicyDecision.DENY:
            return result

        # 2. Sliding-window rate-limit check for allowed or confirmable actions.
        limit_rule = self._rate_limits.get(tool_name)
        if limit_rule is not None:
            allowed, retry_after = self.rate_limiter.check_and_record(
                session_id=session_id,
                tool_name=tool_name,
                rule=limit_rule,
            )
            if not allowed:
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=(
                        f"Rate limit exceeded for '{tool_name}' "
                        f"({limit_rule.max_calls} calls per {limit_rule.window_seconds:.0f}s)"
                    ),
                    rule_name=f"rate_limit:{tool_name}",
                    retry_after_seconds=retry_after,
                )

        return result


# ---------------------------------------------------------------------------
# Concrete dynamic policy evaluators
# ---------------------------------------------------------------------------

def evaluate_refund_policy(args: dict[str, Any], scope: CapabilityScope) -> PolicyResult:
    """Tiered dollar thresholds for issue_refund."""
    raw_amount = args.get("amount_usd")
    if isinstance(raw_amount, bool) or not isinstance(raw_amount, (int, float)):
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="amount_usd must be a number",
            rule_name="refund:invalid_type",
        )

    amount = float(raw_amount)
    if not math.isfinite(amount) or amount <= 0:
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason=f"Refund amount must be positive and finite (got {raw_amount})",
            rule_name="refund:invalid_amount",
        )

    if amount <= scope.max_auto_refund_usd:
        return PolicyResult(
            decision=PolicyDecision.AUTO,
            reason=f"Refund ${amount:.2f} is within auto-approval limit (${scope.max_auto_refund_usd:.2f})",
            rule_name="refund:auto_tier",
        )

    if amount <= scope.max_confirm_refund_usd:
        return PolicyResult(
            decision=PolicyDecision.CONFIRM,
            reason=(
                f"Refund ${amount:.2f} exceeds auto limit (${scope.max_auto_refund_usd:.2f}) "
                f"and requires human approval (up to ${scope.max_confirm_refund_usd:.2f})"
            ),
            rule_name="refund:confirm_tier",
        )

    return PolicyResult(
        decision=PolicyDecision.DENY,
        reason=f"Refund ${amount:.2f} exceeds maximum permitted cap (${scope.max_confirm_refund_usd:.2f})",
        rule_name="refund:hard_cap",
    )


def evaluate_file_read_policy(args: dict[str, Any], scope: CapabilityScope) -> PolicyResult:
    """Enforce workspace root containment and block sensitive filenames."""
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="File path must be a non-empty string",
            rule_name="file:invalid_path",
        )

    root = scope.allowed_workspace_dir.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = candidate.resolve()

    if not resolved.is_relative_to(root):
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason=f"Path '{raw_path}' escapes allowed workspace '{root}'",
            rule_name="file:path_traversal_blocked",
        )

    if resolved.name in scope.blocked_filenames:
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason=f"Access to sensitive file '{resolved.name}' is prohibited",
            rule_name="file:blocked_filename",
        )

    return PolicyResult(
        decision=PolicyDecision.AUTO,
        reason=f"Path '{resolved.relative_to(root)}' is inside allowed workspace",
        rule_name="file:workspace_allowed",
        metadata={"resolved_path": str(resolved)},
    )


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def evaluate_http_fetch_policy(args: dict[str, Any], scope: CapabilityScope) -> PolicyResult:
    """Enforce HTTPS-only domain allowlist and block SSRF / subdomain spoofing."""
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="url must be a non-empty string",
            rule_name="http:invalid_url",
        )

    parsed = urlparse(url)
    if parsed.scheme != "https":
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason=f"Only HTTPS URLs are permitted (got scheme '{parsed.scheme or 'none'}')",
            rule_name="http:scheme_denied",
        )

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or host == "localhost" or _is_ip_literal(host):
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason=f"Direct IP or localhost targets are blocked ('{host}')",
            rule_name="http:ssrf_blocked",
        )

    allowed = any(
        host == domain or host.endswith(f".{domain}")
        for domain in (d.lower().lstrip(".") for d in scope.allowed_http_domains)
    )
    if not allowed:
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason=f"Domain '{host}' is not in allowlist {scope.allowed_http_domains}",
            rule_name="http:domain_denied",
        )

    return PolicyResult(
        decision=PolicyDecision.AUTO,
        reason=f"HTTPS domain '{host}' is allowed",
        rule_name="http:domain_allowed",
    )


FORBIDDEN_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|exec)\b",
    re.IGNORECASE,
)


def evaluate_db_query_policy(args: dict[str, Any], scope: CapabilityScope) -> PolicyResult:
    """Ensure read-only SELECT queries and bound maximum returned rows."""
    sql = args.get("sql", "")
    limit = args.get("limit", 50)

    if not isinstance(sql, str) or not sql.strip().lower().startswith("select "):
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="Only single SELECT statements are allowed",
            rule_name="db:non_select_denied",
        )

    if ";" in sql.strip().rstrip(";"):
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="Multi-statement SQL queries are prohibited",
            rule_name="db:multi_statement_denied",
        )

    if FORBIDDEN_SQL_RE.search(sql):
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="Write or DDL SQL keyword detected",
            rule_name="db:write_keyword_denied",
        )

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="limit must be a positive integer",
            rule_name="db:invalid_limit",
        )

    if limit <= scope.max_auto_db_rows:
        return PolicyResult(
            decision=PolicyDecision.AUTO,
            reason=f"Read-only query within auto row cap ({limit} <= {scope.max_auto_db_rows})",
            rule_name="db:auto_rows",
        )

    if limit <= scope.max_confirm_db_rows:
        return PolicyResult(
            decision=PolicyDecision.CONFIRM,
            reason=f"Bulk export of {limit} rows requires human approval",
            rule_name="db:confirm_rows",
        )

    return PolicyResult(
        decision=PolicyDecision.DENY,
        reason=f"Row limit {limit} exceeds maximum cap ({scope.max_confirm_db_rows})",
        rule_name="db:hard_row_cap",
    )


def evaluate_send_email_policy(args: dict[str, Any], scope: CapabilityScope) -> PolicyResult:
    """Auto-allow internal emails; require confirmation for external recipients."""
    recipient = args.get("to")
    if not isinstance(recipient, str) or "@" not in recipient:
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="Recipient email address is missing or malformed",
            rule_name="email:invalid_recipient",
        )

    domain = recipient.rsplit("@", 1)[-1].strip().lower()
    if domain in (d.lower() for d in scope.internal_email_domains):
        return PolicyResult(
            decision=PolicyDecision.AUTO,
            reason=f"Internal recipient domain '{domain}' is auto-allowed",
            rule_name="email:internal_auto",
        )

    return PolicyResult(
        decision=PolicyDecision.CONFIRM,
        reason=f"External email to '{domain}' requires human confirmation",
        rule_name="email:external_confirm",
    )


# ---------------------------------------------------------------------------
# Policy Enforcement Point: separates policy evaluation from tool execution
# ---------------------------------------------------------------------------

@dataclass
class ExecutionOutcome:
    """Result returned to the agent loop by GuardedToolExecutor."""

    executed: bool
    policy: PolicyResult
    output: Any = None


class GuardedToolExecutor:
    """Wraps raw tool functions so every invocation must pass PolicyEngine first.

    The LLM never holds direct references to tool functions; it only submits
    (tool_name, arguments) payloads to this executor.
    """

    def __init__(self, policy_engine: PolicyEngine) -> None:
        self._policy_engine = policy_engine
        self._tools: dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    def execute(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ExecutionOutcome:
        """Evaluate policy first; run the tool only on AUTO."""
        policy = self._policy_engine.evaluate(session_id, tool_name, arguments)

        if policy.decision != PolicyDecision.AUTO:
            # Both DENY and CONFIRM halt execution here. CONFIRM hands off to
            # an approval workflow (covered in Day 23).
            return ExecutionOutcome(executed=False, policy=policy, output=None)

        tool_fn = self._tools.get(tool_name)
        if tool_fn is None:
            return ExecutionOutcome(
                executed=False,
                policy=PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=f"Tool '{tool_name}' has a policy but no registered implementation",
                    rule_name="missing_tool_impl",
                ),
                output=None,
            )

        # Pass normalized/resolved metadata when available (e.g., resolved file path).
        output = tool_fn(**arguments)
        return ExecutionOutcome(executed=True, policy=policy, output=output)


def run_demo() -> None:
    """Demonstrate static rules, dynamic thresholds, scope checks, and rate limits."""
    fake_now = 1000.0

    def fake_clock() -> float:
        return fake_now

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "notes.txt").write_text("Q3 support runbook: refund SLA is 24h.", encoding="utf-8")
        (workspace / ".env").write_text("DATABASE_URL=postgres://secret", encoding="utf-8")

        scope = CapabilityScope(
            allowed_workspace_dir=workspace,
            allowed_http_domains=("docs.python.org", "status.acme.example"),
            internal_email_domains=("acme.internal",),
        )
        engine = PolicyEngine(scope=scope, rate_limiter=SlidingWindowRateLimiter(clock=fake_clock))

        # Static policies
        engine.register_static("get_order", PolicyDecision.AUTO, RateLimitRule(max_calls=20, window_seconds=60))
        engine.register_static("drop_table", PolicyDecision.DENY)

        # Dynamic policies
        engine.register_dynamic("issue_refund", evaluate_refund_policy, RateLimitRule(max_calls=5, window_seconds=300))
        engine.register_dynamic("read_workspace_file", evaluate_file_read_policy)
        engine.register_dynamic("http_fetch", evaluate_http_fetch_policy)
        engine.register_dynamic("query_orders_db", evaluate_db_query_policy)
        engine.register_dynamic("send_email", evaluate_send_email_policy, RateLimitRule(max_calls=3, window_seconds=60))

        executor = GuardedToolExecutor(engine)
        executor.register_tool("get_order", lambda order_id: {"order_id": order_id, "status": "shipped", "total_usd": 42.0})
        executor.register_tool("issue_refund", lambda order_id, amount_usd: {"order_id": order_id, "refunded_usd": amount_usd})
        executor.register_tool("read_workspace_file", lambda path: (workspace / path).read_text(encoding="utf-8"))
        executor.register_tool("http_fetch", lambda url: {"url": url, "status": 200, "body_preview": "Python 3 docs..."})
        executor.register_tool("query_orders_db", lambda sql, limit=50: {"rows_returned": limit, "sql": sql})
        executor.register_tool("send_email", lambda to, subject: {"sent_to": to, "subject": subject})

        session = "sess_support_42"
        print("=== Day 22: Tool Execution Policy Engine Demo ===\n")

        cases: list[tuple[str, str, dict[str, Any], PolicyDecision]] = [
            ("1. Unknown tool (default-deny)", "shell_exec", {"cmd": "rm -rf /"}, PolicyDecision.DENY),
            ("2. Explicitly denied static tool", "drop_table", {"table": "orders"}, PolicyDecision.DENY),
            ("3. Read-only static tool", "get_order", {"order_id": "ORD-1001"}, PolicyDecision.AUTO),
            ("4. Small refund ($25 <= $50)", "issue_refund", {"order_id": "ORD-1001", "amount_usd": 25.0}, PolicyDecision.AUTO),
            ("5. Medium refund ($180 in $50-$500)", "issue_refund", {"order_id": "ORD-1002", "amount_usd": 180.0}, PolicyDecision.CONFIRM),
            ("6. Large refund ($1,200 > $500)", "issue_refund", {"order_id": "ORD-1003", "amount_usd": 1200.0}, PolicyDecision.DENY),
            ("7. Negative refund (-$10)", "issue_refund", {"order_id": "ORD-1004", "amount_usd": -10.0}, PolicyDecision.DENY),
            ("8. File read inside workspace", "read_workspace_file", {"path": "notes.txt"}, PolicyDecision.AUTO),
            ("9. File path traversal (../)", "read_workspace_file", {"path": "../../etc/passwd"}, PolicyDecision.DENY),
            ("10. Blocked sensitive file (.env)", "read_workspace_file", {"path": ".env"}, PolicyDecision.DENY),
            ("11. Allowed HTTPS domain", "http_fetch", {"url": "https://docs.python.org/3/"}, PolicyDecision.AUTO),
            ("12. Spoofed subdomain attack", "http_fetch", {"url": "https://docs.python.org.attacker.example/"}, PolicyDecision.DENY),
            ("13. Cloud metadata SSRF attempt", "http_fetch", {"url": "http://169.254.169.254/latest/meta-data/"}, PolicyDecision.DENY),
            ("14. SQL write injection attempt", "query_orders_db", {"sql": "SELECT * FROM orders; DROP TABLE orders", "limit": 10}, PolicyDecision.DENY),
            ("15. Bulk DB export (500 rows)", "query_orders_db", {"sql": "SELECT id, total FROM orders", "limit": 500}, PolicyDecision.CONFIRM),
            ("16. External email confirmation", "send_email", {"to": "customer@example.com", "subject": "Update"}, PolicyDecision.CONFIRM),
        ]

        for label, tool, args, expected in cases:
            outcome = executor.execute(session, tool, args)
            status = "EXECUTED" if outcome.executed else outcome.policy.decision.value.upper()
            print(f"{label}")
            print(f"   Decision: {outcome.policy.decision.value.upper():7s} | Executed: {str(outcome.executed):5s} | Rule: {outcome.policy.rule_name}")
            print(f"   Reason:   {outcome.policy.reason}")
            if outcome.executed:
                print(f"   Output:   {outcome.output}")
            print()
            assert outcome.policy.decision == expected, f"Expected {expected}, got {outcome.policy.decision} for {label}"
            assert outcome.executed == (expected == PolicyDecision.AUTO)

        # Demonstrate sliding-window rate limit on send_email (already 1 call used in case 16)
        print("--- 17. Sliding-Window Rate Limit Check (send_email: 3 calls / 60s) ---")
        r2 = executor.execute(session, "send_email", {"to": "ops@acme.internal", "subject": "Alert 2"})
        r3 = executor.execute(session, "send_email", {"to": "ops@acme.internal", "subject": "Alert 3"})
        r4 = executor.execute(session, "send_email", {"to": "ops@acme.internal", "subject": "Alert 4 (should rate-limit)"})
        print(f"   Call 2: {r2.policy.decision.value.upper()} (executed={r2.executed})")
        print(f"   Call 3: {r3.policy.decision.value.upper()} (executed={r3.executed})")
        print(f"   Call 4: {r4.policy.decision.value.upper()} | {r4.policy.reason} | retry_after={r4.policy.retry_after_seconds}s")
        assert r2.executed and r3.executed
        assert r4.policy.decision == PolicyDecision.DENY and r4.policy.rule_name == "rate_limit:send_email"

        # Advance simulated clock by 61 seconds so the sliding window expires
        fake_now += 61.0
        r5 = executor.execute(session, "send_email", {"to": "ops@acme.internal", "subject": "Alert 5 after window"})
        print(f"   Call 5 (+61s): {r5.policy.decision.value.upper()} (executed={r5.executed})\n")
        assert r5.executed

        print("All 17 policy and rate-limit checks passed.")


if __name__ == "__main__":
    run_demo()
