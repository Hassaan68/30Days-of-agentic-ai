"""Production Incident & Operations Assistant Agent orchestrating all 30-day pillars."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable

from audit import AuditLog
from guardrails import InputGuardrail, OutputGuardrail
from memory import LongTermMemory, ShortTermMemory
from models import (
    AgentRunResult,
    PlanStep,
    PolicyAction,
    ToolCall,
    ToolResult,
    canonical_payload_hash,
)
from policy_and_hitl import (
    ApprovalValidationError,
    HITLApprovalGate,
    TOCTOUTamperingError,
    ToolPolicyEngine,
)
from tools import MCPToolRegistry
from tracer import Tracer

# Token pricing estimate (gpt-4o-mini-like rates, USD per 1M tokens)
INPUT_RATE_PER_1M = 0.15
OUTPUT_RATE_PER_1M = 0.60


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * INPUT_RATE_PER_1M + (
        output_tokens / 1_000_000
    ) * OUTPUT_RATE_PER_1M


class IncidentOpsAgent:
    """Autonomous incident assistant with planning, MCP tools, memory, policies, HITL, audit, and tracing."""

    SYSTEM_PROMPT = (
        "You are a production Incident & Operations Assistant. "
        "Always inspect health and error logs before proposing mutations. "
        "Respect tool execution policies and human approvals. Never reveal secrets."
    )

    def __init__(
        self,
        *,
        registry: MCPToolRegistry,
        long_term_memory: LongTermMemory,
        policy_engine: ToolPolicyEngine,
        hitl_gate: HITLApprovalGate,
        audit_log: AuditLog,
        demo: bool = True,
        input_guardrail: InputGuardrail | None = None,
        output_guardrail: OutputGuardrail | None = None,
    ) -> None:
        self.registry = registry
        self.long_term_memory = long_term_memory
        self.policy_engine = policy_engine
        self.hitl_gate = hitl_gate
        self.audit_log = audit_log
        self.input_guardrail = input_guardrail or InputGuardrail()
        self.output_guardrail = output_guardrail or OutputGuardrail()

        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        self.demo = demo or (not self.api_key and not is_local)

    def run(
        self,
        user_request: str,
        *,
        tracer: Tracer | None = None,
        tamper_hook: Callable[[ToolCall], None] | None = None,
        inject_secret_for_test: bool = False,
    ) -> AgentRunResult:
        """Execute a full end-to-end agent run with tracing and audit receipts."""
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        trace = tracer or Tracer()
        short_memory = ShortTermMemory(self.SYSTEM_PROMPT, max_tokens=3200)

        with trace.span("agent_run", {"run_id": run_id, "mode": "demo" if self.demo else "live"}):
            # 1. Input Guardrail
            with trace.span("guardrail_input") as sp:
                in_check = self.input_guardrail.check(user_request)
                sp.attributes["allowed"] = in_check.allowed
                if not in_check.allowed:
                    self.audit_log.record(
                        run_id=run_id,
                        event_type="blocked_guardrail",
                        actor="input_guardrail",
                        outcome="blocked",
                        details={
                            "reason": in_check.reason,
                            "rules": list(in_check.triggered_rules),
                        },
                    )
                    return AgentRunResult(
                        run_id=run_id,
                        user_request=user_request,
                        status="blocked",
                        final_response=f"Request rejected by input guardrail: {in_check.reason}",
                        total_input_tokens=trace.total_input_tokens,
                        total_output_tokens=trace.total_output_tokens,
                        total_cost_usd=trace.total_cost_usd,
                    )

            clean_request = in_check.sanitized_text
            short_memory.add("user", clean_request)

            # 2. MCP Tool Discovery
            with trace.span("mcp_tools_list") as sp:
                discovered_tools = self.registry.discover_tools()
                sp.attributes["tool_count"] = len(discovered_tools)

            # 3. Long-Term Memory Recall
            with trace.span("memory_recall") as sp:
                memories = self.long_term_memory.recall(clean_request, top_k=2)
                recalled_titles = [f"[{m.kind}] {m.title}" for m in memories]
                sp.attributes["hits"] = len(memories)
                if memories:
                    mem_context = "\n".join(
                        f"- {m.title}: {m.content}" for m in memories
                    )
                    short_memory.add(
                        "system",
                        f"Recalled long-term incident memory:\n{mem_context}",
                    )

            # 4. Planning & Task Decomposition
            with trace.span("plan_decomposition") as sp:
                plan, planned_calls = self._build_plan(
                    clean_request, discovered_tools, memories, sp
                )
                sp.attributes["steps"] = len(plan)

            # 5. Controlled Tool Execution Loop (Policy -> HITL -> MCP -> Audit)
            tool_results: list[ToolResult] = []
            for call in planned_calls:
                res = self._execute_governed_tool(
                    run_id=run_id,
                    call=call,
                    trace=trace,
                    tamper_hook=tamper_hook,
                )
                tool_results.append(res)
                short_memory.add(
                    "assistant",
                    f"Tool {call.name} -> ok={res.ok} output={json.dumps(res.output)}",
                )

            # 6. Final Synthesis
            with trace.span("synthesize_response") as sp:
                raw_response = self._synthesize(
                    clean_request, plan, tool_results, recalled_titles, sp
                )
                if inject_secret_for_test:
                    raw_response += " (debug token: sk-liveSecretToken1234567890abcd)"

            # 7. Output Guardrail (Secret Redaction)
            with trace.span("guardrail_output") as sp:
                out_check = self.output_guardrail.check_and_sanitize(raw_response)
                sp.attributes["redacted"] = bool(out_check.triggered_rules)
                if out_check.triggered_rules:
                    self.audit_log.record(
                        run_id=run_id,
                        event_type="output_redacted",
                        actor="output_guardrail",
                        outcome="sanitized",
                        details={"rules": list(out_check.triggered_rules)},
                    )

            # 8. Persist Durable Learning if Incident Was Remediated
            if any(r.tool_name == "rollback_deployment" and r.ok for r in tool_results):
                with trace.span("memory_remember"):
                    self.long_term_memory.remember(
                        kind="incident",
                        title=f"Resolved rollback in {run_id}",
                        content=(
                            f"Request '{clean_request}' resolved via approved rollback; "
                            f"service restored to healthy state."
                        ),
                        tags="resolved rollback checkout-api",
                    )

            return AgentRunResult(
                run_id=run_id,
                user_request=user_request,
                status="completed",
                final_response=out_check.sanitized_text,
                plan=plan,
                tool_results=tool_results,
                recalled_memories=recalled_titles,
                total_input_tokens=trace.total_input_tokens,
                total_output_tokens=trace.total_output_tokens,
                total_cost_usd=trace.total_cost_usd,
            )

    def _build_plan(
        self,
        request: str,
        discovered_tools: list[dict[str, Any]],
        memories: list[Any],
        span: Any,
    ) -> tuple[list[PlanStep], list[ToolCall]]:
        """Decompose the user request into explicit steps and tool calls."""
        lower = request.lower()

        if not self.demo:
            live_plan = self._plan_with_live_llm(request, discovered_tools, span)
            if live_plan is not None:
                return live_plan

        # Deterministic offline planner for `--demo` mode and reproducible evals.
        in_tok = 420 + len(memories) * 90
        out_tok = 115
        span.input_tokens = in_tok
        span.output_tokens = out_tok
        span.cost_usd = _cost_usd(in_tok, out_tok)

        if "purge" in lower and "database" in lower:
            plan = [
                PlanStep(1, "Attempt requested database purge (subject to tool policy)", "purge_database"),
            ]
            calls = [
                ToolCall(
                    call_id="call_purge_1",
                    name="purge_database",
                    arguments={"database": "orders_prod"},
                    scope="prod",
                )
            ]
            return plan, calls

        service = "payments-worker" if "payments-worker" in lower else "checkout-api"
        steps: list[PlanStep] = [
            PlanStep(1, f"Inspect live health metrics for {service}", "get_service_health"),
            PlanStep(2, f"Query recent error logs for {service}", "query_error_logs"),
        ]
        calls: list[ToolCall] = [
            ToolCall("call_1", "get_service_health", {"service": service}, "prod"),
            ToolCall("call_2", "query_error_logs", {"service": service, "limit": 5}, "prod"),
        ]

        if any(word in lower for word in ("incident", "502", "ticket", "rollback", "mitigate")):
            steps.append(
                PlanStep(3, f"Open SEV2 incident ticket documenting {service} errors", "create_incident_ticket")
            )
            calls.append(
                ToolCall(
                    "call_3",
                    "create_incident_ticket",
                    {
                        "service": service,
                        "severity": "sev2",
                        "summary": f"{service} elevated 502 error rate (DB connection pool exhaustion on v2.14.0)",
                    },
                    "prod",
                )
            )

        if any(word in lower for word in ("rollback", "mitigate", "remediate", "fix")):
            steps.append(
                PlanStep(4, f"Roll back {service} to last known-good version v2.13.9 (HITL gated)", "rollback_deployment")
            )
            calls.append(
                ToolCall(
                    "call_4",
                    "rollback_deployment",
                    {"service": service, "target_version": "v2.13.9"},
                    "prod",
                )
            )

        if "restart" in lower:
            steps.append(
                PlanStep(len(steps) + 1, f"Restart {service} (HITL gated)", "restart_service")
            )
            calls.append(
                ToolCall(
                    f"call_{len(calls) + 1}",
                    "restart_service",
                    {"service": service, "force": "force" in lower},
                    "prod",
                )
            )

        return steps, calls

    def _plan_with_live_llm(
        self,
        request: str,
        discovered_tools: list[dict[str, Any]],
        span: Any,
    ) -> tuple[list[PlanStep], list[ToolCall]] | None:
        """Ask an OpenAI-compatible model to return a JSON plan of tool calls."""
        tool_summary = [
            {"name": t["name"], "description": t["description"], "schema": t["inputSchema"]}
            for t in discovered_tools
        ]
        prompt = (
            "Return ONLY valid JSON with key 'steps': a list of objects "
            '{"description": str, "tool": str, "arguments": object}. '
            f"Available tools: {json.dumps(tool_summary)}\nUser request: {request}"
        )
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            usage = data.get("usage", {})
            span.input_tokens = int(usage.get("prompt_tokens", 350))
            span.output_tokens = int(usage.get("completion_tokens", 120))
            span.cost_usd = _cost_usd(span.input_tokens, span.output_tokens)

            parsed = json.loads(data["choices"][0]["message"]["content"])
            steps: list[PlanStep] = []
            calls: list[ToolCall] = []
            for idx, item in enumerate(parsed.get("steps", []), start=1):
                tool_name = str(item["tool"])
                steps.append(PlanStep(idx, str(item["description"]), tool_name))
                calls.append(
                    ToolCall(
                        call_id=f"live_call_{idx}",
                        name=tool_name,
                        arguments=dict(item.get("arguments") or {}),
                        scope="prod",
                    )
                )
            return steps, calls
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def _execute_governed_tool(
        self,
        *,
        run_id: str,
        call: ToolCall,
        trace: Tracer,
        tamper_hook: Callable[[ToolCall], None] | None,
    ) -> ToolResult:
        """Run Policy -> HITL Approval -> TOCTOU Hash Verify -> MCP Tool Call -> Audit."""
        args_hash = canonical_payload_hash(call.name, call.arguments, call.scope)

        # Policy evaluation
        with trace.span("policy_check", {"tool": call.name}) as sp:
            decision = self.policy_engine.evaluate(run_id, call)
            sp.attributes["action"] = decision.action.value
            sp.attributes["rule"] = decision.rule_id

        if decision.action == PolicyAction.DENY:
            self.audit_log.record(
                run_id=run_id,
                event_type="denied",
                tool_name=call.name,
                arguments_hash=args_hash,
                actor="policy_engine",
                outcome="denied",
                details={"reason": decision.reason, "rule_id": decision.rule_id},
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                output={"denied": True, "reason": decision.reason},
                error=decision.reason,
            )

        # HITL confirmation when policy requires human approval
        if decision.action == PolicyAction.CONFIRM:
            with trace.span("hitl_approval", {"tool": call.name}) as sp:
                req, grant = self.hitl_gate.request_approval(
                    run_id=run_id,
                    call=call,
                    reason=decision.reason,
                )
                sp.attributes["approved"] = grant.approved
                sp.attributes["approver"] = grant.approver

            if not grant.approved:
                self.audit_log.record(
                    run_id=run_id,
                    event_type="denied",
                    tool_name=call.name,
                    arguments_hash=args_hash,
                    actor=grant.approver,
                    outcome="human_denied",
                    details={"approval_id": req.approval_id, "comment": grant.comment},
                )
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.name,
                    ok=False,
                    output={"denied_by_human": True, "approver": grant.approver},
                    error=f"Operator {grant.approver} denied {call.name}",
                )

            self.audit_log.record(
                run_id=run_id,
                event_type="approved",
                tool_name=call.name,
                arguments_hash=req.payload_hash,
                actor=grant.approver,
                outcome="approved",
                details={"approval_id": req.approval_id, "comment": grant.comment},
            )

            # Optional hook used by security evals to test TOCTOU tampering after approval.
            if tamper_hook is not None:
                tamper_hook(call)

            try:
                self.hitl_gate.verify_grant_before_execution(grant, call)
            except TOCTOUTamperingError as exc:
                tampered_hash = canonical_payload_hash(call.name, call.arguments, call.scope)
                self.audit_log.record(
                    run_id=run_id,
                    event_type="toctou_rejected",
                    tool_name=call.name,
                    arguments_hash=tampered_hash,
                    actor="hitl_gate",
                    outcome="blocked_tampering",
                    details={
                        "approved_hash": grant.payload_hash,
                        "execution_hash": tampered_hash,
                        "error": str(exc),
                    },
                )
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.name,
                    ok=False,
                    output={"toctou_blocked": True},
                    error=str(exc),
                )
            except ApprovalValidationError as exc:
                self.audit_log.record(
                    run_id=run_id,
                    event_type="denied",
                    tool_name=call.name,
                    arguments_hash=args_hash,
                    actor="hitl_gate",
                    outcome="invalid_grant",
                    details={"error": str(exc)},
                )
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.name,
                    ok=False,
                    output={"approval_invalid": True},
                    error=str(exc),
                )

        # Execute tool via MCP JSON-RPC interface
        with trace.span("mcp_tool_call", {"tool": call.name}) as sp:
            ok, output = self.registry.invoke_tool(call.name, call.arguments)
            sp.status = "ok" if ok else "error"

        self.audit_log.record(
            run_id=run_id,
            event_type="executed" if ok else "failed",
            tool_name=call.name,
            arguments_hash=canonical_payload_hash(call.name, call.arguments, call.scope),
            actor="mcp_tool_registry",
            outcome="success" if ok else "error",
            details={"output_keys": sorted(output.keys())},
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            ok=ok,
            output=output,
            error=None if ok else str(output.get("error", "tool error")),
        )

    def _synthesize(
        self,
        request: str,
        plan: list[PlanStep],
        results: list[ToolResult],
        recalled: list[str],
        span: Any,
    ) -> str:
        """Assemble a grounded summary from tool observations and policy outcomes."""
        in_tok = 520 + len(results) * 80
        out_tok = 140
        span.input_tokens = in_tok
        span.output_tokens = out_tok
        span.cost_usd = _cost_usd(in_tok, out_tok)

        lines = [f"Incident summary for: '{request}'"]
        if recalled:
            lines.append(f"Recalled memory: {', '.join(recalled)}")
        lines.append(f"Executed {len(plan)} planned step(s):")
        for res in results:
            if res.ok:
                lines.append(f"- [OK] {res.tool_name}: {json.dumps(res.output)}")
            else:
                lines.append(f"- [BLOCKED/ERROR] {res.tool_name}: {res.error}")
        return "\n".join(lines)
