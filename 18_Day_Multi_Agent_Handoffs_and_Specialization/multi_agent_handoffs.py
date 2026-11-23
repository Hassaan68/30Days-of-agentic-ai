"""Multi-agent specialization with strict tool boundaries and explicit handoffs.

Implements three specialized agents (`triage`, `tech_support`, `billing`) managed
by a handoff runtime that enforces:
  - Scoped tools per specialist (an agent cannot call another specialist's tools)
  - Explicit `HandoffPacket` transfers with verified facts and bounded context
  - Max-handoff budget and ping-pong cycle detection (A -> B -> A escalation)

Usage:
  # Run offline deterministic demo (multi-hop resolution + ping-pong loop guard)
  python3 18_Day_Multi_Agent_Handoffs_and_Specialization/multi_agent_handoffs.py --demo

  # Run against an OpenAI-compatible API (OpenAI, Ollama, OpenRouter, etc.)
  OPENAI_API_KEY=... python3 18_Day_Multi_Agent_Handoffs_and_Specialization/multi_agent_handoffs.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Callable
import urllib.error
import urllib.request

DEFAULT_QUERY = (
    "Customer cust-42 here. We hit 504 errors on the webhooks API earlier today and "
    "now I see two $199 charges on my account. Can you confirm what happened and "
    "refund the duplicate charge?"
)

# ---------------------------------------------------------------------------
# Simulated backend state & strictly scoped domain tools
# ---------------------------------------------------------------------------

ACCOUNTS: dict[str, dict[str, Any]] = {
    "cust-42": {
        "customer_id": "cust-42",
        "company": "Acme DevOps",
        "plan": "Pro",
        "status": "active",
        "owner_email": "ops@acmedevops.example",
    }
}

SERVICE_STATUS: dict[str, dict[str, Any]] = {
    "webhooks-api": {
        "service": "webhooks-api",
        "status": "degraded",
        "incident_id": "INC-408",
        "summary": "Elevated 504 Gateway Timeouts between 14:00Z and 14:25Z.",
    },
    "auth-api": {
        "service": "auth-api",
        "status": "operational",
        "incident_id": None,
        "summary": "All authentication nodes healthy.",
    },
}

ERROR_LOGS: dict[str, list[str]] = {
    "cust-42": [
        "14:12:03Z POST /v1/billing/charge -> 504 Gateway Timeout (upstream timeout; retry triggered)",
        "14:12:19Z POST /v1/billing/charge -> 200 OK (created invoice inv-9002 without idempotency key)",
    ]
}


def make_initial_invoices() -> dict[str, list[dict[str, Any]]]:
    return {
        "cust-42": [
            {
                "invoice_id": "inv-9001",
                "amount_usd": 199.00,
                "status": "paid",
                "note": "Monthly Pro subscription renewal",
            },
            {
                "invoice_id": "inv-9002",
                "amount_usd": 199.00,
                "status": "paid",
                "note": "Duplicate charge posted 16s after inv-9001 during INC-408",
            },
        ]
    }


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def build_tool_registry(invoices_db: dict[str, list[dict[str, Any]]]) -> dict[str, ToolDef]:
    def lookup_account(args: dict[str, Any]) -> dict[str, Any]:
        cid = str(args.get("customer_id", ""))
        acct = ACCOUNTS.get(cid)
        if not acct:
            return {"error": f"Account '{cid}' not found."}
        return {"account": acct}

    def check_service_status(args: dict[str, Any]) -> dict[str, Any]:
        svc = str(args.get("service_name", ""))
        info = SERVICE_STATUS.get(svc)
        if not info:
            return {"error": f"Unknown service '{svc}'", "known": sorted(SERVICE_STATUS)}
        return info

    def get_error_logs(args: dict[str, Any]) -> dict[str, Any]:
        cid = str(args.get("customer_id", ""))
        return {"customer_id": cid, "entries": ERROR_LOGS.get(cid, [])}

    def rotate_api_key(args: dict[str, Any]) -> dict[str, Any]:
        cid = str(args.get("customer_id", ""))
        return {"customer_id": cid, "status": "rotated", "new_key_prefix": "sk_live_9f2a..."}

    def get_invoices(args: dict[str, Any]) -> dict[str, Any]:
        cid = str(args.get("customer_id", ""))
        return {"customer_id": cid, "invoices": invoices_db.get(cid, [])}

    def issue_refund(args: dict[str, Any]) -> dict[str, Any]:
        inv_id = str(args.get("invoice_id", ""))
        reason = str(args.get("reason", "duplicate charge"))
        for cid, inv_list in invoices_db.items():
            for inv in inv_list:
                if inv["invoice_id"] == inv_id:
                    if inv["status"] == "refunded":
                        return {"status": "already_refunded", "invoice_id": inv_id}
                    inv["status"] = "refunded"
                    return {
                        "status": "refunded",
                        "customer_id": cid,
                        "invoice_id": inv_id,
                        "amount_usd": inv["amount_usd"],
                        "refund_id": f"rf_{inv_id}",
                        "reason": reason,
                    }
        return {"error": f"Invoice '{inv_id}' not found."}

    tools = [
        ToolDef(
            name="lookup_account",
            description="Look up basic customer account metadata by customer_id.",
            parameters={
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
            handler=lookup_account,
        ),
        ToolDef(
            name="check_service_status",
            description="Check current health and active incidents for a platform service.",
            parameters={
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
            handler=check_service_status,
        ),
        ToolDef(
            name="get_error_logs",
            description="Fetch recent API error log entries for a customer_id.",
            parameters={
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
            handler=get_error_logs,
        ),
        ToolDef(
            name="rotate_api_key",
            description="Rotate a compromised or broken API key for a customer_id.",
            parameters={
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
            handler=rotate_api_key,
        ),
        ToolDef(
            name="get_invoices",
            description="List invoices and payment status for a customer_id.",
            parameters={
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
            handler=get_invoices,
        ),
        ToolDef(
            name="issue_refund",
            description="Refund a paid invoice by invoice_id with a documented reason.",
            parameters={
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["invoice_id", "reason"],
            },
            handler=issue_refund,
        ),
    ]
    return {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# Specialist definitions & explicit handoff protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentSpec:
    name: str
    role_summary: str
    system_prompt: str
    allowed_tools: tuple[str, ...]
    allowed_handoff_targets: tuple[str, ...]


AGENTS: dict[str, AgentSpec] = {
    "triage": AgentSpec(
        name="triage",
        role_summary="Front-door router that verifies account identity and picks the first specialist.",
        system_prompt=(
            "You are the Triage Agent. Your job is strictly to verify the customer account "
            "using `lookup_account` and hand off to `tech_support` (if technical/API failures "
            "need diagnosis) or `billing` (for pure invoice/refund questions). Never invent "
            "root causes or promise refunds yourself."
        ),
        allowed_tools=("lookup_account",),
        allowed_handoff_targets=("tech_support", "billing"),
    ),
    "tech_support": AgentSpec(
        name="tech_support",
        role_summary="Diagnoses service incidents, inspects error logs, and manages API keys.",
        system_prompt=(
            "You are the Technical Support Specialist. Use `check_service_status`, "
            "`get_error_logs`, or `rotate_api_key` to diagnose technical issues. You CANNOT "
            "issue refunds. Once you confirm a technical failure caused a billing problem, "
            "call `handoff_to_agent` with target='billing' and attach your verified facts."
        ),
        allowed_tools=("check_service_status", "get_error_logs", "rotate_api_key"),
        allowed_handoff_targets=("billing",),
    ),
    "billing": AgentSpec(
        name="billing",
        role_summary="Inspects invoices and issues verified refunds.",
        system_prompt=(
            "You are the Billing Specialist. Use `get_invoices` and `issue_refund` to resolve "
            "billing requests. Rely on verified facts in any incoming handoff packet. Once "
            "resolved, reply directly to the customer with a clear summary."
        ),
        allowed_tools=("get_invoices", "issue_refund"),
        allowed_handoff_targets=("tech_support",),
    ),
}


@dataclass
class HandoffPacket:
    """Structured context object transferred explicitly between specialists."""

    from_agent: str
    to_agent: str
    reason: str
    summary: str
    customer_id: str
    verified_facts: list[str] = field(default_factory=list)
    requested_action: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def make_handoff_tool_schema(agent: AgentSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "handoff_to_agent",
            "description": (
                "Transfer control to another specialist with a structured context packet. "
                f"Allowed targets from {agent.name}: {list(agent.allowed_handoff_targets)}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to_agent": {
                        "type": "string",
                        "enum": list(agent.allowed_handoff_targets),
                    },
                    "reason": {"type": "string"},
                    "summary": {"type": "string"},
                    "customer_id": {"type": "string"},
                    "verified_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "requested_action": {"type": "string"},
                },
                "required": [
                    "to_agent",
                    "reason",
                    "summary",
                    "customer_id",
                    "verified_facts",
                    "requested_action",
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# Model backends (Offline Demo Mock + Live OpenAI-Compatible API)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelReply:
    content: str | None
    tool_calls: list[ToolCall]


class DemoScenarioLLM:
    """Deterministic scriptable mock LLM demonstrating handoffs and loop guards."""

    def __init__(self, scenario: str = "multi_hop") -> None:
        self.scenario = scenario
        self._agent_turns: dict[str, int] = {}

    def chat(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelReply:
        del messages, tools
        turn = self._agent_turns.get(agent_name, 0) + 1
        self._agent_turns[agent_name] = turn

        if self.scenario == "ping_pong":
            if agent_name == "triage":
                return ModelReply(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="pp_1",
                            name="handoff_to_agent",
                            arguments={
                                "to_agent": "tech_support",
                                "reason": "Possible API billing mismatch",
                                "summary": "Customer reports unexpected invoice after webhook warning.",
                                "customer_id": "cust-42",
                                "verified_facts": ["Account cust-42 is active."],
                                "requested_action": "Investigate webhook logs.",
                            },
                        )
                    ],
                )
            if agent_name == "tech_support":
                return ModelReply(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="pp_2",
                            name="handoff_to_agent",
                            arguments={
                                "to_agent": "billing",
                                "reason": "Looks like a billing question",
                                "summary": "Passing to billing for invoice review.",
                                "customer_id": "cust-42",
                                "verified_facts": [],
                                "requested_action": "Review invoice.",
                            },
                        )
                    ],
                )
            if agent_name == "billing":
                # Deliberately bounce back to tech_support to trigger ping-pong guard!
                return ModelReply(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="pp_3",
                            name="handoff_to_agent",
                            arguments={
                                "to_agent": "tech_support",
                                "reason": "Need engineering confirmation first",
                                "summary": "Bouncing back to tech_support.",
                                "customer_id": "cust-42",
                                "verified_facts": [],
                                "requested_action": "Check logs again.",
                            },
                        )
                    ],
                )

        # Default scenario: multi_hop (triage -> tech_support -> billing)
        if agent_name == "triage":
            if turn == 1:
                return ModelReply(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="t_1",
                            name="lookup_account",
                            arguments={"customer_id": "cust-42"},
                        )
                    ],
                )
            return ModelReply(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="t_2",
                        name="handoff_to_agent",
                        arguments={
                            "to_agent": "tech_support",
                            "reason": "Technical 504 errors need verification before billing refund.",
                            "summary": "Acme DevOps (cust-42, Pro) reported 504 errors and duplicate $199 charges.",
                            "customer_id": "cust-42",
                            "verified_facts": [
                                "Account cust-42 (Acme DevOps) is active on the Pro plan."
                            ],
                            "requested_action": (
                                "Verify webhooks-api incident and error logs, then hand off to billing if duplicate confirmed."
                            ),
                        },
                    )
                ],
            )

        if agent_name == "tech_support":
            if turn == 1:
                return ModelReply(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="ts_1",
                            name="check_service_status",
                            arguments={"service_name": "webhooks-api"},
                        ),
                        ToolCall(
                            id="ts_2",
                            name="get_error_logs",
                            arguments={"customer_id": "cust-42"},
                        ),
                    ],
                )
            return ModelReply(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="ts_3",
                        name="handoff_to_agent",
                        arguments={
                            "to_agent": "billing",
                            "reason": "Confirmed duplicate invoice caused by INC-408 504 retry.",
                            "summary": "Incident INC-408 caused a retry without an idempotency key, creating duplicate inv-9002.",
                            "customer_id": "cust-42",
                            "verified_facts": [
                                "webhooks-api had incident INC-408 (504 timeouts 14:00Z-14:25Z).",
                                "Error logs confirm 14:12:03Z 504 retry created duplicate invoice inv-9002.",
                            ],
                            "requested_action": "Verify invoices for cust-42 and refund duplicate inv-9002.",
                        },
                    )
                ],
            )

        if agent_name == "billing":
            if turn == 1:
                return ModelReply(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="b_1",
                            name="get_invoices",
                            arguments={"customer_id": "cust-42"},
                        )
                    ],
                )
            if turn == 2:
                return ModelReply(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="b_2",
                            name="issue_refund",
                            arguments={
                                "invoice_id": "inv-9002",
                                "reason": "Duplicate charge triggered by INC-408 504 retry",
                            },
                        )
                    ],
                )
            return ModelReply(
                content=(
                    "We verified what happened and refunded your duplicate charge:\n"
                    "- Root cause (Technical Support): `webhooks-api` experienced 504 timeouts "
                    "during incident INC-408 (14:00Z–14:25Z), and a retry without an idempotency "
                    "key created duplicate invoice `inv-9002`.\n"
                    "- Resolution (Billing): Issued a full $199.00 refund (`rf_inv-9002`) for "
                    "`inv-9002`. Your primary subscription invoice `inv-9001` remains active."
                ),
                tool_calls=[],
            )

        return ModelReply(content="Escalating to human operator.", tool_calls=[])


class OpenAICompatibleBackend:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelReply:
        del agent_name
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.1,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc

        msg = data["choices"][0]["message"]
        calls: list[ToolCall] = []
        for raw in msg.get("tool_calls") or []:
            raw_args = raw["function"].get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            calls.append(
                ToolCall(id=raw["id"], name=raw["function"]["name"], arguments=args)
            )
        return ModelReply(content=msg.get("content"), tool_calls=calls)


# ---------------------------------------------------------------------------
# Handoff Runtime with Scoping & Ping-Pong Prevention
# ---------------------------------------------------------------------------

@dataclass
class HandoffRunResult:
    final_agent: str
    final_response: str
    escalated: bool
    handoffs: list[HandoffPacket]
    visited_path: list[str]


class HandoffRuntime:
    """Coordinates specialist agents with strict tool scoping and loop protection."""

    def __init__(
        self,
        agents: dict[str, AgentSpec],
        tool_registry: dict[str, ToolDef],
        llm: DemoScenarioLLM | OpenAICompatibleBackend,
        max_handoffs: int = 3,
        max_steps_per_agent: int = 5,
    ) -> None:
        self.agents = agents
        self.tool_registry = tool_registry
        self.llm = llm
        self.max_handoffs = max_handoffs
        self.max_steps_per_agent = max_steps_per_agent

    def _build_specialist_messages(
        self,
        agent: AgentSpec,
        user_query: str,
        incoming_packet: HandoffPacket | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": user_query},
        ]
        if incoming_packet is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Incoming structured handoff packet from previous specialist:\n"
                        f"{incoming_packet.to_json()}"
                    ),
                }
            )
        return messages

    def _tools_for_agent(self, agent: AgentSpec) -> list[dict[str, Any]]:
        schemas = [
            self.tool_registry[name].to_openai_schema()
            for name in agent.allowed_tools
        ]
        if agent.allowed_handoff_targets:
            schemas.append(make_handoff_tool_schema(agent))
        return schemas

    def run(self, user_query: str, entry_agent: str = "triage") -> HandoffRunResult:
        active_name = entry_agent
        incoming_packet: HandoffPacket | None = None
        handoffs: list[HandoffPacket] = []
        visited_path: list[str] = [entry_agent]

        while True:
            agent = self.agents[active_name]
            print(
                f"\n>>> Active Specialist: [{agent.name}] "
                f"(allowed tools: {list(agent.allowed_tools)})"
            )
            messages = self._build_specialist_messages(agent, user_query, incoming_packet)
            tools = self._tools_for_agent(agent)

            transferred = False
            for step in range(1, self.max_steps_per_agent + 1):
                reply = self.llm.chat(agent.name, messages, tools)
                if not reply.tool_calls:
                    return HandoffRunResult(
                        final_agent=agent.name,
                        final_response=reply.content or "(empty response)",
                        escalated=False,
                        handoffs=handoffs,
                        visited_path=visited_path,
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": reply.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in reply.tool_calls
                        ],
                    }
                )

                for tc in reply.tool_calls:
                    if tc.name == "handoff_to_agent":
                        target = str(tc.arguments.get("to_agent", ""))
                        if target not in agent.allowed_handoff_targets:
                            obs = json.dumps(
                                {
                                    "error": "invalid_handoff_target",
                                    "message": (
                                        f"Agent '{agent.name}' cannot hand off to '{target}'. "
                                        f"Allowed: {list(agent.allowed_handoff_targets)}"
                                    ),
                                }
                            )
                            messages.append(
                                {"role": "tool", "tool_call_id": tc.id, "content": obs}
                            )
                            continue

                        # Guard 1: Ping-pong / revisit cycle detection
                        if target in visited_path:
                            msg = (
                                f"[LOOP GUARD] Blocked ping-pong handoff "
                                f"{agent.name} -> {target} (already visited: {visited_path}). "
                                "Escalating ticket to human support queue."
                            )
                            print(f"  {msg}")
                            return HandoffRunResult(
                                final_agent=agent.name,
                                final_response=msg,
                                escalated=True,
                                handoffs=handoffs,
                                visited_path=visited_path,
                            )

                        # Guard 2: Max-handoff budget
                        if len(handoffs) >= self.max_handoffs:
                            msg = (
                                f"[BUDGET GUARD] Reached max_handoffs={self.max_handoffs}. "
                                "Escalating ticket to human support queue."
                            )
                            print(f"  {msg}")
                            return HandoffRunResult(
                                final_agent=agent.name,
                                final_response=msg,
                                escalated=True,
                                handoffs=handoffs,
                                visited_path=visited_path,
                            )

                        packet = HandoffPacket(
                            from_agent=agent.name,
                            to_agent=target,
                            reason=str(tc.arguments.get("reason", "")),
                            summary=str(tc.arguments.get("summary", "")),
                            customer_id=str(tc.arguments.get("customer_id", "")),
                            verified_facts=list(tc.arguments.get("verified_facts", [])),
                            requested_action=str(tc.arguments.get("requested_action", "")),
                        )
                        handoffs.append(packet)
                        visited_path.append(target)
                        print(
                            f"  [HANDOFF #{len(handoffs)}] {packet.from_agent} -> {packet.to_agent} "
                            f"| reason={packet.reason!r}"
                        )
                        active_name = target
                        incoming_packet = packet
                        transferred = True
                        break

                    # Strict runtime enforcement of scoped tool boundaries
                    if tc.name not in agent.allowed_tools:
                        obs = json.dumps(
                            {
                                "error": "permission_denied",
                                "message": (
                                    f"Agent '{agent.name}' is not authorized to call '{tc.name}'."
                                ),
                            }
                        )
                        print(f"  [DENY ] {agent.name} attempted forbidden tool '{tc.name}'")
                    else:
                        print(
                            f"  [TOOL ] {agent.name}.{tc.name}({json.dumps(tc.arguments)})"
                        )
                        result_dict = self.tool_registry[tc.name].handler(tc.arguments)
                        obs = json.dumps(result_dict)
                        print(f"          -> {obs}")

                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": obs}
                    )

                if transferred:
                    break
            else:
                return HandoffRunResult(
                    final_agent=agent.name,
                    final_response=(
                        f"Agent '{agent.name}' reached max_steps_per_agent without finishing. "
                        "Escalated to human queue."
                    ),
                    escalated=True,
                    handoffs=handoffs,
                    visited_path=visited_path,
                )


def should_use_demo(force_demo: bool) -> bool:
    if force_demo or os.getenv("AGENT_DEMO_MODE") == "1":
        return True
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-agent handoff runtime with strict tool boundaries."
    )
    parser.add_argument("--demo", action="store_true", help="Run offline demo scenarios.")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Customer request.")
    args = parser.parse_args()

    use_demo = should_use_demo(args.demo)
    if use_demo:
        print("=== Day 18: Multi-Agent Handoffs [DEMO MODE] ===")

        print("\n--- Scenario 1: Multi-Hop Handoff (triage -> tech_support -> billing) ---")
        runtime_1 = HandoffRuntime(
            agents=AGENTS,
            tool_registry=build_tool_registry(make_initial_invoices()),
            llm=DemoScenarioLLM(scenario="multi_hop"),
            max_handoffs=3,
        )
        res_1 = runtime_1.run(args.query)
        print(f"\nPath: {' -> '.join(res_1.visited_path)} | Escalated: {res_1.escalated}")
        print(f"Final Agent: {res_1.final_agent}\n")
        print(res_1.final_response)

        print("\n--- Scenario 2: Ping-Pong Loop Guard (triage -> tech_support -> billing -> tech_support) ---")
        runtime_2 = HandoffRuntime(
            agents=AGENTS,
            tool_registry=build_tool_registry(make_initial_invoices()),
            llm=DemoScenarioLLM(scenario="ping_pong"),
            max_handoffs=3,
        )
        res_2 = runtime_2.run("Ambiguous billing/webhook discrepancy on cust-42.")
        print(f"\nPath: {' -> '.join(res_2.visited_path)} | Escalated: {res_2.escalated}")
        print(res_2.final_response)
        return

    backend = OpenAICompatibleBackend(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    )
    print(f"=== Day 18: Multi-Agent Handoffs [LIVE: {backend.model}] ===")
    runtime = HandoffRuntime(
        agents=AGENTS,
        tool_registry=build_tool_registry(make_initial_invoices()),
        llm=backend,
        max_handoffs=3,
    )
    res = runtime.run(args.query)
    print(f"\nPath: {' -> '.join(res.visited_path)} | Escalated: {res.escalated}")
    print(f"Final Agent ({res.final_agent}):\n{res.final_response}")


if __name__ == "__main__":
    main()
