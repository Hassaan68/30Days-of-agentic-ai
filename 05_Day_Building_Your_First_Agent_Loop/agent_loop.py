"""Day 5: Build your first complete multi-step agent loop with max-step guards and step logging.

Implements the core agent loop:
    model -> tool call -> tool execution -> observation -> next step

Features:
- Self-contained ToolRegistry with strict argument validation.
- Explicit termination conditions: natural completion, `max_steps` limit, and
  consecutive duplicate tool-call detection.
- Structured `StepRecord` trace logging for every iteration (latency, tokens, tool calls).
- Works with any OpenAI-compatible API (OpenAI, Ollama, Groq, OpenRouter) and includes
  an offline `--demo` mode demonstrating both a full multi-step run and a `max_steps` halt.

Examples:
    python3 05_Day_Building_Your_First_Agent_Loop/agent_loop.py --demo
    python3 05_Day_Building_Your_First_Agent_Loop/agent_loop.py "Investigate payments-api errors"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TASK = (
    "Investigate elevated 5xx errors on payments-api: check its live metrics, "
    "calculate its error budget burn against the 99.9% SLO, find the matching runbook, "
    "and identify who is on call to page."
)


# ---------------------------------------------------------------------------
# Simulated Operational Data for Local Tools
# ---------------------------------------------------------------------------
SERVICE_METRICS: dict[str, dict[str, Any]] = {
    "payments-api": {
        "service": "payments-api",
        "owner_team": "payments-core",
        "error_rate_pct": 2.4,
        "p95_latency_ms": 1480,
        "top_error": "UpstreamGatewayTimeout (HTTP 504)",
    },
    "search-api": {
        "service": "search-api",
        "owner_team": "discovery",
        "error_rate_pct": 0.04,
        "p95_latency_ms": 95,
        "top_error": "None",
    },
}

RUNBOOKS: list[dict[str, str]] = [
    {
        "id": "RB-402",
        "title": "Payments Upstream Gateway Timeouts (HTTP 504)",
        "keywords": "payments timeout 504 gateway upstream",
        "action": "Enable circuit breaker fallback to secondary processor and page payments-core on-call.",
    },
    {
        "id": "RB-118",
        "title": "Search Index Replication Lag",
        "keywords": "search index lag replica stale",
        "action": "Pause bulk reindexing job and verify read replica disk IOPS.",
    },
]

ONCALL_ROSTER: dict[str, dict[str, str]] = {
    "payments-core": {"primary": "priya.nair@example.com", "slack": "#inc-payments"},
    "discovery": {"primary": "marco.silva@example.com", "slack": "#inc-search"},
}


# ---------------------------------------------------------------------------
# Tool Registry (self-contained so this day runs standalone)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.to_openai_schema() for spec in self._specs.values()]

    def execute(self, name: str, raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
        spec = self._specs.get(name)
        if spec is None:
            return {"ok": False, "error": f"Unknown tool '{name}'."}

        if isinstance(raw_arguments, str):
            try:
                args = json.loads(raw_arguments) if raw_arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"Invalid JSON arguments: {exc.msg}"}
        else:
            args = raw_arguments

        if not isinstance(args, dict):
            return {"ok": False, "error": "Tool arguments must decode to a JSON object."}

        props = spec.parameters.get("properties", {})
        for req in spec.parameters.get("required", []):
            if req not in args:
                return {"ok": False, "error": f"Missing required argument '{req}'."}
        if spec.parameters.get("additionalProperties") is False:
            extras = sorted(set(args.keys()) - set(props.keys()))
            if extras:
                return {"ok": False, "error": f"Unexpected arguments: {extras}"}
        for key, val in args.items():
            if key in props and not _matches_json_type(val, props[key].get("type", "")):
                return {
                    "ok": False,
                    "error": f"Argument '{key}' must be {props[key]['type']}, got {type(val).__name__}.",
                }

        try:
            return {"ok": True, "result": spec.handler(**args)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# Concrete tool handlers
def get_service_metrics(service_name: str) -> dict[str, Any]:
    metrics = SERVICE_METRICS.get(service_name)
    if metrics is None:
        return {"found": False, "error": f"No metrics found for service '{service_name}'."}
    return {"found": True, **metrics}


def search_runbooks(query: str) -> dict[str, Any]:
    terms = [t.lower() for t in query.split() if len(t) >= 3]
    matches = [
        rb
        for rb in RUNBOOKS
        if any(term in rb["keywords"].lower() or term in rb["title"].lower() for term in terms)
    ]
    return {"query": query, "matches": matches}


def calculate_error_budget_burn(error_rate_pct: float, slo_target_pct: float) -> dict[str, Any]:
    allowed_error_pct = max(1e-6, 100.0 - slo_target_pct)
    burn_multiplier = round(error_rate_pct / allowed_error_pct, 2)
    severity = "page_immediately" if burn_multiplier >= 10.0 else "ticket"
    return {
        "error_rate_pct": error_rate_pct,
        "slo_target_pct": slo_target_pct,
        "allowed_error_pct": round(allowed_error_pct, 4),
        "burn_rate_multiplier": burn_multiplier,
        "severity": severity,
    }


def get_oncall_engineer(team: str) -> dict[str, Any]:
    entry = ONCALL_ROSTER.get(team)
    if entry is None:
        return {"found": False, "error": f"Unknown team '{team}'."}
    return {"found": True, "team": team, **entry}


def build_incident_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_service_metrics",
            description="Fetch current error rate, p95 latency, top error, and owning team for a service.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "required": ["service_name"],
                "properties": {"service_name": {"type": "string"}},
            },
            handler=get_service_metrics,
        )
    )
    registry.register(
        ToolSpec(
            name="search_runbooks",
            description="Search internal incident runbooks by keyword (e.g. 'payments timeout 504').",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
            handler=search_runbooks,
        )
    )
    registry.register(
        ToolSpec(
            name="calculate_error_budget_burn",
            description="Compute SLO error budget burn multiplier from current error_rate_pct and slo_target_pct.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "required": ["error_rate_pct", "slo_target_pct"],
                "properties": {
                    "error_rate_pct": {"type": "number"},
                    "slo_target_pct": {"type": "number"},
                },
            },
            handler=calculate_error_budget_burn,
        )
    )
    registry.register(
        ToolSpec(
            name="get_oncall_engineer",
            description="Look up the primary on-call engineer and Slack channel for an owning team.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "required": ["team"],
                "properties": {"team": {"type": "string"}},
            },
            handler=get_oncall_engineer,
        )
    )
    return registry


# ---------------------------------------------------------------------------
# Model Client Abstraction (Live OpenAI-compatible + Offline Demo Mock)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTurn:
    message: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class ChatModel(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn: ...


class OpenAICompatibleModel:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError(f"Model API call failed: {exc}") from exc

        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ModelTurn(
            message=dict(choice["message"]),
            finish_reason=str(choice.get("finish_reason", "stop")),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


class DemoIncidentModel:
    """Deterministic multi-step mock model that inspects prior tool observations."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        tool_results: dict[str, dict[str, Any]] = {}
        for msg in messages:
            if msg.get("role") == "tool":
                decoded = json.loads(msg["content"])
                if decoded.get("ok"):
                    tool_results[msg["name"]] = decoded["result"]

        # Step 1: Fetch service metrics first.
        if "get_service_metrics" not in tool_results:
            return ModelTurn(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_service_metrics",
                                "arguments": json.dumps({"service_name": "payments-api"}),
                            },
                        }
                    ],
                },
                finish_reason="tool_calls",
                prompt_tokens=180,
                completion_tokens=24,
            )

        metrics = tool_results["get_service_metrics"]

        # Step 2: Calculate error budget burn + search matching runbooks.
        if "calculate_error_budget_burn" not in tool_results:
            return ModelTurn(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "calculate_error_budget_burn",
                                "arguments": json.dumps(
                                    {
                                        "error_rate_pct": metrics["error_rate_pct"],
                                        "slo_target_pct": 99.9,
                                    }
                                ),
                            },
                        },
                        {
                            "id": "call_3",
                            "type": "function",
                            "function": {
                                "name": "search_runbooks",
                                "arguments": json.dumps({"query": "payments 504 timeout"}),
                            },
                        },
                    ],
                },
                finish_reason="tool_calls",
                prompt_tokens=295,
                completion_tokens=48,
            )

        # Step 3: Look up the owning team's on-call engineer.
        if "get_oncall_engineer" not in tool_results:
            return ModelTurn(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_4",
                            "type": "function",
                            "function": {
                                "name": "get_oncall_engineer",
                                "arguments": json.dumps({"team": metrics["owner_team"]}),
                            },
                        }
                    ],
                },
                finish_reason="tool_calls",
                prompt_tokens=420,
                completion_tokens=22,
            )

        # Step 4: All required observations collected -> produce final brief.
        burn = tool_results["calculate_error_budget_burn"]
        runbook = tool_results["search_runbooks"]["matches"][0]
        oncall = tool_results["get_oncall_engineer"]
        summary = (
            f"Incident Brief for {metrics['service']}:\n"
            f"- Live Metrics: {metrics['error_rate_pct']}% errors, p95 {metrics['p95_latency_ms']} ms "
            f"({metrics['top_error']}).\n"
            f"- SLO Impact: {burn['burn_rate_multiplier']}x burn rate against 99.9% SLO "
            f"({burn['severity']}).\n"
            f"- Runbook {runbook['id']}: {runbook['action']}\n"
            f"- Page On-Call: {oncall['primary']} ({oncall['slack']})."
        )
        return ModelTurn(
            message={"role": "assistant", "content": summary},
            finish_reason="stop",
            prompt_tokens=510,
            completion_tokens=88,
        )


# ---------------------------------------------------------------------------
# Core Agent Loop with Step Logging and Safety Guards
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolCallTrace:
    call_id: str
    tool_name: str
    arguments: str
    ok: bool
    observation_preview: str


@dataclass(frozen=True)
class StepRecord:
    step_number: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    tool_traces: list[ToolCallTrace] = field(default_factory=list)


@dataclass
class AgentResult:
    final_answer: str | None
    stop_reason: str  # "completed", "max_steps_reached", "duplicate_loop_detected"
    steps: list[StepRecord]
    total_prompt_tokens: int
    total_completion_tokens: int


class AgentLoop:
    """Bounded `model -> tool -> observation` loop with structured trace logging."""

    def __init__(
        self,
        model: ChatModel,
        registry: ToolRegistry,
        system_prompt: str,
        max_steps: int = 6,
        max_identical_repeats: int = 2,
    ) -> None:
        self.model = model
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_identical_repeats = max_identical_repeats

    def run(self, user_goal: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_goal},
        ]
        steps: list[StepRecord] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_signature: str | None = None
        identical_count = 0

        for step_num in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            turn = self.model.complete(messages, self.registry.schemas())
            latency_ms = (time.perf_counter() - t0) * 1000

            total_prompt_tokens += turn.prompt_tokens
            total_completion_tokens += turn.completion_tokens

            assistant_msg = turn.message
            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []

            # Termination condition 1: model produced a final text answer with no tool calls.
            if not tool_calls:
                steps.append(
                    StepRecord(
                        step_number=step_num,
                        latency_ms=latency_ms,
                        prompt_tokens=turn.prompt_tokens,
                        completion_tokens=turn.completion_tokens,
                        finish_reason=turn.finish_reason,
                    )
                )
                return AgentResult(
                    final_answer=str(assistant_msg.get("content") or "").strip(),
                    stop_reason="completed",
                    steps=steps,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                )

            # Termination condition 2: detect repeated identical tool calls (simple cycle guard).
            signature = json.dumps(
                [(c["function"]["name"], c["function"].get("arguments", "")) for c in tool_calls],
                sort_keys=True,
            )
            if signature == last_signature:
                identical_count += 1
            else:
                last_signature = signature
                identical_count = 1

            if identical_count > self.max_identical_repeats:
                steps.append(
                    StepRecord(
                        step_number=step_num,
                        latency_ms=latency_ms,
                        prompt_tokens=turn.prompt_tokens,
                        completion_tokens=turn.completion_tokens,
                        finish_reason="loop_guard",
                    )
                )
                return AgentResult(
                    final_answer=None,
                    stop_reason="duplicate_loop_detected",
                    steps=steps,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                )

            # Execute each requested tool and append its observation message.
            traces: list[ToolCallTrace] = []
            for call in tool_calls:
                call_id = str(call.get("id", f"step_{step_num}_call"))
                fn = call["function"]
                tool_name = str(fn["name"])
                raw_args = fn.get("arguments", "{}")

                obs = self.registry.execute(tool_name, raw_args)
                obs_json = json.dumps(obs)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": obs_json,
                    }
                )
                preview = obs_json if len(obs_json) <= 110 else obs_json[:107] + "..."
                traces.append(
                    ToolCallTrace(
                        call_id=call_id,
                        tool_name=tool_name,
                        arguments=str(raw_args),
                        ok=bool(obs.get("ok")),
                        observation_preview=preview,
                    )
                )

            steps.append(
                StepRecord(
                    step_number=step_num,
                    latency_ms=latency_ms,
                    prompt_tokens=turn.prompt_tokens,
                    completion_tokens=turn.completion_tokens,
                    finish_reason=turn.finish_reason,
                    tool_traces=traces,
                )
            )

        # Termination condition 3: reached max_steps without a final answer.
        return AgentResult(
            final_answer=None,
            stop_reason="max_steps_reached",
            steps=steps,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
        )


def print_trace(title: str, result: AgentResult) -> None:
    print(f"=== {title} ===")
    for step in result.steps:
        print(
            f"Step {step.step_number} | finish={step.finish_reason:<10} | "
            f"tokens={step.prompt_tokens}in/{step.completion_tokens}out | {step.latency_ms:.1f} ms"
        )
        for tr in step.tool_traces:
            status = "OK" if tr.ok else "ERR"
            print(f"  -> {tr.tool_name}({tr.arguments})")
            print(f"  <- [{status}] {tr.observation_preview}")
    print(f"\nStop reason : {result.stop_reason}")
    print(
        f"Total tokens: {result.total_prompt_tokens} prompt + "
        f"{result.total_completion_tokens} completion"
    )
    if result.final_answer:
        print("\nFinal Answer:")
        print(result.final_answer)
    else:
        print("\nFinal Answer: <none — stopped early by safety guard>")
    print()


def is_local_endpoint(base_url: str) -> bool:
    lowered = base_url.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered


def build_model(force_demo: bool) -> tuple[ChatModel, bool]:
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    env_demo = os.getenv("AGENT_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    missing_key = (not api_key) or api_key == "your-api-key-here"

    if force_demo or env_demo or (missing_key and not is_local_endpoint(base_url)):
        return DemoIncidentModel(), True

    effective_key = api_key if not missing_key else "ollama"
    return OpenAICompatibleModel(base_url=base_url, api_key=effective_key, model=model_name), False


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 5 multi-step agent loop.")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="Goal for the agent.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum model iterations.")
    parser.add_argument("--demo", action="store_true", help="Run offline deterministic demo.")
    args = parser.parse_args()

    registry = build_incident_registry()
    model, demo_mode = build_model(force_demo=args.demo)
    system_prompt = (
        "You are an SRE incident triage agent. Use tools to inspect live metrics, "
        "calculate error budget burn, consult runbooks, and check the on-call roster "
        "before writing a concise incident brief."
    )

    loop = AgentLoop(
        model=model,
        registry=registry,
        system_prompt=system_prompt,
        max_steps=args.max_steps,
    )

    try:
        mode_tag = "DEMO" if demo_mode else "LIVE"
        result = loop.run(args.task)
        print_trace(f"Run 1: Multi-Step Incident Investigation [{mode_tag}, max_steps={args.max_steps}]", result)

        # In demo mode, also demonstrate the max_steps guard halting cleanly.
        if demo_mode:
            capped_loop = AgentLoop(
                model=DemoIncidentModel(),
                registry=registry,
                system_prompt=system_prompt,
                max_steps=2,
            )
            capped_result = capped_loop.run(args.task)
            print_trace("Run 2: Safety Guard Demonstration [DEMO, max_steps=2]", capped_result)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Tip: Re-run with `--demo` to inspect the multi-step loop offline.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
