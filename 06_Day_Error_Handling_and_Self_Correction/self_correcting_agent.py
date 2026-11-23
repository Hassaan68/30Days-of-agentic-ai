"""Day 6: Error Handling and Self-Correction in an Agent Loop.

Demonstrates how to return structured validation and tool errors back to an LLM
so it can fix malformed arguments or bad IDs, while enforcing retry caps,
duplicate-call detection, and immediate stops on fatal errors.

Run offline demo (no API key needed):
    python 06_Day_Error_Handling_and_Self_Correction/self_correcting_agent.py --demo

Run a live query against an OpenAI-compatible endpoint:
    export OPENAI_API_KEY="your-key"
    python 06_Day_Error_Handling_and_Self_Correction/self_correcting_agent.py \
        --query "Refund order ORD-1042 for reason damaged_item"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Simulated backend data & tool implementations
# ---------------------------------------------------------------------------

ORDERS_DB: dict[str, dict[str, Any]] = {
    "ORD-1042": {
        "order_id": "ORD-1042",
        "customer": "Maya Lin",
        "amount_usd": 48.50,
        "status": "delivered",
        "refundable": True,
        "requires_manager": False,
    },
    "ORD-2099": {
        "order_id": "ORD-2099",
        "customer": "Devon Brooks",
        "amount_usd": 1450.00,
        "status": "delivered",
        "refundable": True,
        "requires_manager": True,
    },
    "ORD-3011": {
        "order_id": "ORD-3011",
        "customer": "Priya Nair",
        "amount_usd": 19.00,
        "status": "refunded",
        "refundable": False,
        "requires_manager": False,
    },
}

VALID_REFUND_REASONS = ("damaged_item", "late_delivery", "wrong_item", "duplicate_charge")


@dataclass
class ToolError(Exception):
    """Structured tool failure that distinguishes retryable vs. fatal errors."""

    error_type: str
    message: str
    retryable: bool = True
    hint: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_observation(self, tool_name: str, attempts_used: int, max_attempts: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "tool": tool_name,
            "error_type": self.error_type,
            "retryable": self.retryable,
            "message": self.message,
            "attempts_used": attempts_used,
            "attempts_remaining": max(0, max_attempts - attempts_used) if self.retryable else 0,
        }
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload


def tool_list_orders(customer: str | None = None) -> dict[str, Any]:
    """List known orders, optionally filtered by substring match on customer name."""
    results = []
    for order in ORDERS_DB.values():
        if customer and customer.lower() not in order["customer"].lower():
            continue
        results.append(
            {
                "order_id": order["order_id"],
                "customer": order["customer"],
                "amount_usd": order["amount_usd"],
                "status": order["status"],
            }
        )
    return {"ok": True, "orders": results}


def tool_get_order(order_id: str) -> dict[str, Any]:
    """Fetch full details for one order."""
    if not isinstance(order_id, str) or not order_id.strip():
        raise ToolError(
            error_type="validation_error",
            message="order_id must be a non-empty string.",
            retryable=True,
            hint="Call list_orders to find valid order IDs like 'ORD-1042'.",
        )
    order = ORDERS_DB.get(order_id.strip().upper())
    if order is None:
        raise ToolError(
            error_type="not_found",
            message=f"Order '{order_id}' does not exist.",
            retryable=True,
            hint="Call list_orders to look up the exact order_id.",
        )
    return {"ok": True, "order": order}


def tool_issue_refund(order_id: str, amount_usd: float, reason: str) -> dict[str, Any]:
    """Issue a refund after validating order existence, amount, reason, and policy."""
    if not isinstance(order_id, str) or not order_id.strip():
        raise ToolError(
            error_type="validation_error",
            message="order_id must be a non-empty string.",
            retryable=True,
            hint="Provide an order_id such as 'ORD-1042'.",
        )

    order = ORDERS_DB.get(order_id.strip().upper())
    if order is None:
        raise ToolError(
            error_type="not_found",
            message=f"Order '{order_id}' was not found.",
            retryable=True,
            hint="Call list_orders to verify the customer's actual order_id before retrying.",
        )

    if order["requires_manager"]:
        raise ToolError(
            error_type="permission_denied",
            message=(
                f"Order {order['order_id']} (${order['amount_usd']:.2f}) exceeds the "
                "automated refund limit and requires manager approval."
            ),
            retryable=False,
            hint="Do not retry. Inform the user that a manager must approve this refund.",
        )

    if not order["refundable"]:
        raise ToolError(
            error_type="invalid_state",
            message=f"Order {order['order_id']} is already in status '{order['status']}' and cannot be refunded.",
            retryable=False,
            hint="Do not retry. Explain to the user that the order is not eligible for a refund.",
        )

    if not isinstance(amount_usd, (int, float)) or isinstance(amount_usd, bool):
        raise ToolError(
            error_type="validation_error",
            message=f"amount_usd must be a number, got {type(amount_usd).__name__}.",
            retryable=True,
            hint=f"Pass a numeric amount up to {order['amount_usd']}.",
        )

    if amount_usd <= 0 or amount_usd > order["amount_usd"]:
        raise ToolError(
            error_type="validation_error",
            message=(
                f"amount_usd={amount_usd} is out of range. "
                f"Must be > 0 and <= {order['amount_usd']:.2f}."
            ),
            retryable=True,
            hint=f"Set amount_usd to {order['amount_usd']:.2f} for a full refund.",
            details={"max_refundable_usd": order["amount_usd"]},
        )

    if reason not in VALID_REFUND_REASONS:
        raise ToolError(
            error_type="validation_error",
            message=f"Invalid reason '{reason}'.",
            retryable=True,
            hint=f"Use one of the allowed reasons: {', '.join(VALID_REFUND_REASONS)}.",
            details={"allowed_reasons": list(VALID_REFUND_REASONS)},
        )

    return {
        "ok": True,
        "refund_id": f"REF-{order['order_id'].split('-')[1]}",
        "order_id": order["order_id"],
        "amount_usd": round(float(amount_usd), 2),
        "reason": reason,
        "status": "processed",
    }


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "List customer orders to find valid order IDs and amounts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {
                        "type": "string",
                        "description": "Optional customer name filter.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up one order by order_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, e.g. ORD-1042."}
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund for an eligible order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, e.g. ORD-1042."},
                    "amount_usd": {"type": "number", "description": "Refund amount in USD."},
                    "reason": {
                        "type": "string",
                        "enum": list(VALID_REFUND_REASONS),
                        "description": "Structured refund reason code.",
                    },
                },
                "required": ["order_id", "amount_usd", "reason"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "list_orders": tool_list_orders,
    "get_order": tool_get_order,
    "issue_refund": tool_issue_refund,
}


# ---------------------------------------------------------------------------
# Tool execution & error guardrails
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """Caps and loop-breakers for bounded self-correction."""

    max_steps: int = 8
    max_retries_per_tool: int = 3
    max_consecutive_errors: int = 3


@dataclass
class AgentRunResult:
    """Final outcome returned by the self-correcting loop."""

    status: str  # "completed" | "stopped_fatal" | "stopped_circuit_breaker" | "stopped_max_steps"
    final_answer: str
    steps_taken: int
    total_tool_errors: int
    trace: list[str]


def _canonical_call_signature(tool_name: str, raw_args: str) -> str:
    """Hash tool name + normalized arguments to catch repeated broken calls."""
    try:
        parsed = json.loads(raw_args)
        normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except json.JSONDecodeError:
        normalized = raw_args.strip()
    digest = hashlib.sha256(f"{tool_name}:{normalized}".encode("utf-8")).hexdigest()
    return digest[:16]


def execute_tool_safely(
    tool_name: str,
    raw_args: str,
    tool_error_counts: dict[str, int],
    failed_signatures: set[str],
    policy: RetryPolicy,
) -> tuple[dict[str, Any], bool, bool]:
    """Execute a tool and return (observation_dict, had_error, is_fatal).

    Never raises unhandled exceptions back to the main loop; converts parsing,
    unknown-tool, schema, and runtime failures into structured observations.
    """
    if tool_name not in TOOL_REGISTRY:
        tool_error_counts[tool_name] = tool_error_counts.get(tool_name, 0) + 1
        err = ToolError(
            error_type="unknown_tool",
            message=f"Tool '{tool_name}' does not exist.",
            retryable=True,
            hint=f"Available tools: {', '.join(sorted(TOOL_REGISTRY))}.",
        )
        return err.to_observation(tool_name, tool_error_counts[tool_name], policy.max_retries_per_tool), True, False

    sig = _canonical_call_signature(tool_name, raw_args)
    if sig in failed_signatures:
        err = ToolError(
            error_type="duplicate_failed_call",
            message="Identical tool call already failed earlier in this run.",
            retryable=False,
            hint="Change the arguments, call a discovery tool, or report the failure to the user.",
        )
        return err.to_observation(tool_name, tool_error_counts.get(tool_name, 0), policy.max_retries_per_tool), True, True

    if tool_error_counts.get(tool_name, 0) >= policy.max_retries_per_tool:
        err = ToolError(
            error_type="retry_cap_exceeded",
            message=f"Retry cap ({policy.max_retries_per_tool}) exceeded for tool '{tool_name}'.",
            retryable=False,
            hint="Stop calling this tool and explain the issue clearly to the user.",
        )
        return err.to_observation(tool_name, tool_error_counts[tool_name], policy.max_retries_per_tool), True, True

    try:
        parsed_args = json.loads(raw_args) if raw_args.strip() else {}
        if not isinstance(parsed_args, dict):
            raise ToolError(
                error_type="json_parse_error",
                message="Tool arguments must be a JSON object.",
                retryable=True,
                hint="Return a valid JSON object matching the tool parameter schema.",
            )
    except json.JSONDecodeError as exc:
        tool_error_counts[tool_name] = tool_error_counts.get(tool_name, 0) + 1
        failed_signatures.add(sig)
        err = ToolError(
            error_type="json_parse_error",
            message=f"Malformed JSON arguments: {exc.msg} at position {exc.pos}.",
            retryable=True,
            hint="Re-emit valid JSON with double-quoted keys and properly closed braces.",
        )
        return err.to_observation(tool_name, tool_error_counts[tool_name], policy.max_retries_per_tool), True, False

    func = TOOL_REGISTRY[tool_name]
    try:
        result = func(**parsed_args)
        return result, False, False
    except TypeError as exc:
        tool_error_counts[tool_name] = tool_error_counts.get(tool_name, 0) + 1
        failed_signatures.add(sig)
        err = ToolError(
            error_type="schema_mismatch",
            message=f"Argument signature error: {exc}",
            retryable=True,
            hint="Check the required parameter names and remove any unexpected keys.",
        )
        return err.to_observation(tool_name, tool_error_counts[tool_name], policy.max_retries_per_tool), True, False
    except ToolError as exc:
        tool_error_counts[tool_name] = tool_error_counts.get(tool_name, 0) + 1
        failed_signatures.add(sig)
        return (
            exc.to_observation(tool_name, tool_error_counts[tool_name], policy.max_retries_per_tool),
            True,
            not exc.retryable,
        )
    except Exception as exc:  # Unexpected bug inside tool code
        tool_error_counts[tool_name] = tool_error_counts.get(tool_name, 0) + 1
        failed_signatures.add(sig)
        err = ToolError(
            error_type="internal_tool_error",
            message=f"Unexpected failure inside '{tool_name}': {type(exc).__name__}: {exc}",
            retryable=False,
            hint="Do not retry automatically; escalate this error.",
        )
        return err.to_observation(tool_name, tool_error_counts[tool_name], policy.max_retries_per_tool), True, True


# ---------------------------------------------------------------------------
# LLM clients (live OpenAI-compatible HTTP + deterministic offline demo mock)
# ---------------------------------------------------------------------------

class OpenAICompatibleClient:
    """Minimal stdlib client for OpenAI-compatible /chat/completions endpoints."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.0,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM network error: {exc.reason}") from exc

        return parsed["choices"][0]["message"]


class DemoScriptedClient:
    """Deterministic mock LLM that plays realistic error-and-recovery scenarios."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.turn = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.turn += 1
        if self.scenario == "recoverable":
            return self._recoverable_scenario()
        if self.scenario == "fatal":
            return self._fatal_scenario()
        if self.scenario == "loop_breaker":
            return self._loop_breaker_scenario()
        raise ValueError(f"Unknown demo scenario: {self.scenario}")

    def _recoverable_scenario(self) -> dict[str, Any]:
        # Step 1: Model emits malformed JSON (missing closing brace).
        if self.turn == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id": "ORD-9999"',
                        },
                    }
                ],
            }
        # Step 2: Model fixes JSON syntax, but guesses a nonexistent order ID.
        if self.turn == 2:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": json.dumps({"order_id": "ORD-9999"}),
                        },
                    }
                ],
            }
        # Step 3: Model follows the error hint ("Call list_orders") to look up Maya Lin's order.
        if self.turn == 3:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "list_orders",
                            "arguments": json.dumps({"customer": "Maya"}),
                        },
                    }
                ],
            }
        # Step 4: Model uses the right order_id, but passes an out-of-range amount.
        if self.turn == 4:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_4",
                        "type": "function",
                        "function": {
                            "name": "issue_refund",
                            "arguments": json.dumps(
                                {"order_id": "ORD-1042", "amount_usd": 60.0, "reason": "broken"}
                            ),
                        },
                    }
                ],
            }
        # Step 5: Model corrects amount_usd to 48.50 and reason to damaged_item.
        if self.turn == 5:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_5",
                        "type": "function",
                        "function": {
                            "name": "issue_refund",
                            "arguments": json.dumps(
                                {
                                    "order_id": "ORD-1042",
                                    "amount_usd": 48.50,
                                    "reason": "damaged_item",
                                }
                            ),
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": (
                "Refund REF-1042 for $48.50 on Maya Lin's order ORD-1042 has been processed "
                "with reason 'damaged_item'."
            ),
        }

    def _fatal_scenario(self) -> dict[str, Any]:
        # Step 1: Model attempts refund on high-value order requiring manager approval.
        if self.turn == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_f1",
                        "type": "function",
                        "function": {
                            "name": "issue_refund",
                            "arguments": json.dumps(
                                {
                                    "order_id": "ORD-2099",
                                    "amount_usd": 1450.0,
                                    "reason": "damaged_item",
                                }
                            ),
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": (
                "I cannot process the $1,450.00 refund for order ORD-2099 automatically "
                "because it exceeds the automated refund threshold. A manager needs to approve it."
            ),
        }

    def _loop_breaker_scenario(self) -> dict[str, Any]:
        # Stubborn model keeps calling the exact same broken payload over and over.
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_loop_{self.turn}",
                    "type": "function",
                    "function": {
                        "name": "get_order",
                        "arguments": '{"order_id": "ORD-0000"}',
                    },
                }
            ],
        }


# ---------------------------------------------------------------------------
# Self-correcting agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a support operations assistant with access to order and refund tools. "
    "When a tool returns ok=false with retryable=true, read the error message and hint "
    "carefully and adjust your next tool call. "
    "When a tool returns retryable=false, do NOT retry that action; explain the blocker clearly."
)


def run_self_correcting_agent(
    user_query: str,
    client: OpenAICompatibleClient | DemoScriptedClient,
    policy: RetryPolicy | None = None,
) -> AgentRunResult:
    """Run an agent loop with structured error feedback and bounded retries."""
    policy = policy or RetryPolicy()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    tool_error_counts: dict[str, int] = {}
    failed_signatures: set[str] = set()
    consecutive_errors = 0
    total_tool_errors = 0
    trace: list[str] = []

    for step in range(1, policy.max_steps + 1):
        assistant_msg = client.chat(messages, TOOLS_SCHEMA)
        messages.append(assistant_msg)
        tool_calls = assistant_msg.get("tool_calls") or []

        if not tool_calls:
            final_text = (assistant_msg.get("content") or "").strip()
            trace.append(f"Step {step}: assistant produced final response.")
            return AgentRunResult(
                status="completed",
                final_answer=final_text,
                steps_taken=step,
                total_tool_errors=total_tool_errors,
                trace=trace,
            )

        fatal_triggered = False
        fatal_reason = ""

        for call in tool_calls:
            call_id = call["id"]
            tool_name = call["function"]["name"]
            raw_args = call["function"].get("arguments", "{}")

            observation, had_error, is_fatal = execute_tool_safely(
                tool_name=tool_name,
                raw_args=raw_args,
                tool_error_counts=tool_error_counts,
                failed_signatures=failed_signatures,
                policy=policy,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(observation),
                }
            )

            if had_error:
                total_tool_errors += 1
                consecutive_errors += 1
                trace.append(
                    f"Step {step}: {tool_name}({raw_args}) -> ERROR "
                    f"[{observation['error_type']}, retryable={observation['retryable']}]: "
                    f"{observation['message']}"
                )
                if is_fatal:
                    fatal_triggered = True
                    fatal_reason = observation["message"]
            else:
                consecutive_errors = 0
                trace.append(f"Step {step}: {tool_name}({raw_args}) -> OK")

        if fatal_triggered:
            # Duplicate call / retry cap -> stop immediately without letting the model loop.
            if any(
                tag in trace[-1]
                for tag in ("duplicate_failed_call", "retry_cap_exceeded")
            ):
                trace.append(f"Step {step}: circuit breaker tripped ({fatal_reason}).")
                return AgentRunResult(
                    status="stopped_circuit_breaker",
                    final_answer=(
                        "Stopped automatically to prevent an infinite retry loop. "
                        f"Last error: {fatal_reason}"
                    ),
                    steps_taken=step,
                    total_tool_errors=total_tool_errors,
                    trace=trace,
                )

            # Policy/permission fatal error -> give the model one final turn (no tools)
            # to present a clear user-facing explanation, or stop cleanly.
            final_msg = client.chat(messages, tools=[])
            final_text = (final_msg.get("content") or fatal_reason).strip()
            trace.append(f"Step {step}: non-retryable tool error -> returned user explanation.")
            return AgentRunResult(
                status="stopped_fatal",
                final_answer=final_text,
                steps_taken=step,
                total_tool_errors=total_tool_errors,
                trace=trace,
            )

        if consecutive_errors >= policy.max_consecutive_errors:
            trace.append(
                f"Step {step}: circuit breaker tripped after {consecutive_errors} consecutive errors."
            )
            return AgentRunResult(
                status="stopped_circuit_breaker",
                final_answer=(
                    f"Stopped after {consecutive_errors} consecutive tool errors "
                    "to avoid wasting tokens."
                ),
                steps_taken=step,
                total_tool_errors=total_tool_errors,
                trace=trace,
            )

    trace.append(f"Reached max_steps={policy.max_steps} without a final answer.")
    return AgentRunResult(
        status="stopped_max_steps",
        final_answer="Could not complete the request within the step budget.",
        steps_taken=policy.max_steps,
        total_tool_errors=total_tool_errors,
        trace=trace,
    )


def _should_use_demo(force_demo: bool) -> bool:
    if force_demo or os.getenv("AGENT_DEMO_MODE", "").strip() == "1":
        return True
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    return not api_key and not is_local


def _print_result(title: str, result: AgentRunResult) -> None:
    print(f"\n=== {title} ===")
    print(f"Status: {result.status} | Steps: {result.steps_taken} | Tool errors: {result.total_tool_errors}")
    print("Trace:")
    for line in result.trace:
        print(f"  - {line}")
    print(f"Final answer: {result.final_answer}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 6: Self-correcting agent loop demo")
    parser.add_argument("--demo", action="store_true", help="Run offline deterministic scenarios")
    parser.add_argument(
        "--scenario",
        choices=["all", "recoverable", "fatal", "loop_breaker"],
        default="all",
        help="Which demo scenario to run (only used in --demo mode).",
    )
    parser.add_argument(
        "--query",
        default="Please refund Maya Lin's damaged order.",
        help="User query for live LLM mode.",
    )
    args = parser.parse_args()

    if _should_use_demo(args.demo):
        scenarios = (
            ["recoverable", "fatal", "loop_breaker"]
            if args.scenario == "all"
            else [args.scenario]
        )
        prompts = {
            "recoverable": "Please refund Maya Lin's damaged order.",
            "fatal": "Refund order ORD-2099 for $1450 due to damaged_item.",
            "loop_breaker": "Check order ORD-0000.",
        }
        titles = {
            "recoverable": "Scenario 1: Recoverable Error -> Self-Correction",
            "fatal": "Scenario 2: Fatal Policy Error -> Stop Without Retrying",
            "loop_breaker": "Scenario 3: Duplicate Failing Call -> Circuit Breaker",
        }
        for name in scenarios:
            res = run_self_correcting_agent(
                user_query=prompts[name],
                client=DemoScriptedClient(name),
            )
            _print_result(titles[name], res)
        return 0

    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    client = OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)
    result = run_self_correcting_agent(user_query=args.query, client=client)
    _print_result(f"Live Run ({model})", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
