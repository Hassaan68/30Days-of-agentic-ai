"""Day 4: Tool & function calling basics with a validated ToolRegistry.

Demonstrates the complete tool-calling contract:
1. Define local Python functions with strict JSON Schema parameter contracts.
2. Export schemas in the OpenAI-compatible `tools` wire format.
3. Send messages + tools to a model (live API or deterministic `--demo` mock).
4. Parse `tool_calls`, validate arguments before execution, run the Python function,
   and append `{"role": "tool", "tool_call_id": ..., "content": ...}` observations.
5. Send the observations back to the model to produce the final grounded answer.

Examples:
    python3 04_Day_Tool_and_Function_Calling_Basics/tool_calling_basics.py --demo
    python3 04_Day_Tool_and_Function_Calling_Basics/tool_calling_basics.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# Local mock data backing our tools.
ORDERS_DB: dict[str, dict[str, Any]] = {
    "ORD-1042": {
        "order_id": "ORD-1042",
        "status": "in_transit",
        "carrier": "FastShip",
        "subtotal_usd": 120.00,
        "weight_kg": 2.5,
        "destination_zone": "EU",
    },
    "ORD-2088": {
        "order_id": "ORD-2088",
        "status": "delivered",
        "carrier": "MetroPost",
        "subtotal_usd": 45.00,
        "weight_kg": 0.8,
        "destination_zone": "US",
    },
}

FX_RATES_FROM_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 154.0,
}

ZONE_RATE_PER_KG_USD: dict[str, float] = {
    "US": 4.0,
    "EU": 8.5,
    "APAC": 11.0,
}


# ---------------------------------------------------------------------------
# Tool Registry & Schema Validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    """Pairs a Python callable with its JSON Schema contract."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def to_openai_tool(self) -> dict[str, Any]:
        """Export in the OpenAI-compatible `tools` request schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _type_matches(value: Any, expected: str) -> bool:
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


def validate_tool_args(args: Any, parameters_schema: dict[str, Any]) -> list[str]:
    """Validate parsed tool arguments against a JSON Schema object definition."""
    if not isinstance(args, dict):
        return [f"Arguments must be a JSON object, got {type(args).__name__}"]

    errors: list[str] = []
    properties = parameters_schema.get("properties", {})
    required = parameters_schema.get("required", [])

    for key in required:
        if key not in args:
            errors.append(f"Missing required argument '{key}'")

    if parameters_schema.get("additionalProperties") is False:
        for extra in sorted(set(args.keys()) - set(properties.keys())):
            errors.append(f"Unexpected argument '{extra}'")

    for key, val in args.items():
        if key not in properties:
            continue
        prop_schema = properties[key]
        expected_type = prop_schema.get("type")
        if expected_type and not _type_matches(val, expected_type):
            errors.append(f"Argument '{key}' expected {expected_type}, got {type(val).__name__}")
            continue
        if "enum" in prop_schema and val not in prop_schema["enum"]:
            errors.append(f"Argument '{key}' must be one of {prop_schema['enum']}, got {val!r}")
        if expected_type in {"integer", "number"}:
            if "minimum" in prop_schema and val < prop_schema["minimum"]:
                errors.append(f"Argument '{key}' must be >= {prop_schema['minimum']}, got {val}")

    return errors


class ToolRegistry:
    """Stores available tools, exports schemas, and executes validated tool calls."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [spec.to_openai_tool() for spec in self._tools.values()]

    def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Parse, validate, and run one tool_call dict; return a `role: tool` message."""
        call_id = str(tool_call.get("id", "missing_id"))
        fn_block = tool_call.get("function") or {}
        tool_name = str(fn_block.get("name", ""))
        raw_args = fn_block.get("arguments", "{}")

        spec = self._tools.get(tool_name)
        if spec is None:
            observation: dict[str, Any] = {
                "ok": False,
                "error": f"Unknown tool '{tool_name}'. Available: {sorted(self._tools.keys())}",
            }
            return self._wrap_observation(call_id, tool_name, observation)

        # OpenAI sends `arguments` as a JSON-encoded string; parse defensively.
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                observation = {
                    "ok": False,
                    "error": f"Invalid JSON in tool arguments: {exc.msg}",
                }
                return self._wrap_observation(call_id, tool_name, observation)
        else:
            parsed_args = raw_args

        validation_errors = validate_tool_args(parsed_args, spec.parameters)
        if validation_errors:
            observation = {
                "ok": False,
                "error": "Argument validation failed",
                "details": validation_errors,
            }
            return self._wrap_observation(call_id, tool_name, observation)

        try:
            result = spec.handler(**parsed_args)
            observation = {"ok": True, "result": result}
        except Exception as exc:  # Keep tool errors structured instead of crashing the app.
            observation = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        return self._wrap_observation(call_id, tool_name, observation)

    @staticmethod
    def _wrap_observation(call_id: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": json.dumps(payload),
        }


# ---------------------------------------------------------------------------
# Concrete Local Python Tool Functions
# ---------------------------------------------------------------------------
def get_order_status(order_id: str) -> dict[str, Any]:
    order = ORDERS_DB.get(order_id)
    if order is None:
        return {"found": False, "error": f"Order '{order_id}' does not exist."}
    return {"found": True, **order}


def convert_currency(amount_usd: float, target_currency: str) -> dict[str, Any]:
    rate = FX_RATES_FROM_USD[target_currency]
    converted = round(amount_usd * rate, 2)
    return {
        "amount_usd": amount_usd,
        "target_currency": target_currency,
        "rate": rate,
        "converted_amount": converted,
    }


def calculate_shipping(weight_kg: float, destination_zone: str) -> dict[str, Any]:
    rate = ZONE_RATE_PER_KG_USD[destination_zone]
    cost = round(5.0 + weight_kg * rate, 2)
    return {
        "weight_kg": weight_kg,
        "destination_zone": destination_zone,
        "shipping_cost_usd": cost,
    }


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_order_status",
            description="Look up order status, subtotal in USD, package weight, and destination zone.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "required": ["order_id"],
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order identifier such as ORD-1042.",
                    }
                },
            },
            handler=get_order_status,
        )
    )
    registry.register(
        ToolSpec(
            name="convert_currency",
            description="Convert a USD amount into EUR, GBP, JPY, or USD using current store rates.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "required": ["amount_usd", "target_currency"],
                "properties": {
                    "amount_usd": {"type": "number", "minimum": 0},
                    "target_currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"]},
                },
            },
            handler=convert_currency,
        )
    )
    registry.register(
        ToolSpec(
            name="calculate_shipping",
            description="Estimate shipping cost in USD from package weight (kg) and destination zone.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "required": ["weight_kg", "destination_zone"],
                "properties": {
                    "weight_kg": {"type": "number", "minimum": 0.1},
                    "destination_zone": {"type": "string", "enum": ["US", "EU", "APAC"]},
                },
            },
            handler=calculate_shipping,
        )
    )
    return registry


# ---------------------------------------------------------------------------
# Model Client (Live OpenAI-compatible API + Offline Demo Mock)
# ---------------------------------------------------------------------------
def is_local_endpoint(base_url: str) -> bool:
    lowered = base_url.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered


def should_use_demo(force_demo: bool) -> bool:
    env_demo = os.getenv("AGENT_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    missing_key = (not api_key) or api_key == "your-api-key-here"
    return force_demo or env_demo or (missing_key and not is_local_endpoint(base_url))


def call_mock_model(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic two-turn mock demonstrating tool_calls and final answer synthesis."""
    tool_messages = [m for m in messages if m["role"] == "tool"]

    if not tool_messages:
        # Turn 1: Ask the application to fetch order ORD-1042 and convert $120.00?
        # Even better: call get_order_status first, because the model does not know the
        # subtotal or weight until it sees the order record!
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_order_001",
                    "type": "function",
                    "function": {
                        "name": "get_order_status",
                        "arguments": json.dumps({"order_id": "ORD-1042"}),
                    },
                }
            ],
        }

    if len(tool_messages) == 1:
        # Turn 2: Now that we have the order subtotal ($120) and weight (2.5 kg to EU),
        # request both shipping calculation and EUR conversion in parallel!
        order_obs = json.loads(tool_messages[0]["content"])["result"]
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_ship_002",
                    "type": "function",
                    "function": {
                        "name": "calculate_shipping",
                        "arguments": json.dumps(
                            {
                                "weight_kg": order_obs["weight_kg"],
                                "destination_zone": order_obs["destination_zone"],
                            }
                        ),
                    },
                },
                {
                    "id": "call_fx_003",
                    "type": "function",
                    "function": {
                        "name": "convert_currency",
                        "arguments": json.dumps(
                            {
                                "amount_usd": order_obs["subtotal_usd"],
                                "target_currency": "EUR",
                            }
                        ),
                    },
                },
            ],
        }

    # Turn 3: Synthesize a grounded final answer from the tool observations.
    by_name = {m["name"]: json.loads(m["content"])["result"] for m in tool_messages}
    order = by_name["get_order_status"]
    ship = by_name["calculate_shipping"]
    fx = by_name["convert_currency"]
    return {
        "role": "assistant",
        "content": (
            f"Order {order['order_id']} is currently {order['status']} via {order['carrier']}. "
            f"Its subtotal is ${order['subtotal_usd']:.2f} USD ({fx['converted_amount']:.2f} EUR "
            f"at rate {fx['rate']}), and estimated shipping to {order['destination_zone']} for "
            f"{order['weight_kg']} kg is ${ship['shipping_cost_usd']:.2f} USD."
        ),
    }


def call_live_model(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or "ollama"

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Tool-calling API request failed: {exc}") from exc
    return dict(data["choices"][0]["message"])


def run_demo_validation_check(registry: ToolRegistry) -> None:
    """Show how the registry safely rejects invalid tool arguments instead of crashing."""
    print("=== Defensive Argument Validation Check ===")
    bad_calls = [
        {
            "id": "call_bad_1",
            "type": "function",
            "function": {
                "name": "convert_currency",
                "arguments": json.dumps({"amount_usd": -10, "target_currency": "BTC"}),
            },
        },
        {
            "id": "call_bad_2",
            "type": "function",
            "function": {
                "name": "delete_database",
                "arguments": "{}",
            },
        },
    ]
    for call in bad_calls:
        obs = registry.execute_tool_call(call)
        print(f"  Tool '{call['function']['name']}' -> {obs['content']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 4 tool calling basics demo.")
    parser.add_argument("--demo", action="store_true", help="Run with offline mock model.")
    args = parser.parse_args()

    registry = build_default_registry()
    demo_mode = should_use_demo(args.demo)
    mode_label = "DEMO (offline mock)" if demo_mode else "LIVE API"

    print(f"=== Tool Calling Round-Trip [{mode_label}] ===")
    print(f"Registered tools: {[t['function']['name'] for t in registry.openai_schemas()]}\n")

    user_prompt = (
        "Check order ORD-1042, estimate its shipping cost to its destination zone, "
        "and tell me the order subtotal in EUR."
    )
    print(f"User: {user_prompt}\n")

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an order operations assistant. Use the provided tools to look up "
                "real data before answering. Never guess order totals or exchange rates."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    # Run up to 4 model turns (capped so a single script never loops endlessly).
    for turn in range(1, 5):
        try:
            assistant_msg = (
                call_mock_model(messages)
                if demo_mode
                else call_live_model(messages, registry.openai_schemas())
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        messages.append(assistant_msg)
        tool_calls = assistant_msg.get("tool_calls") or []

        if not tool_calls:
            print(f"Turn {turn} [Assistant Final Answer]:")
            print(f"  {assistant_msg.get('content', '').strip()}\n")
            break

        print(f"Turn {turn} [Model requested {len(tool_calls)} tool call(s)]:")
        for call in tool_calls:
            fn = call["function"]
            print(f"  -> Call {call['id']}: {fn['name']}({fn['arguments']})")
            observation_msg = registry.execute_tool_call(call)
            print(f"  <- Tool observation: {observation_msg['content']}")
            messages.append(observation_msg)
        print()

    run_demo_validation_check(registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
