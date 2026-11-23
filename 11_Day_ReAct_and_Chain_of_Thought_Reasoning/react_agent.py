"""Day 11: ReAct (Reason + Act) Agent with Explicit Thought-Action-Observation Loop.

Demonstrates:
- Prompting a model to reason explicitly before choosing a tool call.
- Parsing structured Thought / Action / Action Input / Final Answer turns.
- Keeping internal reasoning trace separate from the user-facing response.
- Supporting both live OpenAI-compatible APIs and an offline `--demo` mode.

Usage:
    python 11_Day_ReAct_and_Chain_of_Thought_Reasoning/react_agent.py --demo
    python 11_Day_ReAct_and_Chain_of_Thought_Reasoning/react_agent.py --demo --show-trace
    python 11_Day_ReAct_and_Chain_of_Thought_Reasoning/react_agent.py --query "Can order ORD-1042 get a refund, and how much?"
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Describes a callable tool exposed to the ReAct agent."""

    name: str
    description: str
    args_schema: str
    handler: Callable[[dict[str, Any]], str]


@dataclass
class ParsedTurn:
    """Structured output parsed from a single LLM turn."""

    thought: str
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    raw_text: str = ""
    parse_error: str | None = None


@dataclass
class ReActStep:
    """One recorded step in the agent's internal execution trace."""

    step_number: int
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    """Separates the clean user-facing answer from internal reasoning steps."""

    query: str
    final_answer: str
    steps: list[ReActStep]
    finished: bool
    mode: str


# ---------------------------------------------------------------------------
# Sample domain tools (order support & refund math)
# ---------------------------------------------------------------------------

ORDERS_DB: dict[str, dict[str, Any]] = {
    "ORD-1042": {
        "order_id": "ORD-1042",
        "customer": "Priya Nair",
        "item": "Mechanical Keyboard (75%)",
        "category": "electronics",
        "subtotal_usd": 180.00,
        "shipping_paid_usd": 12.00,
        "days_since_delivery": 19,
        "opened": True,
        "status": "delivered",
    },
    "ORD-1088": {
        "order_id": "ORD-1088",
        "customer": "Marcus Chen",
        "item": "USB-C Dock",
        "category": "electronics",
        "subtotal_usd": 95.00,
        "shipping_paid_usd": 0.00,
        "days_since_delivery": 42,
        "opened": False,
        "status": "delivered",
    },
}

RETURN_POLICIES: dict[str, dict[str, Any]] = {
    "electronics": {
        "return_window_days": 30,
        "opened_restocking_fee_pct": 15,
        "shipping_refundable": False,
        "notes": "Opened electronics within 30 days incur a 15% restocking fee on item subtotal. Original shipping is non-refundable.",
    },
    "apparel": {
        "return_window_days": 45,
        "opened_restocking_fee_pct": 0,
        "shipping_refundable": True,
        "notes": "Unwashed apparel can be returned within 45 days for a full refund including shipping.",
    },
}


def lookup_order(args: dict[str, Any]) -> str:
    order_id = str(args.get("order_id", "")).strip().upper()
    record = ORDERS_DB.get(order_id)
    if not record:
        return json.dumps({"error": f"Order '{order_id}' not found."})
    return json.dumps(record)


def get_return_policy(args: dict[str, Any]) -> str:
    category = str(args.get("category", "")).strip().lower()
    policy = RETURN_POLICIES.get(category)
    if not policy:
        return json.dumps(
            {
                "error": f"No policy found for category '{category}'.",
                "available_categories": sorted(RETURN_POLICIES.keys()),
            }
        )
    return json.dumps({"category": category, **policy})


_SAFE_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_math_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_math_node(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        return _SAFE_OPS[type(node.op)](left, right)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate(args: dict[str, Any]) -> str:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return json.dumps({"error": "Missing 'expression' argument."})
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_math_node(tree.body)
        return json.dumps({"expression": expr, "result": round(value, 2)})
    except Exception as exc:
        return json.dumps({"error": f"Could not evaluate '{expr}': {exc}"})


def build_default_tools() -> dict[str, ToolSpec]:
    tools = [
        ToolSpec(
            name="lookup_order",
            description="Look up order details by order_id (e.g., ORD-1042).",
            args_schema='{"order_id": "string"}',
            handler=lookup_order,
        ),
        ToolSpec(
            name="get_return_policy",
            description="Fetch return window, restocking fee, and shipping refund rules for a product category.",
            args_schema='{"category": "string"}',
            handler=get_return_policy,
        ),
        ToolSpec(
            name="calculate",
            description="Evaluate a basic arithmetic expression (+, -, *, /) for exact refund math.",
            args_schema='{"expression": "string"}',
            handler=calculate,
        ),
    ]
    return {tool.name: tool for tool in tools}


# ---------------------------------------------------------------------------
# Prompt construction and output parsing
# ---------------------------------------------------------------------------


def build_system_prompt(tools: dict[str, ToolSpec]) -> str:
    tool_lines = [
        f"- {t.name}: {t.description} | Input JSON: {t.args_schema}"
        for t in tools.values()
    ]
    tool_block = "\n".join(tool_lines)
    return (
        "You are a precise customer operations assistant using the ReAct pattern.\n"
        "You have access to these tools:\n"
        f"{tool_block}\n\n"
        "For EVERY turn, respond using ONE of these two exact formats.\n\n"
        "Format A (when you need a tool):\n"
        "Thought: <brief reasoning about what you know and which tool to call next>\n"
        "Action: <exact tool name>\n"
        "Action Input: <valid single-line JSON object>\n\n"
        "Format B (when you have enough verified facts to answer the user):\n"
        "Thought: <brief verification that all numbers and policy rules are checked>\n"
        "Final Answer: <clear, concise answer for the user with no internal scratchpad>\n\n"
        "Rules:\n"
        "- Never invent order data, policy rules, or mental arithmetic.\n"
        "- Emit ONLY ONE Action per turn and stop immediately after Action Input.\n"
        "- Never include both Action and Final Answer in the same turn."
    )


_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_\-]+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.*?)(?=\nObservation:|\Z)", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def parse_react_turn(text: str) -> ParsedTurn:
    """Parse structured Thought / Action / Action Input or Final Answer fields."""
    cleaned = text.strip()
    thought_match = _THOUGHT_RE.search(cleaned)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = _FINAL_RE.search(cleaned)
    action_match = _ACTION_RE.search(cleaned)

    if final_match and action_match:
        return ParsedTurn(
            thought=thought,
            raw_text=cleaned,
            parse_error="Model emitted both Action and Final Answer in the same turn.",
        )

    if final_match:
        return ParsedTurn(
            thought=thought,
            final_answer=final_match.group(1).strip(),
            raw_text=cleaned,
        )

    if action_match:
        action_name = action_match.group(1).strip()
        input_match = _ACTION_INPUT_RE.search(cleaned)
        if not input_match:
            return ParsedTurn(
                thought=thought,
                action=action_name,
                raw_text=cleaned,
                parse_error="Missing 'Action Input:' line for tool call.",
            )
        raw_json = input_match.group(1).strip()
        # Strip optional markdown code fences around JSON
        if raw_json.startswith("```"):
            raw_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json, flags=re.DOTALL).strip()
        try:
            parsed_input = json.loads(raw_json)
            if not isinstance(parsed_input, dict):
                raise ValueError("Action Input must be a JSON object.")
        except Exception as exc:
            return ParsedTurn(
                thought=thought,
                action=action_name,
                raw_text=cleaned,
                parse_error=f"Invalid Action Input JSON ({exc}): {raw_json}",
            )
        return ParsedTurn(
            thought=thought,
            action=action_name,
            action_input=parsed_input,
            raw_text=cleaned,
        )

    return ParsedTurn(
        thought=thought,
        raw_text=cleaned,
        parse_error="Response did not contain 'Action:' or 'Final Answer:'.",
    )


# ---------------------------------------------------------------------------
# LLM backends (Live OpenAI-compatible HTTP + Deterministic Demo)
# ---------------------------------------------------------------------------


class OpenAICompatibleClient:
    """Minimal stdlib client for OpenAI-compatible /chat/completions endpoints."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 30) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "stop": ["\nObservation:"],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach LLM endpoint {url}: {exc}") from exc

        return str(data["choices"][0]["message"]["content"])


class DemoReActLLM:
    """Deterministic local mock model showing a realistic 4-turn ReAct trajectory."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        user_query = messages[1]["content"]
        observations = [
            m["content"] for m in messages[2:] if m["role"] == "user" and m["content"].startswith("Observation:")
        ]

        # Extract any order ID mentioned in the query, defaulting to ORD-1042
        match = re.search(r"ORD-\d+", user_query.upper())
        order_id = match.group(0) if match else "ORD-1042"

        if len(observations) == 0:
            return (
                f"Thought: I need to check the order details for {order_id} first "
                "(item category, delivery date, opened status, and amounts) before applying any policy.\n"
                "Action: lookup_order\n"
                f'Action Input: {{"order_id": "{order_id}"}}'
            )

        order_data = json.loads(observations[0].removeprefix("Observation:").strip())
        if "error" in order_data:
            return (
                "Thought: The order does not exist in our system, so I should tell the user clearly "
                "and ask them to verify the order ID.\n"
                f"Final Answer: I could not find order {order_id}. Please double-check the order number and try again."
            )

        if len(observations) == 1:
            category = order_data["category"]
            return (
                f"Thought: Order {order_id} is in the '{category}' category, delivered "
                f"{order_data['days_since_delivery']} days ago, opened={order_data['opened']}. "
                "I should look up the exact return policy for this category instead of guessing.\n"
                "Action: get_return_policy\n"
                f'Action Input: {{"category": "{category}"}}'
            )

        policy_data = json.loads(observations[1].removeprefix("Observation:").strip())
        days = order_data["days_since_delivery"]
        window = policy_data["return_window_days"]

        if days > window:
            return (
                f"Thought: The item was delivered {days} days ago, which exceeds the {window}-day return window. "
                "No refund calculation is needed because the order is ineligible.\n"
                f"Final Answer: Order {order_id} ({order_data['item']}) is not eligible for a return. "
                f"It was delivered {days} days ago, which is past the {window}-day return window for {order_data['category']}."
            )

        if len(observations) == 2:
            subtotal = order_data["subtotal_usd"]
            fee_pct = policy_data["opened_restocking_fee_pct"] if order_data["opened"] else 0
            expr = f"{subtotal} * (1 - {fee_pct} / 100)"
            return (
                f"Thought: The order is within the {window}-day window ({days} days since delivery). "
                f"Because it is opened, a {fee_pct}% restocking fee applies to the ${subtotal:.2f} item subtotal, "
                "and shipping is non-refundable. Let me use the calculator tool so the refund amount is exact.\n"
                "Action: calculate\n"
                f'Action Input: {{"expression": "{expr}"}}'
            )

        calc_data = json.loads(observations[2].removeprefix("Observation:").strip())
        refund = calc_data["result"]
        subtotal = order_data["subtotal_usd"]
        fee_pct = policy_data["opened_restocking_fee_pct"] if order_data["opened"] else 0
        shipping = order_data["shipping_paid_usd"]
        return (
            "Thought: I now have verified order facts, return policy rules, and the calculated refund amount. "
            "I can give a clear user-facing answer without exposing raw tool JSON.\n"
            f"Final Answer: Yes, order {order_id} ({order_data['item']}) is eligible for a return because it was "
            f"delivered {days} days ago (within the {window}-day window for {order_data['category']}). "
            f"Since the item was opened, a {fee_pct}% restocking fee applies to the ${subtotal:.2f} subtotal, "
            f"and the ${shipping:.2f} original shipping charge is non-refundable. "
            f"Your total refund will be ${refund:.2f}."
        )


# ---------------------------------------------------------------------------
# ReAct loop runner
# ---------------------------------------------------------------------------


def run_react_agent(
    query: str,
    llm: OpenAICompatibleClient | DemoReActLLM,
    tools: dict[str, ToolSpec],
    max_steps: int = 6,
    mode: str = "demo",
) -> AgentResult:
    """Execute a bounded Thought -> Action -> Observation loop."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(tools)},
        {"role": "user", "content": query},
    ]
    trace: list[ReActStep] = []

    for step_num in range(1, max_steps + 1):
        raw_reply = llm.complete(messages)
        parsed = parse_react_turn(raw_reply)
        messages.append({"role": "assistant", "content": raw_reply})

        if parsed.parse_error:
            observation = f"Format error: {parsed.parse_error} Please reply using Format A or Format B."
            trace.append(
                ReActStep(
                    step_number=step_num,
                    thought=parsed.thought or "(unparsed output)",
                    action=parsed.action,
                    observation=observation,
                )
            )
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        if parsed.final_answer is not None:
            trace.append(ReActStep(step_number=step_num, thought=parsed.thought))
            return AgentResult(
                query=query,
                final_answer=parsed.final_answer,
                steps=trace,
                finished=True,
                mode=mode,
            )

        assert parsed.action is not None
        tool = tools.get(parsed.action)
        if tool is None:
            valid = ", ".join(sorted(tools.keys()))
            observation = f"Unknown tool '{parsed.action}'. Available tools: {valid}."
        else:
            try:
                observation = tool.handler(parsed.action_input)
            except Exception as exc:
                observation = f"Tool execution error in '{parsed.action}': {exc}"

        trace.append(
            ReActStep(
                step_number=step_num,
                thought=parsed.thought,
                action=parsed.action,
                action_input=parsed.action_input,
                observation=observation,
            )
        )
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return AgentResult(
        query=query,
        final_answer="I reached the maximum number of reasoning steps before finishing. Please narrow the request.",
        steps=trace,
        finished=False,
        mode=mode,
    )


def select_backend(force_demo: bool) -> tuple[OpenAICompatibleClient | DemoReActLLM, str]:
    env_demo = os.getenv("AGENT_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()

    is_local_endpoint = "localhost" in base_url or "127.0.0.1" in base_url
    if force_demo or env_demo or (not api_key and not is_local_endpoint):
        return DemoReActLLM(), "demo"
    return OpenAICompatibleClient(api_key=api_key or "ollama", base_url=base_url, model=model), f"live ({model})"


def print_result(result: AgentResult, show_trace: bool) -> None:
    print(f"Mode : {result.mode}")
    print(f"Query: {result.query}")
    print("-" * 72)

    if show_trace:
        print("INTERNAL REASONING TRACE (hidden from end users in production):")
        for step in result.steps:
            print(f"\n[Step {step.step_number}]")
            print(f"  Thought     : {step.thought}")
            if step.action:
                print(f"  Action      : {step.action}")
                print(f"  Action Input: {json.dumps(step.action_input)}")
            if step.observation:
                print(f"  Observation : {step.observation}")
        print("-" * 72)

    print("USER-FACING FINAL ANSWER:")
    print(result.final_answer)
    print(f"\nStatus: {'completed' if result.finished else 'stopped (max steps)'} in {len(result.steps)} step(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 11: ReAct Reasoning + Acting Agent")
    parser.add_argument(
        "--query",
        default="Can order ORD-1042 be returned, and what exact refund amount would the customer receive?",
        help="User question to solve.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum ReAct iterations.")
    parser.add_argument("--demo", action="store_true", help="Run with deterministic offline mock LLM.")
    parser.add_argument(
        "--hide-trace",
        action="store_true",
        help="Hide internal Thought/Action/Observation trace and show only the final user answer.",
    )
    args = parser.parse_args()

    llm, mode = select_backend(force_demo=args.demo)
    tools = build_default_tools()
    result = run_react_agent(
        query=args.query,
        llm=llm,
        tools=tools,
        max_steps=args.max_steps,
        mode=mode,
    )
    print_result(result, show_trace=not args.hide_trace)


if __name__ == "__main__":
    main()
