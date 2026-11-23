"""Day 1: Compare a deterministic workflow, a single LLM call, and a minimal agent loop.

All three approaches handle the same support request:
    "My order ORD-1042 still hasn't arrived and I paid for express shipping.
     Am I eligible for a shipping refund?"

Run:
    python3 01_Day_Agentic_AI_Foundations/workflow_vs_agent.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Simulated backend store so the script runs offline with zero setup.
ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1042": {
        "status": "in_transit",
        "shipping_tier": "express",
        "days_in_transit": 5,
        "shipping_fee_usd": 18.00,
    },
    "ORD-1099": {
        "status": "delivered",
        "shipping_tier": "standard",
        "days_in_transit": 3,
        "shipping_fee_usd": 5.00,
    },
}

POLICIES: dict[str, dict[str, Any]] = {
    "express": {"max_days": 2, "refund_percent": 100},
    "standard": {"max_days": 7, "refund_percent": 0},
}


# ---------------------------------------------------------------------------
# Approach 1: Deterministic Workflow (hard-coded control flow)
# ---------------------------------------------------------------------------
def deterministic_workflow(order_id: str) -> str:
    """Fast, cheap, predictable—but rigid. Only handles the exact path we coded."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"Order {order_id} was not found."

    policy = POLICIES.get(order["shipping_tier"])
    if policy is None:
        return f"No refund policy configured for tier {order['shipping_tier']}."

    if order["days_in_transit"] > policy["max_days"] and policy["refund_percent"] > 0:
        refund_amount = order["shipping_fee_usd"] * (policy["refund_percent"] / 100.0)
        return (
            f"Order {order_id} exceeded the {policy['max_days']}-day express window "
            f"({order['days_in_transit']} days in transit). Eligible refund: ${refund_amount:.2f}."
        )
    return f"Order {order_id} is not eligible for a shipping refund."


# ---------------------------------------------------------------------------
# Approach 2: Single-Turn Generative Call (LLM without tools or live data)
# ---------------------------------------------------------------------------
def single_turn_llm_call(user_message: str) -> str:
    """Flexible prose, but blind to live order state unless you pre-fetch and inject it."""
    # Simulated response of an LLM that has no tool access to ORDERS or POLICIES.
    return (
        "I am sorry your order has been delayed. Express shipping delays are often "
        "eligible for a refund depending on our policy and how many days the package "
        "has been in transit. Please check your order tracking page or contact billing "
        "with your order ID so they can verify eligibility."
    )


# ---------------------------------------------------------------------------
# Approach 3: Minimal Agent Loop (decides which tools to call based on observations)
# ---------------------------------------------------------------------------
def tool_get_order(order_id: str) -> dict[str, Any]:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Order {order_id} not found"}
    return {"order_id": order_id, **order}


def tool_get_policy(shipping_tier: str) -> dict[str, Any]:
    policy = POLICIES.get(shipping_tier)
    if policy is None:
        return {"error": f"Unknown shipping tier {shipping_tier}"}
    return {"shipping_tier": shipping_tier, **policy}


@dataclass
class AgentState:
    """Tracks what the minimal agent has learned across steps."""

    goal: str
    order_id: str
    observations: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


def decide_next_action(state: AgentState) -> tuple[str, dict[str, Any]]:
    """Stand-in for an LLM planner that inspects observations and picks the next step.

    In Day 4 and Day 5, an actual LLM produces these tool decisions via function calling.
    """
    if "order" not in state.observations:
        return ("get_order", {"order_id": state.order_id})

    order = state.observations["order"]
    if "error" in order:
        return ("finish", {"answer": f"I could not look up {state.order_id}: {order['error']}"})

    if "policy" not in state.observations:
        return ("get_policy", {"shipping_tier": order["shipping_tier"]})

    policy = state.observations["policy"]
    overdue = order["days_in_transit"] > policy["max_days"]
    refund = order["shipping_fee_usd"] * (policy["refund_percent"] / 100.0) if overdue else 0.0

    if refund > 0:
        answer = (
            f"Yes. Order {order['order_id']} uses {order['shipping_tier']} shipping "
            f"(limit: {policy['max_days']} days) and has been in transit for "
            f"{order['days_in_transit']} days. You are eligible for a ${refund:.2f} refund."
        )
    else:
        answer = (
            f"Order {order['order_id']} is currently within policy limits and is not "
            f"eligible for a shipping refund."
        )
    return ("finish", {"answer": answer})


def minimal_agent_loop(goal: str, order_id: str, max_steps: int = 5) -> tuple[str, list[str]]:
    """Run a bounded decide -> act -> observe loop until completion or max_steps."""
    state = AgentState(goal=goal, order_id=order_id)

    for step in range(1, max_steps + 1):
        action, args = decide_next_action(state)
        state.trace.append(f"Step {step}: action={action} args={args}")

        if action == "finish":
            return args["answer"], state.trace

        if action == "get_order":
            result = tool_get_order(args["order_id"])
            state.observations["order"] = result
            state.trace.append(f"         observation={result}")
        elif action == "get_policy":
            result = tool_get_policy(args["shipping_tier"])
            state.observations["policy"] = result
            state.trace.append(f"         observation={result}")
        else:
            return f"Stopped: unknown action {action}", state.trace

    return "Stopped: reached max_steps without finishing.", state.trace


def main() -> None:
    user_request = (
        "My order ORD-1042 still hasn't arrived and I paid for express shipping. "
        "Am I eligible for a shipping refund?"
    )
    print(f"User Request:\n  {user_request}\n")

    print("1) Deterministic Workflow (hard-coded rules)")
    print(f"   Output: {deterministic_workflow('ORD-1042')}\n")

    print("2) Single-Turn Generative LLM Call (no tools)")
    print(f"   Output: {single_turn_llm_call(user_request)}\n")

    print("3) Minimal Agent Loop (dynamic tool selection + observations)")
    answer, trace = minimal_agent_loop(goal=user_request, order_id="ORD-1042")
    for line in trace:
        print(f"   {line}")
    print(f"   Final Answer: {answer}")


if __name__ == "__main__":
    main()
