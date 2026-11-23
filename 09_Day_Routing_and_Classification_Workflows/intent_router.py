"""Day 9: Routing and Classification Workflows.

Demonstrates a two-stage intent router that dispatches incoming requests across:
1. Zero-cost deterministic Python handlers (order lookup, safe AST calculator).
2. Specialized LLM handlers (billing FAQ, technical support).
3. Policy escalation and confidence-gated fallback routes for ambiguous inputs.

Run offline demo (no API key needed):
    python 09_Day_Routing_and_Classification_Workflows/intent_router.py --demo

Run a single query against a live OpenAI-compatible endpoint:
    export OPENAI_API_KEY="your-key"
    python 09_Day_Routing_and_Classification_Workflows/intent_router.py \
        --query "Why did my webhook return HTTP 429 in eu-west-1?"
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Typed routing models
# ---------------------------------------------------------------------------

VALID_INTENTS = (
    "order_status",
    "calculator",
    "billing_faq",
    "tech_support",
    "refund_escalation",
    "unknown",
)


@dataclass
class RouteDecision:
    """Structured output produced by either deterministic rules or the classifier."""

    intent: str
    confidence: float
    source: str  # "deterministic_rule" | "llm_classifier"
    reasoning: str
    entities: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteResult:
    """Complete execution result and telemetry for one routed user query."""

    query: str
    decision: RouteDecision
    executed_route: str
    handler_type: str  # "deterministic_python" | "fast_llm" | "specialist_llm" | "fallback"
    response: str
    latency_ms: float
    estimated_cost_usd: float


# ---------------------------------------------------------------------------
# Safe deterministic backends (zero LLM tokens)
# ---------------------------------------------------------------------------

ORDERS_DB: dict[str, dict[str, str]] = {
    "ORD-1042": {"status": "delivered", "carrier": "UPS", "eta": "Delivered Nov 21"},
    "ORD-2088": {"status": "in_transit", "carrier": "FedEx", "eta": "Nov 25 by 18:00"},
    "ORD-3190": {"status": "processing", "carrier": "Pending", "eta": "Ships within 24h"},
}

_SAFE_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_SAFE_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_math(expression: str) -> float:
    """Evaluate a basic arithmetic expression safely using Python's AST (no eval())."""
    tree = ast.parse(expression.strip(), mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large.")
            return _SAFE_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
            return _SAFE_UNARYOPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported syntax node: {type(node).__name__}")

    return _eval(tree)


# ---------------------------------------------------------------------------
# Stage 1: Deterministic pre-router (sub-millisecond, $0 cost)
# ---------------------------------------------------------------------------

_ORDER_ID_RE = re.compile(r"\b(ORD-\d{4})\b", re.IGNORECASE)
_ORDER_STATUS_WORDS = ("status", "track", "where", "shipping", "delivery", "eta", "package")
_PURE_MATH_RE = re.compile(r"^\s*(?:calculate|compute|what is|eval)?\s*([\d\.\s\+\-\*\/\(\)]+?)\??\s*$", re.IGNORECASE)


def try_deterministic_route(query: str) -> RouteDecision | None:
    """Match unambiguous requests before spending tokens on an LLM classifier."""
    lower = query.lower()

    # Rule 1: Order status lookup with explicit ORD-XXXX ID (unless asking for a refund).
    order_match = _ORDER_ID_RE.search(query)
    if order_match and any(w in lower for w in _ORDER_STATUS_WORDS) and "refund" not in lower:
        return RouteDecision(
            intent="order_status",
            confidence=1.0,
            source="deterministic_rule",
            reasoning="Explicit order ID + shipping/status keyword matched.",
            entities={"order_id": order_match.group(1).upper()},
        )

    # Rule 2: Pure arithmetic expression.
    pure_match = _PURE_MATH_RE.match(query)
    if pure_match:
        expr = pure_match.group(1).strip()
        if any(op in expr for op in ("+", "-", "*", "/")):
            return RouteDecision(
                intent="calculator",
                confidence=1.0,
                source="deterministic_rule",
                reasoning="Direct arithmetic expression matched.",
                entities={"expression": expr},
            )

    return None


# ---------------------------------------------------------------------------
# LLM provider abstraction (live OpenAI-compatible + offline demo mock)
# ---------------------------------------------------------------------------

class OpenAICompatibleLLM:
    """Minimal stdlib client for OpenAI-compatible /chat/completions endpoints."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def complete(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            return parsed["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            # Some local/older endpoints reject response_format; retry once without it.
            if json_mode and exc.code == 400:
                return self.complete(system_prompt, user_prompt, json_mode=False)
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM network error: {exc.reason}") from exc


class DemoMockLLM:
    """Offline deterministic classifier + specialized handler mock for --demo."""

    def complete(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        if json_mode:
            return json.dumps(self._classify(user_prompt))
        if "BILLING_FAQ_SPECIALIST" in system_prompt:
            return (
                "Invoices are generated on the 1st of each month and can be downloaded as PDFs "
                "under Settings -> Billing -> Invoice History. You can also add a VAT/Tax ID there."
            )
        if "TECH_SUPPORT_SPECIALIST" in system_prompt:
            return (
                "HTTP 429 means your webhook client exceeded the per-minute rate limit (120 req/min). "
                "Read the `Retry-After` response header, apply exponential backoff with jitter, "
                "and batch non-urgent webhook events."
            )
        return "Handled by mock specialist."

    @staticmethod
    def _classify(query: str) -> dict[str, Any]:
        q = query.lower()
        order_match = _ORDER_ID_RE.search(query)
        order_id = order_match.group(1).upper() if order_match else None

        if any(w in q for w in ("refund", "chargeback", "cancel and reimburse", "money back")):
            return {
                "intent": "refund_escalation",
                "confidence": 0.94,
                "reasoning": "User requests a refund or reimbursement.",
                "entities": {"order_id": order_id} if order_id else {},
            }
        if any(w in q for w in ("invoice", "vat", "receipt", "billing", "subscription", "credit card")):
            return {
                "intent": "billing_faq",
                "confidence": 0.91,
                "reasoning": "General billing/invoice documentation question.",
                "entities": {},
            }
        if any(w in q for w in ("429", "502", "webhook", "api key", "timeout", "sdk", "error", "rate limit")):
            return {
                "intent": "tech_support",
                "confidence": 0.89,
                "reasoning": "API/webhook technical troubleshooting question.",
                "entities": {},
            }
        if any(w in q for w in ("track", "order", "package", "shipment")):
            return {
                "intent": "order_status",
                "confidence": 0.82,
                "reasoning": "User asks about an order without providing an order ID.",
                "entities": {"order_id": order_id} if order_id else {},
            }
        # Deliberately low confidence for vague/ambiguous inputs.
        return {
            "intent": "unknown",
            "confidence": 0.42,
            "reasoning": "Input is ambiguous and could apply to multiple support queues.",
            "entities": {},
        }


# ---------------------------------------------------------------------------
# IntentRouter: classify -> confidence gate -> dispatch specialized handler
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = f"""You are a strict intent classifier for a customer & developer support router.
Return ONLY valid JSON with these exact keys:
- "intent": one of {list(VALID_INTENTS)}
- "confidence": float between 0.0 and 1.0
- "reasoning": short 1-sentence justification
- "entities": object with optional keys like "order_id"

Use low confidence (< 0.65) or "unknown" when the user's message is vague or spans unrelated requests."""


class IntentRouter:
    """Two-stage router combining deterministic rules, LLM classification, and safe fallback."""

    def __init__(
        self,
        classifier_llm: OpenAICompatibleLLM | DemoMockLLM,
        fast_llm: OpenAICompatibleLLM | DemoMockLLM,
        specialist_llm: OpenAICompatibleLLM | DemoMockLLM,
        confidence_threshold: float = 0.72,
    ) -> None:
        self.classifier_llm = classifier_llm
        self.fast_llm = fast_llm
        self.specialist_llm = specialist_llm
        self.confidence_threshold = confidence_threshold

    def classify(self, query: str) -> tuple[RouteDecision, float]:
        """Classify query and return (RouteDecision, classifier_estimated_cost_usd)."""
        rule_decision = try_deterministic_route(query)
        if rule_decision is not None:
            return rule_decision, 0.0

        raw = self.classifier_llm.complete(
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=query,
            json_mode=True,
        )
        classifier_cost = 0.00015  # ~300 tokens on a small classifier model

        try:
            parsed = json.loads(raw)
            intent = str(parsed.get("intent", "unknown"))
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = str(parsed.get("reasoning", ""))
            entities = parsed.get("entities") if isinstance(parsed.get("entities"), dict) else {}
            if intent not in VALID_INTENTS:
                intent = "unknown"
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
        except (json.JSONDecodeError, ValueError, TypeError):
            return (
                RouteDecision(
                    intent="unknown",
                    confidence=0.0,
                    source="llm_classifier",
                    reasoning="Classifier returned malformed JSON; routing to fallback.",
                ),
                classifier_cost,
            )

        return (
            RouteDecision(
                intent=intent,
                confidence=confidence,
                source="llm_classifier",
                reasoning=reasoning,
                entities=entities,
            ),
            classifier_cost,
        )

    def route(self, query: str) -> RouteResult:
        """Classify and execute the appropriate downstream handler."""
        t0 = time.perf_counter()
        decision, cost = self.classify(query)

        # Confidence threshold gate: never guess when uncertain.
        if decision.intent == "unknown" or decision.confidence < self.confidence_threshold:
            response = self._handle_fallback(query, decision)
            return RouteResult(
                query=query,
                decision=decision,
                executed_route="fallback_clarify",
                handler_type="fallback",
                response=response,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                estimated_cost_usd=cost,
            )

        if decision.intent == "order_status":
            response = self._handle_order_status(decision.entities)
            handler_type = "deterministic_python"
        elif decision.intent == "calculator":
            response = self._handle_calculator(decision.entities)
            handler_type = "deterministic_python"
        elif decision.intent == "refund_escalation":
            response = self._handle_refund_escalation(query, decision.entities)
            handler_type = "deterministic_python"
        elif decision.intent == "billing_faq":
            response = self.fast_llm.complete(
                system_prompt=(
                    "BILLING_FAQ_SPECIALIST: Answer billing and invoice questions concisely "
                    "in 2-3 sentences based on standard SaaS billing workflows."
                ),
                user_prompt=query,
            )
            handler_type = "fast_llm"
            cost += 0.0003
        elif decision.intent == "tech_support":
            response = self.specialist_llm.complete(
                system_prompt=(
                    "TECH_SUPPORT_SPECIALIST: Provide concrete debugging steps for API, "
                    "webhook, and HTTP error questions."
                ),
                user_prompt=query,
            )
            handler_type = "specialist_llm"
            cost += 0.0025
        else:
            response = self._handle_fallback(query, decision)
            handler_type = "fallback"

        return RouteResult(
            query=query,
            decision=decision,
            executed_route=decision.intent,
            handler_type=handler_type,
            response=response,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            estimated_cost_usd=cost,
        )

    @staticmethod
    def _handle_order_status(entities: dict[str, Any]) -> str:
        order_id = str(entities.get("order_id") or "").upper()
        if not order_id:
            return "Please share your order ID (format ORD-1234) so I can check live shipping status."
        order = ORDERS_DB.get(order_id)
        if not order:
            return f"I could not find order {order_id}. Double-check the ID or contact support."
        return (
            f"Order {order_id}: status={order['status']}, carrier={order['carrier']}, "
            f"ETA={order['eta']}."
        )

    @staticmethod
    def _handle_calculator(entities: dict[str, Any]) -> str:
        expr = str(entities.get("expression", "")).strip()
        try:
            value = safe_eval_math(expr)
            formatted = f"{value:.4f}".rstrip("0").rstrip(".")
            return f"{expr} = {formatted}"
        except Exception as exc:
            return f"Could not safely evaluate '{expr}': {exc}"

    @staticmethod
    def _handle_refund_escalation(query: str, entities: dict[str, Any]) -> str:
        order_id = entities.get("order_id") or "unspecified"
        return (
            f"Created priority refund triage ticket for order={order_id}. "
            "A billing specialist will review eligibility and follow up within 4 business hours."
        )

    @staticmethod
    def _handle_fallback(query: str, decision: RouteDecision) -> str:
        return (
            f"I want to make sure I route this correctly (classifier confidence was "
            f"{decision.confidence:.2f}). Are you asking about (1) order tracking, "
            "(2) invoices/billing, (3) a refund request, or (4) API technical support?"
        )


# ---------------------------------------------------------------------------
# CLI & runnable demonstration
# ---------------------------------------------------------------------------

def _should_use_demo(force_demo: bool) -> bool:
    if force_demo or os.getenv("AGENT_DEMO_MODE", "").strip() == "1":
        return True
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    return not api_key and not is_local


def _print_result(idx: int, res: RouteResult) -> None:
    print(f"[{idx}] Query: {res.query}")
    print(
        f"    Route: {res.executed_route} (intent={res.decision.intent}, "
        f"conf={res.decision.confidence:.2f}, via={res.decision.source}, "
        f"handler={res.handler_type})"
    )
    print(f"    Cost: ${res.estimated_cost_usd:.5f} | Latency: {res.latency_ms:.2f}ms")
    print(f"    Response: {res.response}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 9: Intent routing & classification workflow")
    parser.add_argument("--demo", action="store_true", help="Run offline deterministic routing demo")
    parser.add_argument("--query", type=str, default=None, help="Single query to route")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.72,
        help="Minimum confidence required before falling back to clarification.",
    )
    args = parser.parse_args()

    if _should_use_demo(args.demo):
        mock = DemoMockLLM()
        router = IntentRouter(
            classifier_llm=mock,
            fast_llm=mock,
            specialist_llm=mock,
            confidence_threshold=args.threshold,
        )
        mode_label = "offline deterministic demo"
    else:
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        specialist_model = os.getenv("SPECIALIST_MODEL_NAME", model)
        fast = OpenAICompatibleLLM(api_key=api_key, base_url=base_url, model=model)
        specialist = OpenAICompatibleLLM(api_key=api_key, base_url=base_url, model=specialist_model)
        router = IntentRouter(
            classifier_llm=fast,
            fast_llm=fast,
            specialist_llm=specialist,
            confidence_threshold=args.threshold,
        )
        mode_label = f"live models (fast={model}, specialist={specialist_model})"

    print(f"=== Day 9: Routing & Classification Workflow ({mode_label}) ===\n")

    queries = (
        [args.query]
        if args.query
        else [
            "Where is my package for order ORD-2088?",
            "What is (125 * 12) + 450 / 3?",
            "How do I download last month's invoice PDF with our VAT ID?",
            "Our webhook consumer keeps hitting HTTP 429 rate limits in production.",
            "Please refund order ORD-1042, the package arrived damaged.",
            "Can you check my shipment status?",
            "Hey, something seems off since Tuesday, can you help?",
        ]
    )

    total_routed_cost = 0.0
    monolithic_baseline_per_query = 0.0045  # Sending full multi-tool prompt to a large model

    for i, q in enumerate(queries, start=1):
        res = router.route(q)
        total_routed_cost += res.estimated_cost_usd
        _print_result(i, res)

    baseline_total = len(queries) * monolithic_baseline_per_query
    savings_pct = (1.0 - total_routed_cost / baseline_total) * 100.0 if baseline_total > 0 else 0.0
    print("--- Batch Summary ---")
    print(f"Queries routed:        {len(queries)}")
    print(f"Estimated routed cost: ${total_routed_cost:.5f}")
    print(f"Monolithic baseline:   ${baseline_total:.5f} (always calling a large prompt/model)")
    print(f"Estimated cost saved:  {savings_pct:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
