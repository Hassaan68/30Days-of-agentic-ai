"""Day 25: Observability, Tracing, and Telemetry for Agent Runs.

Implements a lightweight, OpenTelemetry-aligned context-manager tracer for
multi-step agent workflows:
- W3C-compatible 128-bit `trace_id` and 64-bit `span_id` with `contextvars`
  parent propagation across nested function calls.
- Captures latency, token counts, model cost estimates, span events, and errors.
- Exports structured JSONL traces compatible with OpenTelemetry GenAI concepts.
- Renders a terminal ASCII waterfall timeline and summary metrics breakdown.

Run directly from the repository root:
    python3 25_Day_Observability_Tracing_and_Telemetry/agent_tracer.py
"""

from __future__ import annotations

import argparse
import contextvars
import json
import secrets
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


# USD per 1,000,000 tokens: (input_price, output_price)
MODEL_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "llama3.2": (0.00, 0.00),  # Local Ollama model
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a model call from token usage."""
    in_price, out_price = MODEL_PRICING_PER_1M_TOKENS.get(model, (0.0, 0.0))
    cost = (input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price
    return round(cost, 8)


@dataclass
class SpanEvent:
    """Timestamped annotation attached to a span (e.g., retry or exception)."""

    name: str
    timestamp_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanRecord:
    """Completed span exported to JSONL and rendered in the waterfall."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str  # "agent", "step", "llm", "tool", "guardrail"
    start_time_ns: int
    end_time_ns: int = 0
    duration_ms: float = 0.0
    status: str = "OK"  # "OK" or "ERROR"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(
            SpanEvent(
                name=name,
                timestamp_ns=time.time_ns(),
                attributes=attributes or {},
            )
        )

    def record_llm_usage(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: str = "openai",
    ) -> float:
        """Record OpenTelemetry GenAI-style usage attributes and return estimated cost."""
        cost_usd = estimate_cost_usd(model, input_tokens, output_tokens)
        self.attributes.update(
            {
                "gen_ai.system": provider,
                "gen_ai.request.model": model,
                "gen_ai.usage.input_tokens": input_tokens,
                "gen_ai.usage.output_tokens": output_tokens,
                "gen_ai.usage.total_tokens": input_tokens + output_tokens,
                "gen_ai.usage.cost_usd": cost_usd,
            }
        )
        return cost_usd


@dataclass(frozen=True)
class TraceSummary:
    """Aggregated performance, token, cost, and error metrics for one trace."""

    trace_id: str
    total_wall_ms: float
    span_count: int
    llm_calls: int
    tool_calls: int
    error_spans: int
    llm_time_ms: float
    tool_time_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class AgentTracer:
    """Context-manager tracer capturing nested spans across an agent run."""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or secrets.token_hex(16)  # 128-bit W3C trace ID
        self._spans: list[SpanRecord] = []
        self._active_span_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"active_span_{self.trace_id}", default=None
        )

    @property
    def spans(self) -> list[SpanRecord]:
        return list(self._spans)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[SpanRecord]:
        """Start a nested span, track its timing/errors, and close it cleanly."""
        parent_id = self._active_span_var.get()
        record = SpanRecord(
            trace_id=self.trace_id,
            span_id=secrets.token_hex(8),  # 64-bit W3C span ID
            parent_span_id=parent_id,
            name=name,
            kind=kind,
            start_time_ns=time.time_ns(),
            attributes=dict(attributes or {}),
        )
        self._spans.append(record)
        token = self._active_span_var.set(record.span_id)
        t0 = time.perf_counter()

        try:
            yield record
        except BaseException as exc:
            record.status = "ERROR"
            record.error_type = type(exc).__name__
            record.error_message = str(exc)
            record.add_event(
                "exception",
                {"exception.type": record.error_type, "exception.message": record.error_message},
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            record.end_time_ns = record.start_time_ns + int(elapsed_ms * 1_000_000)
            record.duration_ms = round(elapsed_ms, 2)
            self._active_span_var.reset(token)

    def summarize(self) -> TraceSummary:
        """Compute rollup metrics across all recorded spans."""
        if not self._spans:
            return TraceSummary(self.trace_id, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0.0)

        trace_start = min(s.start_time_ns for s in self._spans)
        trace_end = max(s.end_time_ns for s in self._spans)
        total_wall_ms = round((trace_end - trace_start) / 1_000_000.0, 2)

        llm_spans = [s for s in self._spans if s.kind == "llm"]
        tool_spans = [s for s in self._spans if s.kind == "tool"]
        error_spans = [s for s in self._spans if s.status == "ERROR"]

        in_tokens = sum(int(s.attributes.get("gen_ai.usage.input_tokens", 0)) for s in llm_spans)
        out_tokens = sum(int(s.attributes.get("gen_ai.usage.output_tokens", 0)) for s in llm_spans)
        total_cost = sum(float(s.attributes.get("gen_ai.usage.cost_usd", 0.0)) for s in llm_spans)

        return TraceSummary(
            trace_id=self.trace_id,
            total_wall_ms=total_wall_ms,
            span_count=len(self._spans),
            llm_calls=len(llm_spans),
            tool_calls=len(tool_spans),
            error_spans=len(error_spans),
            llm_time_ms=round(sum(s.duration_ms for s in llm_spans), 2),
            tool_time_ms=round(sum(s.duration_ms for s in tool_spans), 2),
            total_input_tokens=in_tokens,
            total_output_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
            estimated_cost_usd=round(total_cost, 6),
        )

    def export_jsonl(self, output_path: Path) -> Path:
        """Write all spans to a structured JSONL file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in self._spans:
                fh.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return path

    def render_waterfall(self, bar_width: int = 38) -> str:
        """Render an indented ASCII timeline waterfall of the trace."""
        if not self._spans:
            return "(no spans recorded)"

        trace_start = min(s.start_time_ns for s in self._spans)
        trace_end = max(s.end_time_ns for s in self._spans)
        total_ns = max(1, trace_end - trace_start)

        children_map: dict[str | None, list[SpanRecord]] = {}
        for s in self._spans:
            children_map.setdefault(s.parent_span_id, []).append(s)
        for siblings in children_map.values():
            siblings.sort(key=lambda sp: sp.start_time_ns)

        # Build depth-first flattened list with tree prefixes
        rows: list[tuple[str, SpanRecord]] = []

        def _walk(parent_id: str | None, prefix: str) -> None:
            kids = children_map.get(parent_id, [])
            for idx, child in enumerate(kids):
                is_last = idx == len(kids) - 1
                if parent_id is None:
                    label = child.name
                    next_prefix = ""
                else:
                    connector = "└─ " if is_last else "├─ "
                    label = f"{prefix}{connector}{child.name}"
                    next_prefix = prefix + ("   " if is_last else "│  ")
                rows.append((label, child))
                _walk(child.span_id, next_prefix)

        _walk(None, "")

        label_col = max(28, min(42, max(len(lbl) for lbl, _ in rows) + 1))
        header = (
            f"Trace ID: {self.trace_id}  (total wall time: {total_ns / 1_000_000.0:.1f} ms)\n"
            f"{'Span':<{label_col}} | {'Timeline':<{bar_width}} | {'Dur (ms)':>8} | Details\n"
            + "-" * (label_col + bar_width + 44)
        )

        lines = [header]
        for label, sp in rows:
            rel_start = (sp.start_time_ns - trace_start) / total_ns
            rel_end = (sp.end_time_ns - trace_start) / total_ns

            start_col = min(bar_width - 1, max(0, int(rel_start * bar_width)))
            end_col = min(bar_width, max(start_col + 1, int(round(rel_end * bar_width))))

            fill = "!" if sp.status == "ERROR" else "█"
            bar = ("." * start_col) + (fill * (end_col - start_col)) + ("." * (bar_width - end_col))

            details_parts: list[str] = []
            if sp.status == "ERROR":
                details_parts.append(f"[ERR: {sp.error_type}]")
            if "gen_ai.usage.total_tokens" in sp.attributes:
                toks = sp.attributes["gen_ai.usage.total_tokens"]
                cost = sp.attributes.get("gen_ai.usage.cost_usd", 0.0)
                details_parts.append(f"{toks} tok (${cost:.5f})")
            if "tool.attempt" in sp.attributes:
                details_parts.append(f"attempt={sp.attributes['tool.attempt']}")
            if sp.kind == "guardrail" and "redactions" in sp.attributes:
                details_parts.append(f"redactions={sp.attributes['redactions']}")

            trimmed_label = label if len(label) <= label_col else label[: label_col - 1] + "…"
            lines.append(
                f"{trimmed_label:<{label_col}} | {bar} | {sp.duration_ms:8.1f} | {' '.join(details_parts)}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulated multi-step agent workflow for runnable demonstration
# ---------------------------------------------------------------------------

def _simulate_llm_call(
    tracer: AgentTracer,
    step_name: str,
    model: str,
    in_tokens: int,
    out_tokens: int,
    latency_s: float,
) -> dict[str, Any]:
    with tracer.span(f"llm.chat_completion ({step_name})", kind="llm") as sp:
        time.sleep(latency_s)
        sp.record_llm_usage(model=model, input_tokens=in_tokens, output_tokens=out_tokens)
        return {"model": model, "tokens": in_tokens + out_tokens}


def _simulate_fetch_order(tracer: AgentTracer, order_id: str, attempt: int) -> dict[str, Any]:
    with tracer.span(
        "tool.fetch_order",
        kind="tool",
        attributes={"tool.name": "fetch_order", "tool.attempt": attempt, "order.id": order_id},
    ) as sp:
        if attempt == 1:
            time.sleep(0.055)
            raise TimeoutError("Upstream order service timed out after 55ms")
        time.sleep(0.025)
        sp.set_attribute("http.status_code", 200)
        return {"order_id": order_id, "sku": "SKU-440", "zip": "94107"}


def run_traced_agent_demo(export_path: Path | None = None) -> None:
    """Run a 3-step agent workflow with a retried tool failure and print the waterfall."""
    tracer = AgentTracer()

    with tracer.span(
        "agent.run",
        kind="agent",
        attributes={"session.id": "sess_trace_25", "user.intent": "check_replacement_shipping"},
    ):
        # Step 1: Plan + fetch order (with one simulated transient failure + retry)
        with tracer.span("step.1_fetch_context", kind="step"):
            _simulate_llm_call(tracer, "plan", "gpt-4o-mini", 410, 68, 0.045)

            order_data: dict[str, Any] | None = None
            for attempt in (1, 2):
                try:
                    order_data = _simulate_fetch_order(tracer, "ORD-7741", attempt=attempt)
                    break
                except TimeoutError:
                    # Exception was already captured on the failed tool span; retry cleanly.
                    continue
            assert order_data is not None

        # Step 2: Decide follow-up lookups + call two downstream tools
        with tracer.span("step.2_verify_fulfillment", kind="step"):
            _simulate_llm_call(tracer, "tool_select", "gpt-4o-mini", 620, 95, 0.050)

            with tracer.span(
                "tool.check_inventory",
                kind="tool",
                attributes={"tool.name": "check_inventory", "sku": order_data["sku"]},
            ) as sp:
                time.sleep(0.030)
                sp.set_attribute("inventory.in_stock", True)

            with tracer.span(
                "tool.calculate_shipping",
                kind="tool",
                attributes={"tool.name": "calculate_shipping", "zip": order_data["zip"]},
            ) as sp:
                time.sleep(0.035)
                sp.set_attribute("shipping.eta_days", 2)

        # Step 3: Draft final answer + run output guardrail check
        with tracer.span("step.3_synthesize_reply", kind="step"):
            _simulate_llm_call(tracer, "final_answer", "gpt-4o-mini", 890, 140, 0.060)

            with tracer.span("guardrail.output_scan", kind="guardrail") as sp:
                time.sleep(0.008)
                sp.set_attribute("redactions", 0)

    print("=== Day 25: Agent Observability & Trace Waterfall ===\n")
    print(tracer.render_waterfall())

    summary = tracer.summarize()
    print("\n=== Trace Summary Metrics ===")
    print(f"  Trace ID:         {summary.trace_id}")
    print(f"  Total wall time:  {summary.total_wall_ms:.1f} ms")
    print(f"  Spans recorded:   {summary.span_count} (LLM: {summary.llm_calls}, Tools: {summary.tool_calls}, Errors: {summary.error_spans})")
    print(f"  LLM latency sum:  {summary.llm_time_ms:.1f} ms")
    print(f"  Tool latency sum: {summary.tool_time_ms:.1f} ms")
    print(f"  Tokens used:      {summary.total_tokens} (in={summary.total_input_tokens}, out={summary.total_output_tokens})")
    print(f"  Est. model cost:  ${summary.estimated_cost_usd:.6f}")

    # Export JSONL and verify round-trip parsing
    if export_path is not None:
        out_file = tracer.export_jsonl(export_path)
        exported_lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"\nExported {len(exported_lines)} spans to: {out_file}")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = tracer.export_jsonl(Path(tmpdir) / "trace.jsonl")
            exported_lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            print(f"\nVerified JSONL export ({len(exported_lines)} spans written and parsed cleanly).")

    # Assertions verifying trace integrity
    assert summary.span_count == 12
    assert summary.llm_calls == 3
    assert summary.tool_calls == 4  # includes the failed attempt + retry
    assert summary.error_spans == 1
    assert summary.total_input_tokens == 410 + 620 + 890
    assert summary.total_output_tokens == 68 + 95 + 140
    assert len(exported_lines) == summary.span_count
    print("All trace integrity assertions passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 25 Agent Observability Tracer")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional path to save exported JSONL trace spans.",
    )
    args = parser.parse_args()
    run_traced_agent_demo(export_path=args.export)


if __name__ == "__main__":
    main()
