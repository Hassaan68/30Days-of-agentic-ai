"""Nested span tracer with token/cost accounting and ASCII waterfall renderer."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from models import Span


class Tracer:
    """Captures parent/child execution spans across an agent run."""

    def __init__(self) -> None:
        self._origin = time.perf_counter()
        self._stack: list[str] = []
        self.spans: list[Span] = []

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        span_id = uuid.uuid4().hex[:8]
        parent_id = self._stack[-1] if self._stack else None
        start_perf = time.perf_counter()
        start_offset_ms = (start_perf - self._origin) * 1000.0

        record = Span(
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            start_offset_ms=start_offset_ms,
            duration_ms=0.0,
            attributes=dict(attributes or {}),
        )
        self.spans.append(record)
        self._stack.append(span_id)
        try:
            yield record
        except Exception as exc:
            record.status = "error"
            record.attributes["error"] = str(exc)
            raise
        finally:
            record.duration_ms = (time.perf_counter() - start_perf) * 1000.0
            self._stack.pop()

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def render_waterfall(self) -> str:
        """Render nested spans as an indented ASCII timeline with token/cost totals."""
        if not self.spans:
            return "(no spans recorded)"

        depth_by_id: dict[str, int] = {}
        for sp in self.spans:
            depth = 0 if sp.parent_id is None else depth_by_id.get(sp.parent_id, 0) + 1
            depth_by_id[sp.span_id] = depth

        max_end = max(sp.start_offset_ms + sp.duration_ms for sp in self.spans) or 1.0
        bar_width = 24
        lines = [
            f"{'Span':<34} {'Status':<7} {'Dur':>8}  {'Timeline':<{bar_width}}  Details",
            "-" * 96,
        ]

        for sp in self.spans:
            depth = depth_by_id[sp.span_id]
            prefix = "  " * depth + ("|- " if depth > 0 else "")
            label = f"{prefix}{sp.name}"[:34]

            start_col = int((sp.start_offset_ms / max_end) * (bar_width - 1))
            width_cols = max(1, int((sp.duration_ms / max_end) * bar_width))
            end_col = min(bar_width, start_col + width_cols)
            bar = "." * start_col + "#" * (end_col - start_col) + "." * (bar_width - end_col)

            meta_parts: list[str] = []
            if sp.input_tokens or sp.output_tokens:
                meta_parts.append(f"tok={sp.input_tokens}in/{sp.output_tokens}out")
            if sp.cost_usd > 0:
                meta_parts.append(f"${sp.cost_usd:.5f}")
            for k, v in sp.attributes.items():
                meta_parts.append(f"{k}={v}")

            lines.append(
                f"{label:<34} {sp.status:<7} {sp.duration_ms:>6.2f}ms  [{bar}]  {', '.join(meta_parts)}"
            )

        lines.append("-" * 96)
        lines.append(
            f"Totals: spans={len(self.spans)} | "
            f"tokens={self.total_input_tokens}in/{self.total_output_tokens}out | "
            f"est_cost=${self.total_cost_usd:.5f}"
        )
        return "\n".join(lines)
