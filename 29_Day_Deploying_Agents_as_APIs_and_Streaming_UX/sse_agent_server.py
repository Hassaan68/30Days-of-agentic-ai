"""Day 29: FastAPI + Server-Sent Events (SSE) Streaming Agent API.

Exposes an agent over HTTP using `text/event-stream` so web, mobile, and CLI
clients receive live step updates (`plan`, `tool_start`, `tool_result`,
`token`, `approval_required`, `done`, `error`) instead of staring at a frozen
45-second POST request.

Run locally after installing dependencies (`pip install fastapi uvicorn`):
    uvicorn 29_Day_Deploying_Agents_as_APIs_and_Streaming_UX.sse_agent_server:app --reload --port 8000

Or from inside the Day 29 directory:
    uvicorn sse_agent_server:app --reload --port 8000

Test with curl:
    curl -N -X POST http://127.0.0.1:8000/v1/agent/stream \\
      -H "Content-Type: application/json" \\
      -H "X-Request-ID: req-demo-001" \\
      -d '{"prompt": "Check checkout-api errors and prepare a rollback", "demo": true}'
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Agent SSE Streaming API",
    version="1.0.0",
    description="Day 29 reference server streaming typed agent lifecycle events over SSE.",
)


# ---------------------------------------------------------------------------
# Request & SSE Framing Helpers
# ---------------------------------------------------------------------------


class AgentStreamRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None)
    demo: bool = Field(
        default=True,
        description="Run deterministic local streaming demo when True.",
    )


def format_sse(event: str, data: dict[str, Any], event_id: int) -> str:
    """Serialize one Server-Sent Event frame according to the WHATWG SSE spec."""
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"id: {event_id}\nevent: {event}\ndata: {payload}\n\n"


def format_heartbeat() -> str:
    """SSE comment line that keeps reverse proxies and load balancers open."""
    return f": keep-alive {datetime.now(timezone.utc).isoformat()}\n\n"


# ---------------------------------------------------------------------------
# Streaming Agent Generator (with Disconnect & Cancellation Handling)
# ---------------------------------------------------------------------------


async def stream_agent_events(
    request: Request,
    body: AgentStreamRequest,
    request_id: str,
) -> AsyncIterator[str]:
    """Yield typed SSE frames while watching for client disconnects."""
    seq = 0
    started = time.perf_counter()

    def next_event(event: str, data: dict[str, Any]) -> str:
        nonlocal seq
        seq += 1
        enriched = {"request_id": request_id, "seq": seq, **data}
        return format_sse(event, enriched, event_id=seq)

    try:
        # 1. Explicit plan event so UI renders a step tracker immediately.
        steps = [
            "Inspect checkout-api service health",
            "Query recent 502 error logs",
            "Draft remediation plan (gate production rollback on approval)",
        ]
        yield next_event("plan", {"steps": steps, "session_id": body.session_id})

        if await request.is_disconnected():
            return

        # 2. First tool execution with a keep-alive heartbeat during slow work.
        yield next_event(
            "tool_start",
            {"tool": "get_service_health", "arguments": {"service": "checkout-api"}},
        )
        await asyncio.sleep(0.25)
        yield format_heartbeat()
        if await request.is_disconnected():
            return

        yield next_event(
            "tool_result",
            {
                "tool": "get_service_health",
                "status": "degraded",
                "error_rate_pct": 14.8,
                "p95_latency_ms": 1820,
            },
        )

        # 3. Second tool execution.
        yield next_event(
            "tool_start",
            {"tool": "query_error_logs", "arguments": {"service": "checkout-api", "limit": 5}},
        )
        await asyncio.sleep(0.25)
        if await request.is_disconnected():
            return

        yield next_event(
            "tool_result",
            {
                "tool": "query_error_logs",
                "matched_lines": 42,
                "top_error": "UpstreamTimeout after deploy v2.14.0 (connection pool exhausted)",
            },
        )

        # 4. If prompt mentions rollback or restart, pause and emit HITL approval event.
        if any(word in body.prompt.lower() for word in ("rollback", "restart", "deploy")):
            yield next_event(
                "approval_required",
                {
                    "approval_id": f"apr_{uuid.uuid4().hex[:10]}",
                    "tool": "rollback_deployment",
                    "arguments": {"service": "checkout-api", "target_version": "v2.13.9"},
                    "reason": "Production mutation requires human confirmation.",
                },
            )
            await asyncio.sleep(0.15)
            if await request.is_disconnected():
                return

        # 5. Stream assistant tokens progressively for low perceived latency.
        answer_chunks = [
            "checkout-api is currently degraded ",
            "(14.8% 502 error rate) following deploy v2.14.0. ",
            "Error logs point to database connection pool exhaustion. ",
            "A rollback to v2.13.9 is staged and waiting for operator approval.",
        ]
        for chunk in answer_chunks:
            if await request.is_disconnected():
                return
            yield next_event("token", {"delta": chunk})
            await asyncio.sleep(0.08)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        yield next_event(
            "done",
            {
                "elapsed_ms": elapsed_ms,
                "model": os.getenv("MODEL_NAME", "demo-stream-model"),
                "demo": body.demo,
            },
        )

    except asyncio.CancelledError:
        # Client closed the socket or server is shutting down; clean up quietly.
        return
    except Exception as exc:
        yield next_event(
            "error",
            {"error_type": type(exc).__name__, "message": str(exc)},
        )


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "agent-sse-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/agent/stream")
async def agent_stream(
    body: AgentStreamRequest,
    request: Request,
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> StreamingResponse:
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"
    headers = {
        "X-Request-ID": request_id,
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Critical when running behind Nginx so SSE frames flush immediately.
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        stream_agent_events(request, body, request_id),
        media_type="text/event-stream",
        headers=headers,
    )
