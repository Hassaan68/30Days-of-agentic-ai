"""Day 29: Zero-Dependency Python Stdlib SSE Agent Server & Client Demo.

Starts a real HTTP/1.1 Server-Sent Events (SSE) server on an ephemeral local
port (`127.0.0.1:0`), connects with a streaming client, parses every
`text/event-stream` frame (`id:`, `event:`, `data:`, and `: keep-alive`
comments), demonstrates clean mid-stream client cancellation, and shuts down.

If executed inside a restricted sandbox that blocks TCP loopback connections,
the demo automatically runs the exact same HTTP/1.1 handler and SSE parser over
an in-memory OS `socket.socketpair()` so learners can verify the wire protocol
anywhere with zero installs.

Run from the repository root:
    python3 29_Day_Deploying_Agents_as_APIs_and_Streaming_UX/stdlib_sse_demo.py
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


# ---------------------------------------------------------------------------
# SSE Formatting & Parsing Helpers
# ---------------------------------------------------------------------------


def encode_sse_event(event: str, data: dict[str, Any], event_id: int) -> bytes:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    frame = f"id: {event_id}\nevent: {event}\ndata: {payload}\n\n"
    return frame.encode("utf-8")


def encode_sse_heartbeat() -> bytes:
    ts = datetime.now(timezone.utc).isoformat()
    return f": keep-alive {ts}\n\n".encode("utf-8")


@dataclass
class SSEFrame:
    event_id: str | None
    event: str
    data: dict[str, Any]


# ---------------------------------------------------------------------------
# Stdlib Threading HTTP Server with Real Chunk-Flushed SSE Streaming
# ---------------------------------------------------------------------------


class AgentSSEHandler(BaseHTTPRequestHandler):
    """Handles `/health` and `/v1/agent/stream` over HTTP/1.1."""

    protocol_version = "HTTP/1.1"
    server_disconnected_runs: list[str] = []

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default stderr access logs so demo output stays readable.
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "stdlib-sse-agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/v1/agent/stream":
            self._send_json(404, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid_json"})
            return

        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            self._send_json(422, {"error": "prompt is required"})
            return

        request_id = self.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:10]}"

        # Send SSE response headers immediately.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Request-ID", request_id)
        self.send_header("Connection", "close")
        self.end_headers()

        self._stream_agent_run(prompt, request_id)

    def _write_frame(self, raw: bytes) -> bool:
        """Write and flush one SSE frame; return False if client disconnected."""
        try:
            self.wfile.write(raw)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _stream_agent_run(self, prompt: str, request_id: str) -> None:
        started = time.perf_counter()
        seq = 0

        def emit(event: str, payload: dict[str, Any]) -> bool:
            nonlocal seq
            seq += 1
            frame = encode_sse_event(
                event, {"request_id": request_id, "seq": seq, **payload}, event_id=seq
            )
            return self._write_frame(frame)

        events: list[tuple[str, dict[str, Any], float]] = [
            (
                "plan",
                {
                    "steps": [
                        "Check checkout-api service health",
                        "Query recent 502 error logs",
                        "Prepare rollback pending operator approval",
                    ]
                },
                0.06,
            ),
            (
                "tool_start",
                {"tool": "get_service_health", "arguments": {"service": "checkout-api"}},
                0.08,
            ),
            (
                "tool_result",
                {"tool": "get_service_health", "status": "degraded", "error_rate_pct": 14.8},
                0.06,
            ),
            (
                "tool_start",
                {"tool": "query_error_logs", "arguments": {"service": "checkout-api"}},
                0.08,
            ),
            (
                "tool_result",
                {
                    "tool": "query_error_logs",
                    "top_error": "DB pool exhausted after v2.14.0 deploy",
                },
                0.06,
            ),
            (
                "approval_required",
                {
                    "approval_id": "apr_demo_901",
                    "tool": "rollback_deployment",
                    "arguments": {"service": "checkout-api", "target_version": "v2.13.9"},
                },
                0.06,
            ),
            ("token", {"delta": "checkout-api has a 14.8% 502 rate "}, 0.05),
            ("token", {"delta": "caused by DB pool exhaustion in v2.14.0. "}, 0.05),
            ("token", {"delta": "Rollback to v2.13.9 is queued for approval."}, 0.05),
        ]

        for idx, (ev_name, payload, delay_s) in enumerate(events):
            if idx == 2:
                if not self._write_frame(encode_sse_heartbeat()):
                    self.server_disconnected_runs.append(request_id)
                    return
            if not emit(ev_name, payload):
                self.server_disconnected_runs.append(request_id)
                return
            time.sleep(delay_s)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        emit("done", {"elapsed_ms": elapsed_ms, "prompt_chars": len(prompt)})


# ---------------------------------------------------------------------------
# Transport Helpers (TCP Loopback + Sandboxed Socketpair Fallback)
# ---------------------------------------------------------------------------


class _SocketPairHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection bound to an already-connected local socketpair endpoint."""

    def __init__(self, client_sock: socket.socket) -> None:
        super().__init__("localhost", timeout=10)
        self._preconnected = client_sock

    def connect(self) -> None:
        self.sock = self._preconnected


class _DummyServer:
    server_name = "localhost"
    server_port = 0


def _open_connection(host: str | None, port: int) -> http.client.HTTPConnection:
    if host is not None:
        return http.client.HTTPConnection(host, port, timeout=10)

    # Socketpair mode: serve one HTTP/1.1 request on a real kernel socketpair.
    client_sock, server_sock = socket.socketpair()

    def _serve_one() -> None:
        try:
            AgentSSEHandler(server_sock, ("127.0.0.1", 0), _DummyServer())
        except OSError:
            pass
        finally:
            try:
                server_sock.close()
            except OSError:
                pass

    threading.Thread(target=_serve_one, daemon=True).start()
    return _SocketPairHTTPConnection(client_sock)


# ---------------------------------------------------------------------------
# Streaming SSE Client
# ---------------------------------------------------------------------------


def consume_sse_stream(
    host: str | None,
    port: int,
    prompt: str,
    request_id: str,
    cancel_after_events: int | None = None,
) -> tuple[list[SSEFrame], int]:
    """Connect to `/v1/agent/stream` and parse incoming SSE frames incrementally."""
    conn = _open_connection(host, port)
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    conn.request(
        "POST",
        "/v1/agent/stream",
        body=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Request-ID": request_id,
        },
    )
    resp = conn.getresponse()
    assert resp.status == 200, f"Unexpected HTTP status: {resp.status}"
    assert "text/event-stream" in resp.getheader("Content-Type", "")
    assert resp.getheader("X-Request-ID") == request_id

    frames: list[SSEFrame] = []
    heartbeats = 0
    cur_id: str | None = None
    cur_event = "message"
    cur_data_lines: list[str] = []

    try:
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8").rstrip("\r\n")

            if line.startswith(":"):
                heartbeats += 1
                print(f"  [heartbeat] {line}")
                continue

            if line == "":
                if cur_data_lines:
                    payload = json.loads("\n".join(cur_data_lines))
                    frame = SSEFrame(cur_id, cur_event, payload)
                    frames.append(frame)
                    _print_frame(frame)
                    if (
                        cancel_after_events is not None
                        and len(frames) >= cancel_after_events
                    ):
                        print(
                            f"  [client] Cancelling stream early after {len(frames)} events..."
                        )
                        break
                cur_id = None
                cur_event = "message"
                cur_data_lines = []
                continue

            if line.startswith("id:"):
                cur_id = line[3:].strip()
            elif line.startswith("event:"):
                cur_event = line[6:].strip()
            elif line.startswith("data:"):
                cur_data_lines.append(line[5:].strip())
    finally:
        conn.close()

    return frames, heartbeats


def _print_frame(frame: SSEFrame) -> None:
    if frame.event == "token":
        print(f"  [id={frame.event_id} event=token] delta={frame.data['delta']!r}")
    else:
        compact = {k: v for k, v in frame.data.items() if k not in ("request_id", "seq")}
        print(f"  [id={frame.event_id} event={frame.event}] {compact}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 29 Stdlib SSE Demo")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the self-contained local server + SSE client demo (default).",
    )
    parser.parse_args()

    server: ThreadingHTTPServer | None = None
    host: str | None = None
    port = 0

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), AgentSSEHandler)
        host, port = server.server_address
        threading.Thread(target=server.serve_forever, daemon=True).start()

        # Verify TCP loopback connectivity (may be blocked in air-gapped sandboxes).
        probe = http.client.HTTPConnection(host, port, timeout=5)
        probe.request("GET", "/health")
        health_resp = probe.getresponse()
        health_data = json.loads(health_resp.read().decode("utf-8"))
        probe.close()
        print(f"Started stdlib SSE agent server on http://{host}:{port}")
    except PermissionError:
        if server is not None:
            server.shutdown()
            server.server_close()
            server = None
        host, port = None, 0
        print(
            "Local TCP loopback is blocked by sandbox; running identical HTTP/1.1 SSE "
            "handler over local socketpair."
        )
        conn = _open_connection(None, 0)
        conn.request("GET", "/health")
        health_resp = conn.getresponse()
        health_data = json.loads(health_resp.read().decode("utf-8"))
        conn.close()

    try:
        print(f"\n=== 1. Health Check ===\n  HTTP {health_resp.status}: {health_data}")

        # 2. Full end-to-end SSE stream.
        print("\n=== 2. Full Agent SSE Stream (POST /v1/agent/stream) ===")
        frames, heartbeats = consume_sse_stream(
            host,
            port,
            prompt="Investigate checkout-api 502 errors and stage a rollback",
            request_id="req-stream-full-01",
        )
        token_text = "".join(f.data["delta"] for f in frames if f.event == "token")
        print(f"\n  Reassembled streamed text: {token_text}")
        print(f"  Received {len(frames)} typed events + {heartbeats} keep-alive heartbeat(s).")
        assert frames[-1].event == "done"

        # 3. Client disconnect / mid-stream cancellation test.
        print("\n=== 3. Mid-Stream Client Disconnect & Server Cleanup ===")
        partial_frames, _ = consume_sse_stream(
            host,
            port,
            prompt="Investigate checkout-api 502 errors",
            request_id="req-stream-cancel-02",
            cancel_after_events=3,
        )
        time.sleep(0.25)
        print(
            f"  Client stopped after {len(partial_frames)} events; "
            f"server cleanly recorded disconnect for: {AgentSSEHandler.server_disconnected_runs}"
        )
        assert "req-stream-cancel-02" in AgentSSEHandler.server_disconnected_runs

    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        print("\nServer shut down cleanly.")


if __name__ == "__main__":
    main()
