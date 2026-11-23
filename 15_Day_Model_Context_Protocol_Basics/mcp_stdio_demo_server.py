"""Day 15: Minimal JSON-RPC 2.0 MCP Server over stdio (Pure Python Stdlib).

Implements the Model Context Protocol (MCP) wire protocol directly over
newline-delimited JSON-RPC 2.0 on stdin/stdout so you can see every handshake
and tool-invocation frame without framework magic.

Important stdio transport rule:
- `stdout` is strictly reserved for JSON-RPC 2.0 frames (one JSON object per line).
- Any server logs MUST go to `stderr` so they never corrupt the protocol stream.

Run directly (normally spawned by `mcp_client_basics.py`):
    python3 15_Day_Model_Context_Protocol_Basics/mcp_stdio_demo_server.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {
    "name": "ops-demo-mcp-server",
    "version": "1.0.0",
}

# ---------------------------------------------------------------------------
# Sample operational data exposed through MCP Tools & Resources
# ---------------------------------------------------------------------------

SERVICES: dict[str, dict[str, Any]] = {
    "payments-api": {
        "service": "payments-api",
        "status": "degraded",
        "owner": "team-payments",
        "p95_latency_ms": 420,
        "error_rate_pct": 1.8,
        "region": "us-east-1",
    },
    "checkout-web": {
        "service": "checkout-web",
        "status": "healthy",
        "owner": "team-storefront",
        "p95_latency_ms": 95,
        "error_rate_pct": 0.03,
        "region": "us-east-1",
    },
    "webhook-dispatcher": {
        "service": "webhook-dispatcher",
        "status": "healthy",
        "owner": "team-platform",
        "p95_latency_ms": 140,
        "error_rate_pct": 0.12,
        "region": "eu-west-1",
    },
}

ACTIVE_INCIDENTS: list[dict[str, Any]] = [
    {
        "incident_id": "INC-2041",
        "severity": "SEV-2",
        "service": "payments-api",
        "summary": "Elevated 502 responses from upstream card processor in us-east-1.",
        "started_utc": "2026-05-14T11:20:00Z",
    },
]

# ---------------------------------------------------------------------------
# MCP Primitive Definitions: Tools, Resources, Prompts
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_service_status",
        "description": "Fetch current health status, latency, and error rate for a named service.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "Service identifier (e.g., payments-api, checkout-web, webhook-dispatcher).",
                }
            },
            "required": ["service_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_active_incidents",
        "description": "List all currently open operational incidents, optionally filtered by severity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["SEV-1", "SEV-2", "SEV-3"],
                    "description": "Optional severity filter.",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate_error_budget",
        "description": "Compute allowed downtime minutes for a given availability SLA percentage over a window in days.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sla_pct": {
                    "type": "number",
                    "description": "Availability target as a percentage (e.g., 99.95).",
                },
                "window_days": {
                    "type": "integer",
                    "description": "Measurement window in days (default: 30).",
                },
            },
            "required": ["sla_pct"],
            "additionalProperties": False,
        },
    },
]

RESOURCES: list[dict[str, Any]] = [
    {
        "uri": "ops://catalog/services",
        "name": "Service Ownership Catalog",
        "description": "Read-only JSON list of services, owners, and primary regions.",
        "mimeType": "application/json",
    }
]

PROMPTS: list[dict[str, Any]] = [
    {
        "name": "incident_triage",
        "description": "Structured checklist prompt for triaging a degraded service.",
        "arguments": [
            {
                "name": "service_name",
                "description": "Name of the affected service.",
                "required": True,
            }
        ],
    }
]


# ---------------------------------------------------------------------------
# Tool execution handlers
# ---------------------------------------------------------------------------


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool and return an MCP CallToolResult payload."""
    if name == "get_service_status":
        service_name = str(arguments.get("service_name", "")).strip()
        if not service_name:
            return _tool_error("Missing required argument 'service_name'.")
        record = SERVICES.get(service_name)
        if record is None:
            known = ", ".join(sorted(SERVICES.keys()))
            return _tool_error(f"Unknown service '{service_name}'. Known services: {known}.")
        return _tool_ok(record)

    if name == "list_active_incidents":
        severity = arguments.get("severity")
        if severity is not None:
            filtered = [inc for inc in ACTIVE_INCIDENTS if inc["severity"] == severity]
            return _tool_ok({"count": len(filtered), "incidents": filtered})
        return _tool_ok({"count": len(ACTIVE_INCIDENTS), "incidents": ACTIVE_INCIDENTS})

    if name == "calculate_error_budget":
        if "sla_pct" not in arguments:
            return _tool_error("Missing required argument 'sla_pct'.")
        try:
            sla_pct = float(arguments["sla_pct"])
            window_days = int(arguments.get("window_days", 30))
        except (TypeError, ValueError) as exc:
            return _tool_error(f"Invalid numeric argument: {exc}")

        if not (0.0 < sla_pct < 100.0):
            return _tool_error("sla_pct must be strictly between 0 and 100 (e.g., 99.95).")
        if window_days <= 0:
            return _tool_error("window_days must be a positive integer.")

        total_minutes = window_days * 24 * 60
        allowed_downtime_minutes = round(total_minutes * (1.0 - sla_pct / 100.0), 2)
        return _tool_ok(
            {
                "sla_pct": sla_pct,
                "window_days": window_days,
                "total_window_minutes": total_minutes,
                "allowed_downtime_minutes": allowed_downtime_minutes,
            }
        )

    return _tool_error(f"Unknown tool '{name}'.")


def _tool_ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": False,
    }


def _tool_error(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 dispatcher over stdin / stdout
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Always log diagnostics to stderr, never stdout."""
    print(f"[mcp-server] {msg}", file=sys.stderr, flush=True)


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _jsonrpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def handle_message(msg: dict[str, Any], state: dict[str, bool]) -> dict[str, Any] | None:
    """Dispatch one JSON-RPC 2.0 request or notification."""
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}
    is_notification = "id" not in msg

    if method == "initialize":
        state["initialized_requested"] = True
        _log("Handled initialize handshake.")
        return _jsonrpc_result(
            req_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": SERVER_INFO,
            },
        )

    if method == "notifications/initialized":
        state["ready"] = True
        _log("Client sent notifications/initialized. Session ready.")
        return None  # Notifications never receive a response

    if method == "ping":
        return _jsonrpc_result(req_id, {})

    if not state.get("initialized_requested"):
        if is_notification:
            return None
        return _jsonrpc_error(req_id, -32002, "Server not initialized. Send 'initialize' first.")

    if method == "tools/list":
        return _jsonrpc_result(req_id, {"tools": TOOL_DEFINITIONS})

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(tool_name, str) or not isinstance(arguments, dict):
            return _jsonrpc_error(req_id, -32602, "Invalid params for tools/call: expected 'name' and 'arguments'.")
        return _jsonrpc_result(req_id, call_tool(tool_name, arguments))

    if method == "resources/list":
        return _jsonrpc_result(req_id, {"resources": RESOURCES})

    if method == "resources/read":
        uri = params.get("uri")
        if uri == "ops://catalog/services":
            catalog = [
                {"service": s["service"], "owner": s["owner"], "region": s["region"]}
                for s in SERVICES.values()
            ]
            return _jsonrpc_result(
                req_id,
                {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(catalog, indent=2),
                        }
                    ]
                },
            )
        return _jsonrpc_error(req_id, -32602, f"Unknown resource URI: {uri}")

    if method == "prompts/list":
        return _jsonrpc_result(req_id, {"prompts": PROMPTS})

    if method == "prompts/get":
        prompt_name = params.get("name")
        prompt_args = params.get("arguments") or {}
        if prompt_name == "incident_triage":
            svc = str(prompt_args.get("service_name", "<service>"))
            return _jsonrpc_result(
                req_id,
                {
                    "description": f"Incident triage checklist for {svc}",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": (
                                    f"Triage service '{svc}': check current health status, list active incidents, "
                                    "verify remaining monthly error budget, and propose immediate mitigation steps."
                                ),
                            },
                        }
                    ],
                },
            )
        return _jsonrpc_error(req_id, -32602, f"Unknown prompt name: {prompt_name}")

    if is_notification:
        return None
    return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


def serve_stdio() -> None:
    """Read newline-delimited JSON-RPC 2.0 messages until stdin closes."""
    state = {"initialized_requested": False, "ready": False}
    _log("Starting stdio MCP server.")

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _send(_jsonrpc_error(None, -32700, f"Parse error: {exc}"))
            continue

        response = handle_message(msg, state)
        if response is not None:
            _send(response)

    _log("stdin closed; shutting down cleanly.")


if __name__ == "__main__":
    serve_stdio()
