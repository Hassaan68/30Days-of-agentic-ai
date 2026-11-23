"""Day 15: Minimal MCP stdio Client (Pure Python Stdlib).

Spawns `mcp_stdio_demo_server.py` over stdio and runs the full MCP lifecycle:
1. `initialize` handshake + `notifications/initialized` notification.
2. Dynamic tool discovery via `tools/list`.
3. Converting MCP tool definitions (`inputSchema`) into OpenAI-compatible
   agent tool schemas (`{"type": "function", "function": ...}`).
4. Invoking tools via `tools/call` (including inspecting `isError: true` responses).
5. Listing and reading MCP Resources and Prompts.

Runs end-to-end with zero external packages installed.

Usage:
    python3 15_Day_Model_Context_Protocol_Basics/mcp_client_basics.py
    python3 15_Day_Model_Context_Protocol_Basics/mcp_client_basics.py --verbose
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class StdioMCPClient:
    """Minimal JSON-RPC 2.0 MCP client communicating with a local server over stdio."""

    def __init__(self, command: list[str], verbose: bool = False) -> None:
        self.command = command
        self.verbose = verbose
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 1

    def __enter__(self) -> StdioMCPClient:
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        finally:
            if self.verbose and self._proc.stderr:
                stderr_output = self._proc.stderr.read().strip()
                if stderr_output:
                    print("\n--- SERVER STDERR LOGS ---")
                    print(stderr_output)
            self._proc = None

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request and wait for its matching response."""
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        req_id = self._next_id
        self._next_id += 1

        frame: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            frame["params"] = params

        raw_out = json.dumps(frame)
        if self.verbose:
            print(f"\n[wire -> server] {raw_out}")

        self._proc.stdin.write(raw_out + "\n")
        self._proc.stdin.flush()

        raw_in = self._proc.stdout.readline()
        if not raw_in:
            stderr_hint = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"MCP server closed stdout unexpectedly. stderr={stderr_hint!r}")

        if self.verbose:
            print(f"[wire <- server] {raw_in.strip()}")

        response = json.loads(raw_in)
        if response.get("id") != req_id:
            raise RuntimeError(f"Mismatched JSON-RPC id: expected {req_id}, got {response.get('id')}")
        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"JSON-RPC error {err.get('code')}: {err.get('message')}")
        return dict(response["result"])

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC 2.0 notification (no 'id', expects no response)."""
        assert self._proc is not None and self._proc.stdin
        frame: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            frame["params"] = params

        raw_out = json.dumps(frame)
        if self.verbose:
            print(f"\n[wire -> server] {raw_out}  (notification)")

        self._proc.stdin.write(raw_out + "\n")
        self._proc.stdin.flush()

    # --- High-level MCP operations ---

    def initialize(self) -> dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "day15-demo-client", "version": "1.0.0"},
            },
        )
        # Complete the handshake with the mandatory initialized notification
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.request("tools/list").get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})

    def list_resources(self) -> list[dict[str, Any]]:
        return list(self.request("resources/list").get("resources", []))

    def read_resource(self, uri: str) -> list[dict[str, Any]]:
        return list(self.request("resources/read", {"uri": uri}).get("contents", []))

    def list_prompts(self) -> list[dict[str, Any]]:
        return list(self.request("prompts/list").get("prompts", []))


def mcp_tools_to_openai_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tools/list entries into OpenAI-compatible function-calling schemas."""
    openai_tools: list[dict[str, Any]] = []
    for tool in mcp_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
        )
    return openai_tools


def extract_text_content(call_result: dict[str, Any]) -> str:
    parts = [
        block.get("text", "")
        for block in call_result.get("content", [])
        if block.get("type") == "text"
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 15: Minimal MCP stdio Client Demo")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every raw JSON-RPC 2.0 wire frame and server stderr logs.",
    )
    args = parser.parse_args()

    server_script = Path(__file__).resolve().parent / "mcp_stdio_demo_server.py"
    if not server_script.exists():
        raise FileNotFoundError(f"Expected MCP demo server at {server_script}")

    print(f"Spawning MCP server over stdio: {server_script.name}")
    print("=" * 72)

    with StdioMCPClient([sys.executable, str(server_script)], verbose=args.verbose) as client:
        # 1. MCP Handshake
        init_result = client.initialize()
        server_info = init_result["serverInfo"]
        print(
            f"1. Handshake complete -> {server_info['name']} v{server_info['version']} "
            f"(protocol {init_result['protocolVersion']})"
        )
        print(f"   Capabilities: {', '.join(init_result['capabilities'].keys())}")

        # 2. Discover Tools dynamically
        mcp_tools = client.list_tools()
        print(f"\n2. Discovered {len(mcp_tools)} MCP tool(s) via tools/list:")
        for tool in mcp_tools:
            req = tool.get("inputSchema", {}).get("required", [])
            print(f"   - {tool['name']} (required args: {req}): {tool['description']}")

        # 3. Convert MCP tool definitions to OpenAI-compatible agent schemas
        agent_tools = mcp_tools_to_openai_tools(mcp_tools)
        print("\n3. Converted first MCP tool into OpenAI-compatible agent schema:")
        print(json.dumps(agent_tools[0], indent=2))

        # 4. Invoke tools via tools/call
        print("\n4. Invoking discovered tools via tools/call:")

        incidents_res = client.call_tool("list_active_incidents", {})
        print(f"\n   [tools/call] list_active_incidents -> isError={incidents_res.get('isError')}")
        print(extract_text_content(incidents_res))

        status_res = client.call_tool("get_service_status", {"service_name": "payments-api"})
        print(f"\n   [tools/call] get_service_status(payments-api) -> isError={status_res.get('isError')}")
        print(extract_text_content(status_res))

        budget_res = client.call_tool("calculate_error_budget", {"sla_pct": 99.95, "window_days": 30})
        print(f"\n   [tools/call] calculate_error_budget(99.95%, 30d) -> isError={budget_res.get('isError')}")
        print(extract_text_content(budget_res))

        # Demonstrate tool-level error (isError=True) vs JSON-RPC protocol error
        missing_res = client.call_tool("get_service_status", {"service_name": "unknown-svc"})
        print(f"\n   [tools/call] get_service_status(unknown-svc) -> isError={missing_res.get('isError')}")
        print(f"   Error text returned to agent: {extract_text_content(missing_res)}")

        # 5. Inspect Resources and Prompts primitives
        resources = client.list_resources()
        print(f"\n5. Discovered {len(resources)} MCP resource(s) via resources/list:")
        for res in resources:
            print(f"   - {res['uri']} ({res['mimeType']}): {res['name']}")
        catalog_contents = client.read_resource(resources[0]["uri"])
        print("   Read resource preview:")
        print(catalog_contents[0]["text"])

        prompts = client.list_prompts()
        print(f"\n6. Discovered {len(prompts)} MCP prompt(s) via prompts/list:")
        for prm in prompts:
            print(f"   - {prm['name']}: {prm['description']}")

    print("\n" + "=" * 72)
    print("MCP stdio session closed cleanly.")


if __name__ == "__main__":
    main()
