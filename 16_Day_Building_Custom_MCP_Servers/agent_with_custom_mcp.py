"""Agent client wired to the custom `notes_mcp_server.py` over stdio.

This script spawns `notes_mcp_server.py` as a child process, performs the MCP
handshake, discovers tools and resources dynamically, injects the `notes://index`
resource into the prompt, and runs an agent loop against the live MCP server.

Usage:
  # Run with deterministic offline mock model (still uses the real MCP server!)
  python3 16_Day_Building_Custom_MCP_Servers/agent_with_custom_mcp.py --demo

  # Run against an OpenAI-compatible API (OpenAI, Ollama, OpenRouter, etc.)
  OPENAI_API_KEY=... python3 16_Day_Building_Custom_MCP_Servers/agent_with_custom_mcp.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any
import urllib.error
import urllib.request

DEFAULT_TASK = (
    "Find our database failover runbook, read its full details, create a companion "
    "note for Redis cache failover referencing the 30s DNS TTL, and verify whether "
    "obsolete draft 'note-999' needs cleanup."
)


class MCPError(RuntimeError):
    """Raised when the MCP server returns a JSON-RPC protocol error."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"MCP error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPStdioClient:
    """Minimal JSON-RPC 2.0 MCP client communicating with a child process over stdio."""

    def __init__(self, command: list[str]) -> None:
        self._command = command
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 0
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def __enter__(self) -> MCPStdioClient:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Drain stderr on a background thread so server logs never block pipes.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_lines.append(line.rstrip("\n"))

    @property
    def stderr_logs(self) -> list[str]:
        return list(self._stderr_lines)

    def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _read_response(self, expected_id: int) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            raw = self._proc.stdout.readline()
            if not raw:
                logs = "\n".join(self._stderr_lines[-10:])
                raise RuntimeError(
                    f"MCP server closed stdout unexpectedly.\nRecent stderr:\n{logs}"
                )
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Corrupted stdout frame from MCP server (check for stray print()): {raw!r}"
                ) from exc

            if msg.get("id") == expected_id:
                if "error" in msg:
                    err = msg["error"]
                    raise MCPError(err["code"], err["message"], err.get("data"))
                return msg["result"]

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        req_id = self._next_id
        self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            }
        )
        return self._read_response(req_id)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def initialize(self) -> dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "day16-agent-client", "version": "1.0.0"},
            },
        )
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def list_resources(self) -> list[dict[str, Any]]:
        return self.request("resources/list").get("resources", [])

    def read_resource(self, uri: str) -> str:
        result = self.request("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        return "\n".join(part.get("text", "") for part in contents)

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        if self._stderr_thread:
            self._stderr_thread.join(timeout=1)


def mcp_tools_to_openai_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate MCP tool definitions into OpenAI-compatible function schemas."""
    openai_tools: list[dict[str, Any]] = []
    for tool in mcp_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "inputSchema", {"type": "object", "properties": {}}
                    ),
                },
            }
        )
    # Also give the model a read-only bridge tool for MCP resources/read.
    openai_tools.append(
        {
            "type": "function",
            "function": {
                "name": "read_mcp_resource",
                "description": (
                    "Read a read-only MCP resource by URI (such as 'notes://index' "
                    "or 'notes://note-1')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "Resource URI starting with notes://",
                        }
                    },
                    "required": ["uri"],
                },
            },
        }
    )
    return openai_tools


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[LLMToolCall]


class DemoMockLLM:
    """Deterministic model driver that exercises the real MCP server end to end."""

    def __init__(self) -> None:
        self._turn = 0

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        del tools
        self._turn += 1

        if self._turn == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id="call_1",
                        name="search_notes",
                        arguments={"query": "failover"},
                    )
                ],
            )
        if self._turn == 2:
            return LLMResponse(
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id="call_2",
                        name="read_mcp_resource",
                        arguments={"uri": "notes://note-1"},
                    )
                ],
            )
        if self._turn == 3:
            return LLMResponse(
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id="call_3",
                        name="create_note",
                        arguments={
                            "title": "Redis Cache Failover Runbook",
                            "content": (
                                "Confirm sentinel quorum before promoting replica. "
                                "Flush stale connection pools and switch DNS with 30s TTL "
                                "to match the Database Failover Runbook."
                            ),
                            "tags": ["redis", "ops", "runbook"],
                        },
                    )
                ],
            )
        if self._turn == 4:
            # Trigger a structured isError=True response from the MCP server to
            # demonstrate graceful recovery without crashing the agent loop.
            return LLMResponse(
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id="call_4",
                        name="delete_note",
                        arguments={"note_id": "note-999"},
                    )
                ],
            )

        # Inspect latest observations to build an accurate final answer.
        created_uri = "notes://note-3"
        for msg in reversed(messages):
            if msg.get("role") == "tool" and "Redis Cache Failover Runbook" in msg.get(
                "content", ""
            ):
                try:
                    data = json.loads(msg["content"])
                    created_uri = data.get("note", {}).get("uri", created_uri)
                except json.JSONDecodeError:
                    pass
                break

        return LLMResponse(
            content=(
                "Completed all requested note updates via the custom MCP server:\n"
                "1. Located and read `notes://note-1` (Database Failover Runbook), confirming "
                "the 30s DNS TTL requirement.\n"
                f"2. Created companion note `Redis Cache Failover Runbook` at `{created_uri}`.\n"
                "3. Checked `note-999` for deletion; the server returned a structured `not_found` "
                "error confirming no obsolete `note-999` exists."
            ),
            tool_calls=[],
        )


class OpenAICompatibleLLM:
    """Minimal stdlib HTTP client for OpenAI-compatible /chat/completions endpoints."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.1,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc

        message = data["choices"][0]["message"]
        parsed_calls: list[LLMToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            raw_args = raw_call["function"].get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            parsed_calls.append(
                LLMToolCall(
                    id=raw_call["id"],
                    name=raw_call["function"]["name"],
                    arguments=args,
                )
            )
        return LLMResponse(content=message.get("content"), tool_calls=parsed_calls)


def should_use_demo(force_demo: bool) -> bool:
    if force_demo or os.getenv("AGENT_DEMO_MODE") == "1":
        return True
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
        return True
    return False


def execute_tool_call(client: MCPStdioClient, call: LLMToolCall) -> str:
    """Route a model tool call either to MCP resources/read or MCP tools/call."""
    if call.name == "read_mcp_resource":
        uri = str(call.arguments.get("uri", ""))
        try:
            return client.read_resource(uri)
        except MCPError as exc:
            return json.dumps({"error": exc.message, "code": exc.code, "uri": uri})

    try:
        result = client.call_tool(call.name, call.arguments)
    except MCPError as exc:
        return json.dumps({"error": exc.message, "code": exc.code})

    text_blocks = [
        block.get("text", "")
        for block in result.get("content", [])
        if block.get("type") == "text"
    ]
    payload = "\n".join(text_blocks)
    if result.get("isError"):
        return f"[MCP TOOL ERROR isError=True]\n{payload}"
    return payload


def run_agent(task: str, force_demo: bool = False, max_steps: int = 8) -> None:
    server_path = Path(__file__).with_name("notes_mcp_server.py")
    use_demo = should_use_demo(force_demo)

    if use_demo:
        llm: DemoMockLLM | OpenAICompatibleLLM = DemoMockLLM()
        mode_label = "DEMO (mock LLM + real custom MCP server)"
    else:
        llm = OpenAICompatibleLLM(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        )
        mode_label = f"LIVE ({llm.model} @ {llm.base_url})"

    print(f"=== Day 16: Agent + Custom MCP Server [{mode_label}] ===\n")

    with MCPStdioClient([sys.executable, str(server_path)]) as mcp:
        init_info = mcp.initialize()
        server_info = init_info["serverInfo"]
        print(
            f"Connected to MCP server: {server_info['name']} v{server_info['version']} "
            f"(protocol {init_info['protocolVersion']})"
        )

        mcp_tools = mcp.list_tools()
        print("Discovered MCP tools:", [t["name"] for t in mcp_tools])

        initial_resources = mcp.list_resources()
        print("Discovered MCP resources:", [r["uri"] for r in initial_resources])

        # Load read-only catalog resource into initial system context.
        notes_index_json = mcp.read_resource("notes://index")
        print("\n--- Initial Resource Snapshot (notes://index) ---")
        print(notes_index_json)

        openai_tools = mcp_tools_to_openai_tools(mcp_tools)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an engineering operations assistant connected to a custom "
                    "MCP notes server. Use tools to search, create, or delete notes, and "
                    "use `read_mcp_resource` to inspect full note contents by URI. "
                    "If a tool returns an error, inspect it and handle it gracefully.\n\n"
                    f"Current notes://index snapshot:\n{notes_index_json}"
                ),
            },
            {"role": "user", "content": task},
        ]

        print(f"\n--- Task ---\n{task}\n")

        for step in range(1, max_steps + 1):
            response = llm.chat(messages, openai_tools)

            if not response.tool_calls:
                print(f"--- Final Agent Response (step {step}) ---")
                print(response.content or "(empty response)")
                break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                print(f"[Step {step}] Tool call -> {tc.name}({json.dumps(tc.arguments)})")
                observation = execute_tool_call(mcp, tc)
                preview = observation if len(observation) <= 320 else observation[:320] + "..."
                print(f"[Step {step}] Observation <- {preview}\n")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation,
                    }
                )
        else:
            print("Stopped after reaching max_steps without a final response.")

        print("\n--- Updated MCP Resources After Run ---")
        updated_resources = mcp.list_resources()
        for res in updated_resources:
            print(f"  - {res['uri']}: {res['name']}")

        if mcp.stderr_logs:
            print("\n--- MCP Server stderr Logs (kept separate from stdout!) ---")
            for line in mcp.stderr_logs:
                print(f"  {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an agent connected to the custom notes MCP server."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use deterministic local mock LLM instead of a live API.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Task instruction for the agent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_agent(task=args.task, force_demo=args.demo)


if __name__ == "__main__":
    main()
