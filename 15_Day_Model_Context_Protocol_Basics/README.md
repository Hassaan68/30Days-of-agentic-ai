[<< Day 14](../14_Day_Agentic_RAG/README.md) | [Home](../README.md) | [Day 16 >>](../16_Day_Building_Custom_MCP_Servers/README.md)

# Day 15: Model Context Protocol (MCP) Basics

Today we inspect the **Model Context Protocol (MCP)** from the wire up. Instead of writing custom glue code every time a new agent needs to talk to Postgres, GitHub, Slack, or an internal operations API, MCP standardizes how AI applications discover and invoke external tools, read-only resources, and prompt templates over JSON-RPC 2.0.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `mcp_stdio_demo_server.py` & `mcp_client_basics.py`](#walkthrough-mcp_stdio_demo_serverpy--mcp_client_basicspy)
- [Equivalent Official `mcp` Python SDK Client](#equivalent-official-mcp-python-sdk-client)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 15](#-exercises---day-15)

---

## Core Concept & Why It Matters

Before MCP, connecting AI applications to external systems suffered from the **M x N integration problem**:

- Suppose you maintain **M** AI hosts/agents (an IDE assistant, a CLI agent, a Slack support bot, an internal incident triage agent).
- You want them all to access **N** data sources and tools (GitHub, Jira, Postgres, Sentry, PagerDuty).
- Without a shared protocol, every host re-implements tool schemas, auth plumbing, error formatting, and transport glue for every integration (**M x N** custom connectors).

MCP turns **M x N** into **M + N**:

- Each data source or tool provider exposes **one MCP Server** speaking JSON-RPC 2.0.
- Each AI application embeds **one MCP Client** capable of speaking to *any* MCP Server.

When a new MCP server is added, every MCP-compatible host can discover its tools at runtime via `tools/list` without hardcoding tool definitions in the agent.

### MCP Architecture: Host, Client, Server

People often mix up "Host" and "Client." They have distinct jobs:

```text
┌─────────────────────────────────────────────────────────────┐
│  MCP HOST (e.g., your Python agent, IDE, desktop app)       │
│  Owns the LLM loop, permissions, and user UI                │
│                                                             │
│   ┌──────────────────┐         ┌──────────────────┐         │
│   │  MCP Client #1   │         │  MCP Client #2   │         │
│   │  (1:1 session)   │         │  (1:1 session)   │         │
│   └────────┬─────────┘         └────────┬─────────┘         │
└────────────┼────────────────────────────┼───────────────────┘
             │ stdio (local pipe)         │ Streamable HTTP
             ▼                            ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│  MCP Server A (local)    │  │  MCP Server B (remote)       │
│  ops-demo-mcp-server     │  │  company-jira-mcp            │
│  Tools / Resources /     │  │  Tools / Resources /         │
│  Prompts                 │  │  Prompts                     │
└──────────────────────────┘  └──────────────────────────────┘
```

- **Host:** The top-level AI application running the LLM conversation, deciding tool policies, and coordinating multiple server connections.
- **Client:** A stateful 1:1 protocol session managed inside the Host, talking to a single MCP Server.
- **Server:** A lightweight process (local or remote) that exposes capabilities via standard MCP primitives.

### Core Transports

1. **`stdio` (Standard Input / Standard Output):**
   - Best for local tools, developer CLIs, local filesystems, and desktop/IDE agents.
   - The Host spawns the server as a child process and exchanges **newline-delimited JSON-RPC 2.0** messages over `stdin` and `stdout`.
   - Zero network ports or TLS certificates needed—and the process lifecycle is tied directly to the client.
2. **`Streamable HTTP`:**
   - Best for shared, multi-tenant, or cloud-hosted remote MCP servers.
   - Uses standard HTTP `POST` requests to a single MCP endpoint, with optional Server-Sent Events (SSE) streams when the server needs to push progress updates or notifications (superseding the older dual-endpoint HTTP+SSE transport).

### The Three Core Primitives

| Primitive | Who Controls It? | Purpose | Key Methods |
| :--- | :--- | :--- | :--- |
| **Tools** | Model-controlled (with host approval) | Executable actions with JSON Schema inputs (`get_service_status`, `create_ticket`) | `tools/list`, `tools/call` |
| **Resources** | Application-controlled | Read-only structured context addressed by URI (`ops://catalog/services`, `file:///...`) | `resources/list`, `resources/read` |
| **Prompts** | User-controlled | Reusable, parameterized prompt templates (`incident_triage`) | `prompts/list`, `prompts/get` |

---

## How It Works Under the Hood

Every MCP session follows a strict **JSON-RPC 2.0** lifecycle:

```text
  MCP Client                                    MCP Server
      │                                              │
      │  1. Request (id=1): initialize               │
      │     {protocolVersion, capabilities,          │
      │      clientInfo}                             │
      ├─────────────────────────────────────────────►│
      │                                              │
      │  2. Response (id=1): InitializeResult        │
      │     {protocolVersion, capabilities,          │
      │      serverInfo}                             │
      │◄─────────────────────────────────────────────┤
      │                                              │
      │  3. Notification (no id):                    │
      │     notifications/initialized                │
      ├─────────────────────────────────────────────►│
      │                                              │
      │  4. Request (id=2): tools/list               │
      ├─────────────────────────────────────────────►│
      │  5. Response (id=2): {tools: [...]}          │
      │◄─────────────────────────────────────────────┤
      │                                              │
      │  6. Request (id=3): tools/call               │
      │     {name: "get_service_status",             │
      │      arguments: {"service_name": ...}}       │
      ├─────────────────────────────────────────────►│
      │  7. Response (id=3):                         │
      │     {content: [{type: "text", text: ...}],   │
      │      isError: false}                         │
      │◄─────────────────────────────────────────────┤
```

Notice two subtle wire-level details that trip up first-time implementers:

1. **Requests have an `"id"`, notifications do not.** When the client sends `notifications/initialized`, it omits `"id"`, and the server **must not** send a response back.
2. **Protocol errors vs. tool execution errors:**
   - If the client sends malformed JSON or calls an unknown JSON-RPC method, the server returns a JSON-RPC `"error"` object (`-32700`, `-32601`, `-32602`).
   - If the client calls a valid tool with a nonexistent service name (`"unknown-svc"`), the server returns a normal JSON-RPC `"result"` with `"isError": true` and a helpful text message inside `content`. That lets the LLM read the error, understand what went wrong, and recover cleanly on the next turn.

---

## Walkthrough: `mcp_stdio_demo_server.py` & `mcp_client_basics.py`

This folder contains two zero-dependency Python stdlib scripts so you can watch the exact MCP protocol work end-to-end before installing any SDK:

1. [`mcp_stdio_demo_server.py`](./mcp_stdio_demo_server.py): A minimal, spec-faithful JSON-RPC 2.0 MCP server over `stdio`.
   - Exposes 3 tools (`get_service_status`, `list_active_incidents`, `calculate_error_budget`), 1 resource (`ops://catalog/services`), and 1 prompt (`incident_triage`).
   - Sends every diagnostic log message to `sys.stderr`, leaving `sys.stdout` strictly for JSON-RPC frames.
2. [`mcp_client_basics.py`](./mcp_client_basics.py): Spawns `mcp_stdio_demo_server.py` with `subprocess.Popen`, runs the `initialize` + `notifications/initialized` handshake, discovers tools via `tools/list`, converts MCP tool schemas (`inputSchema`) into OpenAI-compatible agent schemas (`{"type": "function", "function": {...}}`), calls tools via `tools/call` (including an `isError=True` case), and inspects Resources and Prompts.

### Running the End-to-End Demo

Run the client from the repository root (it automatically spawns the sibling server script):

```bash
python3 15_Day_Model_Context_Protocol_Basics/mcp_client_basics.py
```

Sample output (abbreviated):

```text
Spawning MCP server over stdio: mcp_stdio_demo_server.py
========================================================================
1. Handshake complete -> ops-demo-mcp-server v1.0.0 (protocol 2025-03-26)
   Capabilities: tools, resources, prompts

2. Discovered 3 MCP tool(s) via tools/list:
   - get_service_status (required args: ['service_name']): Fetch current health status, latency, and error rate for a named service.
   - list_active_incidents (required args: []): List all currently open operational incidents, optionally filtered by severity.
   - calculate_error_budget (required args: ['sla_pct']): Compute allowed downtime minutes for a given availability SLA percentage over a window in days.

3. Converted first MCP tool into OpenAI-compatible agent schema:
{
  "type": "function",
  "function": {
    "name": "get_service_status",
    "description": "Fetch current health status, latency, and error rate for a named service.",
    "parameters": {
      "type": "object",
      "properties": {
        "service_name": {
          "type": "string",
          "description": "Service identifier (e.g., payments-api, checkout-web, webhook-dispatcher)."
        }
      },
      "required": ["service_name"],
      "additionalProperties": false
    }
  }
}

4. Invoking discovered tools via tools/call:
   [tools/call] list_active_incidents -> isError=False
   ...
   [tools/call] get_service_status(payments-api) -> isError=False
   ...
   [tools/call] calculate_error_budget(99.95%, 30d) -> isError=False
   ...
   [tools/call] get_service_status(unknown-svc) -> isError=True
   Error text returned to agent: Unknown service 'unknown-svc'. Known services: checkout-web, payments-api, webhook-dispatcher.

5. Discovered 1 MCP resource(s) via resources/list:
   - ops://catalog/services (application/json): Service Ownership Catalog
...
6. Discovered 1 MCP prompt(s) via prompts/list:
   - incident_triage: Structured checklist prompt for triaging a degraded service.
```

Inspect every raw JSON-RPC 2.0 frame and server `stderr` log line with `--verbose`:

```bash
python3 15_Day_Model_Context_Protocol_Basics/mcp_client_basics.py --verbose
```

---

## Equivalent Official `mcp` Python SDK Client

Once you understand the wire protocol above, you rarely need to hand-write JSON-RPC framing in production. With the official `mcp` Python SDK (`pip install mcp`), the exact same client flow looks like this:

```python
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_sdk_client() -> None:
    server_path = Path("15_Day_Model_Context_Protocol_Basics/mcp_stdio_demo_server.py").resolve()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Sends `initialize` + `notifications/initialized`
            init_result = await session.initialize()
            print("Connected to:", init_result.serverInfo.name)

            # Discover tools (`tools/list`)
            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                print(f"- {tool.name}: {tool.description}")

            # Call a tool (`tools/call`)
            result = await session.call_tool(
                "get_service_status",
                arguments={"service_name": "payments-api"},
            )
            print("Tool result:", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(run_sdk_client())
```

Because `mcp_stdio_demo_server.py` speaks standard MCP JSON-RPC 2.0 over stdio, that official SDK client can connect directly to our stdlib server without changing a single line on the server side. Tomorrow (Day 16), we build full custom MCP servers with `FastMCP`, strict schemas, and safety boundaries.

---

## Common Pitfalls & Bugs Developers Run Into

1. **Using `print(...)` inside a `stdio` MCP server:** Plain `print()` writes to `stdout`, which injects non-JSON text into the JSON-RPC stream and crashes the client parser immediately. Always direct logs to `sys.stderr` (or a file).
2. **Forgetting `notifications/initialized`:** Sending `tools/list` immediately after `initialize` without the initialized notification violates the MCP lifecycle; stricter SDK servers reject or stall subsequent calls.
3. **Treating `isError: true` as a crash:** Tool errors (bad ID, empty search result, validation error) should be returned cleanly inside the agent's tool result so the LLM can correct its arguments, whereas transport disconnects or JSON-RPC `-32601` errors indicate infrastructure bugs.
4. **Blinding your host to tool name collisions:** When one Host connects to three MCP servers, two servers might both expose a tool named `search`. Prefix discovered tool names internally (for example, `github__search` vs. `docs__search`) and map back when dispatching `tools/call`.
5. **Trusting third-party MCP tool descriptions blindly:** Tool descriptions and tool outputs enter the LLM's context window. Malicious or overly broad MCP servers can inject instructions into descriptions ("tool poisoning") or read sensitive local files. Only connect trusted servers and apply host-side approval policies.

---

## 💻 Exercises - Day 15

### Exercise: Level 1

1. Run `mcp_client_basics.py --verbose` and trace the exact JSON-RPC `id` numbers for `initialize`, `tools/list`, each `tools/call`, `resources/list`, and `prompts/list`. Confirm `notifications/initialized` has no `"id"` field.
2. Temporarily add a plain `print("debug hello")` inside `handle_message()` in `mcp_stdio_demo_server.py`, run `mcp_client_basics.py`, and observe how it breaks `stdout` JSON parsing. Then fix it by routing it to `sys.stderr`.
3. Add a `get_prompt()` method to `StdioMCPClient` in `mcp_client_basics.py` that calls `prompts/get` for `"incident_triage"` with `{"service_name": "payments-api"}` and prints the returned prompt message.

### Exercise: Level 2

1. Add a 4th tool `acknowledge_incident` to `mcp_stdio_demo_server.py` (accepting `incident_id` and `owner`), validate that the incident exists, update its in-memory state, and call it from `mcp_client_basics.py`.
2. Add a request timeout to `StdioMCPClient.request()` (for example using `select` or a worker thread) so a hung MCP server raises a clear `TimeoutError` instead of blocking forever on `stdout.readline()`.
3. Connect `mcp_client_basics.py` to an OpenAI-compatible model (or a small mock loop): pass the output of `mcp_tools_to_openai_tools()` into `/chat/completions`, let the model pick a tool call, forward the tool call to `client.call_tool()`, and feed the text back to the model for a final answer.

### Exercise: Level 3

1. Build a **multi-server MCP Host registry** (`MultiServerMCPHost`) that spawns two independent stdio MCP servers, namespaces their discovered tools (`<server_name>__<tool_name>`), prevents collisions, and routes each `tools/call` to the matching child process.
2. Add a **schema validator** on the client side before sending `tools/call`: inspect the server's advertised `inputSchema` (`required` fields, basic JSON types, `enum` values, `additionalProperties: false`) and reject invalid model-generated arguments locally before sending them across the wire.
3. Add a simple **client-side policy hook** (`allowed_tools` allowlist + confirmation prompt for write tools) so your Host enforces least-privilege permissions even if an MCP server advertises 20 tools.

---

[<< Day 14](../14_Day_Agentic_RAG/README.md) | [Home](../README.md) | [Day 16 >>](../16_Day_Building_Custom_MCP_Servers/README.md)
