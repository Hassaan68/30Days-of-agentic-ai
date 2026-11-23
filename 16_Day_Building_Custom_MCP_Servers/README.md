[<< Day 15](../15_Day_Model_Context_Protocol_Basics/README.md) | [Home](../README.md) | [Day 17 >>](../17_Day_Stateful_Workflows_and_Checkpointing/README.md)

# Day 16: Building Custom MCP Servers

Yesterday you learned how an agent connects to an existing Model Context Protocol (MCP) server. Today you build your own custom MCP server from scratch—exposing both **tools** (actions with JSON Schema input validation) and **resources** (`notes://...` URIs for read-only context)—and wire it into a working agent over `stdio`. You will also see the exact same server implemented with the official `FastMCP` SDK so you know both the wire protocol and the ergonomic production library.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
  - [Tools vs. Resources in MCP](#tools-vs-resources-in-mcp)
  - [Why Stdout Hygiene Makes or Breaks Stdio Servers](#why-stdout-hygiene-makes-or-breaks-stdio-servers)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
  - [JSON-RPC 2.0 Lifecycle & Method Map](#json-rpc-20-lifecycle--method-map)
  - [Protocol Errors vs. Tool Errors (`isError: true`)](#protocol-errors-vs-tool-errors-iserror-true)
- [Walkthrough of the Python Examples](#walkthrough-of-the-python-examples)
  - [1. Wire-Level Server (`notes_mcp_server.py`)](#1-wire-level-server-notes_mcp_serverpy)
  - [2. Agent Client Wired to the Custom Server (`agent_with_custom_mcp.py`)](#2-agent-client-wired-to-the-custom-server-agent_with_custom_mcppy)
  - [3. Official SDK Version (`fastmcp_example.py`)](#3-official-sdk-version-fastmcp_examplepy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 16](#-exercises---day-16)

---

## Core Concept & Why It Matters

Most real teams do not stop at public MCP servers like filesystem or GitHub connectors. Your actual business logic lives in internal services: runbook databases, ticketing queues, billing tables, feature flags, or deployment checklists. Wrapping those internal capabilities in a custom MCP server decouples your agent runtime from your backend integrations.

Once a capability is exposed behind an MCP server:
- Any MCP-compatible client (your own Python agent, an IDE assistant, or a CLI tool) can discover and call it without duplicating glue code.
- Input validation, authentication, and data formatting stay inside the server boundary instead of leaking into prompts.
- Teams can version, test, and audit the server independently from the LLM loop.

The trade-off is an extra process boundary (or HTTP hop for remote transports). For tiny one-off scripts, direct Python functions are simpler. Once two or more agents or clients share the same integration—or you want process isolation around credentials—a custom MCP server pays for itself quickly.

### Tools vs. Resources in MCP

MCP separates **tools** from **resources** on purpose:

| Primitive | Controlled By | Purpose | Side Effects? | Example in Today's Server |
| :--- | :--- | :--- | :--- | :--- |
| **Tools** (`tools/list`, `tools/call`) | Usually the **model** | Perform actions, queries with parameters, or state mutations | Often yes | `create_note`, `search_notes`, `delete_note` |
| **Resources** (`resources/list`, `resources/read`) | Often the **host/client** (or read-only bridge) | Expose addressable read-only data identified by URIs | No (read-only) | `notes://index`, `notes://note-1` |

Keeping read-only state as resources lets the host application preload a lightweight catalog (`notes://index`) into the system prompt before the first model turn, then fetch individual documents (`notes://note-1`) only when needed—saving tokens and avoiding unnecessary write-capable tool exposure.

### Why Stdout Hygiene Makes or Breaks Stdio Servers

Over the `stdio` transport, the client spawns your server as a subprocess and reads **every line written to `stdout`** as a JSON-RPC 2.0 frame.

A single stray `print("Server started!")` or a third-party library logging to `stdout` corrupts the stream immediately. The client tries to run `json.loads("Server started!")`, throws a parse error, and drops the connection.

The rule for stdio MCP servers is absolute:
- **`sys.stdout`** is exclusively for newline-delimited JSON-RPC responses.
- **`sys.stderr`** is for all human-readable logs, tracebacks, and diagnostics.

```python
logging.basicConfig(
    stream=sys.stderr,  # NEVER sys.stdout
    level=logging.INFO,
    format="[notes-mcp-server] %(levelname)s: %(message)s",
)
```

---

## How It Works Under the Hood

Here is the full communication flow between our agent host (`agent_with_custom_mcp.py`) and the custom server subprocess (`notes_mcp_server.py`):

```text
+-----------------------------------+          stdin (JSON-RPC requests)          +-----------------------------+
|  Agent Host / MCPStdioClient      | ------------------------------------------> |  notes_mcp_server.py        |
|                                   |                                             |                             |
|  1. initialize + initialized      | <------------------------------------------ |  NotesMCPServer             |
|  2. tools/list & resources/list   |         stdout (JSON-RPC responses)         |   - validate_against_schema |
|  3. resources/read (notes://index)|                                             |   - NoteStore (CRUD)        |
|  4. LLM loop -> tools/call        | < - - - - - stderr (server logs) - - - - -  |   - stderr logger only      |
+-----------------------------------+                                             +-----------------------------+
```

### JSON-RPC 2.0 Lifecycle & Method Map

Every request frame contains `jsonrpc: "2.0"`, an integer `id`, a `method` name, and a `params` object. Notifications omit `id` and **must not** receive a reply.

1. **`initialize`** (request): Client sends its protocol version and `clientInfo`. Server responds with supported `protocolVersion`, `capabilities` (`tools`, `resources`), and `serverInfo`.
2. **`notifications/initialized`** (notification): Client signals handshake completion. Server logs it and writes nothing to stdout.
3. **`tools/list`**: Returns tool names, descriptions, and JSON Schema `inputSchema` objects.
4. **`resources/list`**: Returns available resource URIs (`notes://index`, `notes://note-1`, ...), names, and MIME types.
5. **`resources/read`**: Takes `{"uri": "notes://note-1"}` and returns `{"contents": [{"uri": ..., "mimeType": ..., "text": ...}]}`.
6. **`tools/call`**: Takes `{"name": "create_note", "arguments": {...}}`, validates the arguments against `inputSchema`, executes the handler, and returns `{"content": [{"type": "text", "text": "..."}], "isError": false}`.

### Protocol Errors vs. Tool Errors (`isError: true`)

An easy design mistake is treating every failure as a top-level JSON-RPC error. MCP distinguishes two categories:

- **JSON-RPC protocol errors** (`{"error": {"code": -32601, "message": ...}}`): Use these when the wire contract itself is broken—invalid JSON (`-32700`), unknown method (`-32601`), unknown tool name (`-32602`), or missing resource URI (`-32002`).
- **Tool execution errors** (`{"result": {"content": [...], "isError": true}}`): Use these when the call is valid JSON-RPC, but the business operation failed—for example, a missing required field, an empty title, or `delete_note("note-999")` where the note ID does not exist. Returning structured JSON inside `content` with `isError: true` lets the LLM read the explanation (such as `existing_ids`) and recover on the next turn instead of crashing the client loop.

---

## Walkthrough of the Python Examples

### 1. Wire-Level Server (`notes_mcp_server.py`)

[notes_mcp_server.py](./notes_mcp_server.py) uses only the Python standard library. It implements:
- `NoteStore`: an in-memory note repository pre-seeded with two runbooks (plus optional `--store notes.json` file persistence).
- `validate_against_schema()`: enforces required fields, string types, non-empty `minLength`, string arrays, and `additionalProperties: False`.
- `NotesMCPServer`: handles `initialize`, `notifications/initialized`, `ping`, `tools/list`, `tools/call`, `resources/list`, and `resources/read`.

You can verify the server standalone using its built-in protocol self-test:

```bash
python3 16_Day_Building_Custom_MCP_Servers/notes_mcp_server.py --self-test
```

### 2. Agent Client Wired to the Custom Server (`agent_with_custom_mcp.py`)

[agent_with_custom_mcp.py](./agent_with_custom_mcp.py) spawns `notes_mcp_server.py` as a subprocess, drains its `stderr` logs on a background thread (preventing pipe deadlocks), translates MCP tool schemas into OpenAI-compatible function schemas, preloads `notes://index`, and runs a multi-step agent loop.

Run it offline with the deterministic mock LLM (still talking to the **real** custom MCP server over `stdio`):

```bash
python3 16_Day_Building_Custom_MCP_Servers/agent_with_custom_mcp.py --demo
```

Sample output:

```text
=== Day 16: Agent + Custom MCP Server [DEMO (mock LLM + real custom MCP server)] ===

Connected to MCP server: custom-notes-mcp v1.0.0 (protocol 2024-11-05)
Discovered MCP tools: ['create_note', 'search_notes', 'delete_note']
Discovered MCP resources: ['notes://index', 'notes://note-1', 'notes://note-2']

[Step 1] Tool call -> search_notes({"query": "failover"})
[Step 1] Observation <- { "query": "failover", "count": 1, "matches": [{ "id": "note-1", "uri": "notes://note-1", ... }] }

[Step 2] Tool call -> read_mcp_resource({"uri": "notes://note-1"})
[Step 2] Observation <- { "id": "note-1", "title": "Database Failover Runbook", ... }

[Step 3] Tool call -> create_note({"title": "Redis Cache Failover Runbook", ...})
[Step 3] Observation <- { "status": "created", "note": { "id": "note-3", "uri": "notes://note-3", ... } }

[Step 4] Tool call -> delete_note({"note_id": "note-999"})
[Step 4] Observation <- [MCP TOOL ERROR isError=True]
{
  "error": "not_found",
  "message": "No note exists with id 'note-999'.",
  "existing_ids": ["note-1", "note-2", "note-3"]
}

--- Updated MCP Resources After Run ---
  - notes://index: Notes Index
  - notes://note-1: Database Failover Runbook
  - notes://note-2: API Rate Limit Policy
  - notes://note-3: Redis Cache Failover Runbook

--- MCP Server stderr Logs (kept separate from stdout!) ---
  [notes-mcp-server] INFO: Listening for MCP JSON-RPC frames on stdio...
  [notes-mcp-server] INFO: Initialize requested by client=day16-agent-client protocol=2024-11-05
  [notes-mcp-server] INFO: Client completed initialization handshake.
  [notes-mcp-server] INFO: Created note-3 (Redis Cache Failover Runbook)
```

To run against a live model (OpenAI or local Ollama), set your `.env` variables and drop `--demo`:

```bash
OPENAI_API_KEY=your-key python3 16_Day_Building_Custom_MCP_Servers/agent_with_custom_mcp.py \
  --task "Search for rate limit notes and add a note documenting our 429 retry backoff."
```

### 3. Official SDK Version (`fastmcp_example.py`)

Once you have installed `mcp>=1.2.0` (`pip install -r requirements.txt`), [fastmcp_example.py](./fastmcp_example.py) shows how `mcp.server.fastmcp.FastMCP` replaces manual JSON-RPC routing with `@mcp.tool()` and `@mcp.resource("notes://{note_id}")` decorators while preserving the same protocol contract and `stderr`-only logging discipline.

```bash
python3 16_Day_Building_Custom_MCP_Servers/fastmcp_example.py
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Printing debug messages to `stdout`.** Even one `print("debug:", args)` breaks JSON-RPC framing in stdio clients. Always configure logging to `sys.stderr`, and watch out for third-party libraries that print banners on import.
2. **Forgetting to flush `stdout`.** Python buffers `stdout` when piped to another process instead of a terminal. If you omit `sys.stdout.flush()` after writing a JSON-RPC reply, the client hangs waiting for a response that is still sitting in a buffer.
3. **Not draining `stderr` in the client.** If a chatty MCP server writes more than ~64 KB to `stderr` and the parent process never reads `proc.stderr`, the OS pipe buffer fills up and the server blocks forever on its next log call. Always drain `stderr` on a separate thread or redirect it to a log file.
4. **Replying to JSON-RPC notifications.** Notifications like `notifications/initialized` have no `id` field. Sending a response frame for them desynchronizes request/response ordering on strict clients.
5. **Leaking raw stack traces as tool outputs.** Uncaught exceptions should be logged to `stderr` and returned as sanitized, actionable JSON error payloads so the model knows what input to fix without exposing internal file paths or secrets.

---

## 💻 Exercises - Day 16

### Exercise: Level 1

1. Run `notes_mcp_server.py --self-test` and inspect every request/response pair. Add a test case to `run_self_test()` that sends a `create_note` call with an empty `title` (`""`) and verify that `isError` is `True`.
2. Add an `update_note` tool to `notes_mcp_server.py` (taking `note_id`, optional `title`, optional `content`, and optional `tags`) with a strict `inputSchema` and `additionalProperties: False`.
3. Add a `notes://tags` resource to `resources/list` and `resources/read` that returns a JSON map of each tag to the count and list of note IDs carrying that tag.
4. Intentionally add a `print("broken stdout")` inside `NotesMCPServer._handle_initialize` and run `agent_with_custom_mcp.py --demo` to observe the exact framing error, then revert it.

### Exercise: Level 2

1. Wire `agent_with_custom_mcp.py` to accept a `--store /tmp/day16_notes.json` flag and forward it to `notes_mcp_server.py`. Run the agent twice and verify that notes created in run 1 appear immediately in `notes://index` during run 2.
2. Extend `notes_mcp_server.py` to support MCP **resource templates** (`resources/templates/list`) declaring `notes://{note_id}` so clients know the URI pattern even when the note store starts empty (`--empty`).
3. Add pagination (`cursor` / `nextCursor`) to `tools/list` or `resources/list` so a server holding 500 notes does not dump all 500 URIs in a single response.
4. Update `agent_with_custom_mcp.py` to swap the subprocess command to `fastmcp_example.py` (once `mcp` is installed) and verify that the exact same client works with zero changes to the agent loop.

### Exercise: Level 3

1. Add a read-only mode flag (`--read-only`) to `notes_mcp_server.py` that hides mutating tools (`create_note`, `delete_note`) from `tools/list` AND rejects them at `tools/call` with a structured policy error if a client tries to invoke them anyway.
2. Add per-tool rate limiting and maximum payload size checks (for example, rejecting notes larger than 16 KB or more than 20 writes per minute) to protect the server from runaway agent loops.
3. Build a unit test suite using `unittest` or `pytest` that spawns `notes_mcp_server.py` as a real subprocess, tests concurrent/sequential requests, malformed JSON recovery (`-32700`), unknown methods (`-32601`), missing resources (`-32002`), and clean EOF shutdown.
4. Implement structured request correlation IDs and execution timing logs on `stderr` (method name, request ID, duration in milliseconds, success/error status) so operators can debug slow tool calls in production.

---

[<< Day 15](../15_Day_Model_Context_Protocol_Basics/README.md) | [Home](../README.md) | [Day 17 >>](../17_Day_Stateful_Workflows_and_Checkpointing/README.md)
