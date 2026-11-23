"""Custom MCP server exposing a notes store over JSON-RPC 2.0 (stdio).

This server uses only the Python standard library so you can see the exact
wire-level mechanics of the Model Context Protocol (MCP):
  - Handshake (`initialize`, `notifications/initialized`, `ping`)
  - Tools (`tools/list`, `tools/call`) with JSON Schema validation
  - Resources (`resources/list`, `resources/read`) addressed via `notes://` URIs
  - Structured error responses (protocol errors vs. tool-level `isError`)
  - Strict stdout hygiene (all logs go to stderr; stdout carries JSON-RPC only)

Usage:
  # Run as a stdio MCP server (typically launched by an MCP client)
  python3 16_Day_Building_Custom_MCP_Servers/notes_mcp_server.py

  # Run an offline protocol self-test without an external client
  python3 16_Day_Building_Custom_MCP_Servers/notes_mcp_server.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any

# CRITICAL FOR STDIO MCP SERVERS:
# Never write logs or debug print() calls to sys.stdout.
# The MCP client parses every line on stdout as a JSON-RPC 2.0 frame.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[notes-mcp-server] %(levelname)s: %(message)s",
)
logger = logging.getLogger("notes_mcp_server")

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "custom-notes-mcp"
SERVER_VERSION = "1.0.0"

# Standard JSON-RPC 2.0 & MCP error codes
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_RESOURCE_NOT_FOUND = -32002


@dataclass
class Note:
    """A single note stored by the MCP server."""

    id: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def uri(self) -> str:
        return f"notes://{self.id}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["uri"] = self.uri
        return payload


class NoteStore:
    """In-memory note repository with optional JSON file persistence."""

    def __init__(self, store_path: Path | None = None, seed: bool = True) -> None:
        self._store_path = store_path
        self._notes: dict[str, Note] = {}
        self._counter = 0

        if store_path and store_path.exists():
            self._load_from_disk()
        elif seed:
            self._seed_defaults()

    def _seed_defaults(self) -> None:
        self.create_note(
            title="Database Failover Runbook",
            content=(
                "Verify replica lag is under 500ms before promoting standby. "
                "Drain read-only pool first, then update primary DNS with 30s TTL."
            ),
            tags=["database", "ops", "runbook"],
        )
        self.create_note(
            title="API Rate Limit Policy",
            content=(
                "Default tenant quota is 120 requests/minute with burst 30. "
                "Return HTTP 429 with a Retry-After header when exceeded."
            ),
            tags=["api", "limits"],
        )

    def _load_from_disk(self) -> None:
        assert self._store_path is not None
        raw = json.loads(self._store_path.read_text(encoding="utf-8"))
        for item in raw.get("notes", []):
            note = Note(
                id=item["id"],
                title=item["title"],
                content=item["content"],
                tags=list(item.get("tags", [])),
                created_at=item.get("created_at", ""),
            )
            self._notes[note.id] = note
        self._counter = int(raw.get("counter", len(self._notes)))

    def _save_to_disk(self) -> None:
        if not self._store_path:
            return
        payload = {
            "counter": self._counter,
            "notes": [asdict(n) for n in self._notes.values()],
        }
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def create_note(
        self, title: str, content: str, tags: list[str] | None = None
    ) -> Note:
        self._counter += 1
        note_id = f"note-{self._counter}"
        normalized_tags = sorted({t.strip().lower() for t in (tags or []) if t.strip()})
        note = Note(
            id=note_id,
            title=title.strip(),
            content=content.strip(),
            tags=normalized_tags,
        )
        self._notes[note_id] = note
        self._save_to_disk()
        return note

    def get_note(self, note_id: str) -> Note | None:
        return self._notes.get(note_id)

    def list_notes(self) -> list[Note]:
        return list(self._notes.values())

    def search_notes(self, query: str, tag: str | None = None) -> list[Note]:
        needle = query.strip().lower()
        tag_filter = tag.strip().lower() if tag else None
        matches: list[Note] = []
        for note in self._notes.values():
            if tag_filter and tag_filter not in note.tags:
                continue
            haystack = f"{note.title}\n{note.content}\n{' '.join(note.tags)}".lower()
            if not needle or needle in haystack:
                matches.append(note)
        return matches

    def delete_note(self, note_id: str) -> Note | None:
        removed = self._notes.pop(note_id, None)
        if removed is not None:
            self._save_to_disk()
        return removed


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "create_note",
        "description": (
            "Create a new note in the knowledge base and return its ID and notes:// URI."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Short, descriptive title for the note.",
                },
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Body text of the note.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of lowercase topic tags.",
                },
            },
            "required": ["title", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_notes",
        "description": (
            "Search notes by keyword across titles, content, and tags, with an optional tag filter."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword or phrase (empty string lists all matching tag).",
                },
                "tag": {
                    "type": "string",
                    "description": "Optional exact tag filter (e.g., 'runbook' or 'api').",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "delete_note",
        "description": "Delete an existing note by its note_id (for example, 'note-1').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Identifier of the note to delete.",
                }
            },
            "required": ["note_id"],
            "additionalProperties": False,
        },
    },
]

TOOL_SCHEMA_MAP: dict[str, dict[str, Any]] = {
    tool["name"]: tool["inputSchema"] for tool in TOOL_DEFINITIONS
}


def validate_against_schema(
    arguments: Any, schema: dict[str, Any]
) -> list[str]:
    """Validate tool call arguments against our subset of JSON Schema."""
    errors: list[str] = []
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]

    properties: dict[str, dict[str, Any]] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    for field_name in required:
        if field_name not in arguments:
            errors.append(f"missing required field '{field_name}'")

    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments.keys()) - set(properties.keys()))
        if unknown:
            errors.append(f"unexpected field(s): {', '.join(unknown)}")

    for key, value in arguments.items():
        if key not in properties:
            continue
        prop = properties[key]
        expected_type = prop.get("type")
        if expected_type == "string":
            if not isinstance(value, str):
                errors.append(f"field '{key}' must be a string")
            elif prop.get("minLength", 0) > 0 and len(value.strip()) < prop["minLength"]:
                errors.append(
                    f"field '{key}' must be a non-empty string (minLength={prop['minLength']})"
                )
        elif expected_type == "array":
            if not isinstance(value, list):
                errors.append(f"field '{key}' must be an array")
            else:
                item_type = prop.get("items", {}).get("type")
                if item_type == "string" and not all(isinstance(v, str) for v in value):
                    errors.append(f"all items in '{key}' must be strings")

    return errors


def _tool_result(payload: dict[str, Any], is_error: bool = False) -> dict[str, Any]:
    """Wrap a dict payload in an MCP CallToolResult envelope."""
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": is_error,
    }


def _jsonrpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(
    req_id: Any, code: int, message: str, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


class NotesMCPServer:
    """JSON-RPC 2.0 dispatcher implementing MCP tools and resources."""

    def __init__(self, store: NoteStore) -> None:
        self.store = store
        self.initialized = False

    def handle_request_dict(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return _jsonrpc_error(
                message.get("id"),
                ERR_INVALID_REQUEST,
                "Invalid JSON-RPC 2.0 request",
            )

        method: str = message["method"]
        params: Any = message.get("params") or {}
        is_notification = "id" not in message
        req_id = message.get("id")

        # Notifications do not receive a JSON-RPC response frame.
        if is_notification:
            if method == "notifications/initialized":
                self.initialized = True
                logger.info("Client completed initialization handshake.")
            else:
                logger.info("Received notification: %s", method)
            return None

        if not isinstance(params, dict):
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "params must be an object")

        try:
            if method == "initialize":
                return _jsonrpc_result(req_id, self._handle_initialize(params))
            if method == "ping":
                return _jsonrpc_result(req_id, {})
            if method == "tools/list":
                return _jsonrpc_result(req_id, {"tools": TOOL_DEFINITIONS})
            if method == "tools/call":
                return self._handle_tools_call(req_id, params)
            if method == "resources/list":
                return _jsonrpc_result(req_id, self._handle_resources_list())
            if method == "resources/read":
                return self._handle_resources_read(req_id, params)
            return _jsonrpc_error(
                req_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method}"
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.exception("Unhandled error while processing %s", method)
            return _jsonrpc_error(req_id, ERR_INTERNAL, f"Internal server error: {exc}")

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        client_info = params.get("clientInfo", {})
        logger.info(
            "Initialize requested by client=%s protocol=%s",
            client_info.get("name", "unknown"),
            params.get("protocolVersion", "unspecified"),
        )
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def _handle_tools_call(
        self, req_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not isinstance(tool_name, str) or tool_name not in TOOL_SCHEMA_MAP:
            return _jsonrpc_error(
                req_id,
                ERR_INVALID_PARAMS,
                f"Unknown tool: {tool_name!r}",
                data={"availableTools": sorted(TOOL_SCHEMA_MAP.keys())},
            )

        validation_errors = validate_against_schema(
            arguments, TOOL_SCHEMA_MAP[tool_name]
        )
        if validation_errors:
            # Return a structured tool error (isError=True) rather than crashing
            # the protocol so an agent can inspect the message and retry.
            return _jsonrpc_result(
                req_id,
                _tool_result(
                    {
                        "error": "validation_error",
                        "tool": tool_name,
                        "details": validation_errors,
                    },
                    is_error=True,
                ),
            )

        if tool_name == "create_note":
            note = self.store.create_note(
                title=arguments["title"],
                content=arguments["content"],
                tags=arguments.get("tags"),
            )
            logger.info("Created %s (%s)", note.id, note.title)
            return _jsonrpc_result(
                req_id,
                _tool_result({"status": "created", "note": note.to_dict()}),
            )

        if tool_name == "search_notes":
            matches = self.store.search_notes(
                query=arguments["query"],
                tag=arguments.get("tag"),
            )
            return _jsonrpc_result(
                req_id,
                _tool_result(
                    {
                        "query": arguments["query"],
                        "tag": arguments.get("tag"),
                        "count": len(matches),
                        "matches": [
                            {
                                "id": n.id,
                                "uri": n.uri,
                                "title": n.title,
                                "tags": n.tags,
                                "snippet": n.content[:140],
                            }
                            for n in matches
                        ],
                    }
                ),
            )

        if tool_name == "delete_note":
            note_id = arguments["note_id"]
            removed = self.store.delete_note(note_id)
            if removed is None:
                return _jsonrpc_result(
                    req_id,
                    _tool_result(
                        {
                            "error": "not_found",
                            "message": f"No note exists with id '{note_id}'.",
                            "existing_ids": [n.id for n in self.store.list_notes()],
                        },
                        is_error=True,
                    ),
                )
            logger.info("Deleted %s", note_id)
            return _jsonrpc_result(
                req_id,
                _tool_result({"status": "deleted", "deleted_id": removed.id}),
            )

        return _jsonrpc_error(req_id, ERR_INTERNAL, f"Unhandled tool: {tool_name}")

    def _handle_resources_list(self) -> dict[str, Any]:
        resources: list[dict[str, Any]] = [
            {
                "uri": "notes://index",
                "name": "Notes Index",
                "description": "Catalog of all stored notes with IDs, titles, tags, and URIs.",
                "mimeType": "application/json",
            }
        ]
        for note in self.store.list_notes():
            resources.append(
                {
                    "uri": note.uri,
                    "name": note.title,
                    "description": f"Note {note.id} (tags: {', '.join(note.tags) or 'none'})",
                    "mimeType": "application/json",
                }
            )
        return {"resources": resources}

    def _handle_resources_read(
        self, req_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str) or not uri.startswith("notes://"):
            return _jsonrpc_error(
                req_id,
                ERR_INVALID_PARAMS,
                "uri must be a string starting with 'notes://'",
            )

        if uri == "notes://index":
            index_payload = {
                "total": len(self.store.list_notes()),
                "notes": [
                    {
                        "id": n.id,
                        "uri": n.uri,
                        "title": n.title,
                        "tags": n.tags,
                        "created_at": n.created_at,
                    }
                    for n in self.store.list_notes()
                ],
            }
            return _jsonrpc_result(
                req_id,
                {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(index_payload, indent=2),
                        }
                    ]
                },
            )

        note_id = uri.removeprefix("notes://")
        note = self.store.get_note(note_id)
        if note is None:
            return _jsonrpc_error(
                req_id,
                ERR_RESOURCE_NOT_FOUND,
                f"Resource not found: {uri}",
                data={"uri": uri},
            )

        return _jsonrpc_result(
            req_id,
            {
                "contents": [
                    {
                        "uri": note.uri,
                        "mimeType": "application/json",
                        "text": json.dumps(note.to_dict(), indent=2),
                    }
                ]
            },
        )

    def handle_raw_line(self, line: str) -> str | None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            err = _jsonrpc_error(None, ERR_PARSE, f"JSON parse error: {exc.msg}")
            return json.dumps(err)

        if not isinstance(message, dict):
            err = _jsonrpc_error(
                None, ERR_INVALID_REQUEST, "JSON-RPC payload must be an object"
            )
            return json.dumps(err)

        response = self.handle_request_dict(message)
        if response is None:
            return None
        return json.dumps(response)

    def serve_stdio(self) -> None:
        """Read newline-delimited JSON-RPC messages from stdin and reply on stdout."""
        logger.info("Listening for MCP JSON-RPC frames on stdio...")
        for raw_line in sys.stdin:
            stripped = raw_line.strip()
            if not stripped:
                continue
            reply = self.handle_raw_line(stripped)
            if reply is not None:
                sys.stdout.write(reply + "\n")
                sys.stdout.flush()
        logger.info("stdin closed; shutting down server.")


def run_self_test() -> None:
    """Exercise all server handlers in-process to verify protocol behavior."""
    server = NotesMCPServer(NoteStore(seed=True))
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "clientInfo": {"name": "self-test", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/read",
            "params": {"uri": "notes://index"},
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "create_note",
                "arguments": {
                    "title": "MCP Stdout Hygiene",
                    "content": "Send all server logs to stderr so stdout stays valid JSON-RPC.",
                    "tags": ["mcp", "stdio"],
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "delete_note",
                "arguments": {"note_id": "note-999"},
            },
        },
    ]

    for req in requests:
        reply = server.handle_request_dict(req)
        print(f"-> {req['method']}")
        if reply is None:
            print("   (notification: no response sent)")
        elif "error" in reply:
            print(f"   ERROR {reply['error']['code']}: {reply['error']['message']}")
        else:
            summary_keys = list(reply["result"].keys())
            print(f"   OK result keys={summary_keys}")
            if req["method"] == "tools/call" and reply["result"].get("isError"):
                print(f"   Tool reported isError=True -> {reply['result']['content'][0]['text']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom stdio MCP server for notes.")
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Optional JSON file path to persist notes across restarts.",
    )
    parser.add_argument(
        "--empty",
        action="store_true",
        help="Start with an empty note store instead of seeded sample notes.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run an in-process protocol check and exit instead of reading stdin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    store = NoteStore(store_path=args.store, seed=not args.empty)
    server = NotesMCPServer(store=store)
    server.serve_stdio()


if __name__ == "__main__":
    main()
