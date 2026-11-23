"""Idiomatic custom MCP server built with the official `mcp` SDK (`FastMCP`).

Compare this file with `notes_mcp_server.py`:
- `notes_mcp_server.py` implements the JSON-RPC 2.0 protocol manually with stdlib
  so you understand every byte on the wire.
- `fastmcp_example.py` exposes the exact same tools and `notes://` resources
  using `mcp.server.fastmcp.FastMCP`, which generates JSON Schemas from Python
  type hints and handles stdio framing automatically.

Prerequisites:
  pip install "mcp>=1.2.0"

Usage:
  python3 16_Day_Building_Custom_MCP_Servers/fastmcp_example.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import sys

# Always log to stderr in stdio MCP servers; stdout is reserved for JSON-RPC.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[fastmcp-notes] %(levelname)s: %(message)s",
)
logger = logging.getLogger("fastmcp_notes")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The official `mcp` package is not installed yet.\n"
        "Install it with: pip install 'mcp>=1.2.0'\n"
        "Or run the zero-dependency version: "
        "python3 16_Day_Building_Custom_MCP_Servers/agent_with_custom_mcp.py --demo"
    ) from exc


@dataclass
class Note:
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


mcp = FastMCP("fastmcp-notes-server")

NOTES: dict[str, Note] = {
    "note-1": Note(
        id="note-1",
        title="Database Failover Runbook",
        content=(
            "Verify replica lag is under 500ms before promoting standby. "
            "Drain read-only pool first, then update primary DNS with 30s TTL."
        ),
        tags=["database", "ops", "runbook"],
    ),
    "note-2": Note(
        id="note-2",
        title="API Rate Limit Policy",
        content=(
            "Default tenant quota is 120 requests/minute with burst 30. "
            "Return HTTP 429 with a Retry-After header when exceeded."
        ),
        tags=["api", "limits"],
    ),
}
_NEXT_ID = 3


@mcp.tool()
def create_note(title: str, content: str, tags: list[str] | None = None) -> str:
    """Create a new note and return its ID and `notes://` resource URI."""
    global _NEXT_ID
    clean_title = title.strip()
    clean_content = content.strip()
    if not clean_title or not clean_content:
        raise ValueError("Both 'title' and 'content' must be non-empty strings.")

    note_id = f"note-{_NEXT_ID}"
    _NEXT_ID += 1
    normalized_tags = sorted({t.strip().lower() for t in (tags or []) if t.strip()})
    note = Note(
        id=note_id,
        title=clean_title,
        content=clean_content,
        tags=normalized_tags,
    )
    NOTES[note_id] = note
    logger.info("Created %s (%s)", note.id, note.title)
    payload = asdict(note)
    payload["uri"] = note.uri
    return json.dumps({"status": "created", "note": payload}, indent=2)


@mcp.tool()
def search_notes(query: str, tag: str | None = None) -> str:
    """Search stored notes by keyword across titles, content, and tags."""
    needle = query.strip().lower()
    tag_filter = tag.strip().lower() if tag else None
    matches = []
    for note in NOTES.values():
        if tag_filter and tag_filter not in note.tags:
            continue
        haystack = f"{note.title}\n{note.content}\n{' '.join(note.tags)}".lower()
        if not needle or needle in haystack:
            matches.append(
                {
                    "id": note.id,
                    "uri": note.uri,
                    "title": note.title,
                    "tags": note.tags,
                    "snippet": note.content[:140],
                }
            )
    return json.dumps({"query": query, "count": len(matches), "matches": matches}, indent=2)


@mcp.tool()
def delete_note(note_id: str) -> str:
    """Delete a note by ID. Raises ValueError if the note does not exist."""
    removed = NOTES.pop(note_id, None)
    if removed is None:
        existing = ", ".join(sorted(NOTES.keys())) or "none"
        raise ValueError(f"Note '{note_id}' was not found. Existing IDs: {existing}")
    logger.info("Deleted %s", note_id)
    return json.dumps({"status": "deleted", "deleted_id": removed.id}, indent=2)


@mcp.resource("notes://index", mime_type="application/json")
def read_notes_index() -> str:
    """Read-only catalog listing every stored note and its resource URI."""
    payload = {
        "total": len(NOTES),
        "notes": [
            {
                "id": note.id,
                "uri": note.uri,
                "title": note.title,
                "tags": note.tags,
                "created_at": note.created_at,
            }
            for note in NOTES.values()
        ],
    }
    return json.dumps(payload, indent=2)


@mcp.resource("notes://{note_id}", mime_type="application/json")
def read_single_note(note_id: str) -> str:
    """Read-only view of a single note addressed by `notes://<note_id>`."""
    note = NOTES.get(note_id)
    if note is None:
        raise ValueError(f"Resource not found: notes://{note_id}")
    payload = asdict(note)
    payload["uri"] = note.uri
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    logger.info("Starting FastMCP notes server on stdio...")
    mcp.run(transport="stdio")
