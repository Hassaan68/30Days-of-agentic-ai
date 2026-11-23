"""Day 3: Structured outputs and strict validation using Pydantic v2.

Demonstrates:
1. Defining strict nested Pydantic v2 models (`ConfigDict(extra="forbid")`, `Field` constraints).
2. Generating a JSON Schema (`model_json_schema()`) compatible with OpenAI `response_format`.
3. Parsing and validating raw JSON strings with `model_validate_json()`.
4. Catching `ValidationError` and extracting structured error details for logs or retries.
5. Calling an OpenAI-compatible API with `response_format` (or running in `--demo` mode).

Note: Requires `pydantic>=2.0` (`pip install pydantic`). If `pydantic` is not installed
yet, this script prints a clear setup hint and exits cleanly; you can run
`schema_validator_stdlib.py` immediately with zero third-party packages.

Examples:
    python3 03_Day_Structured_Outputs_and_Schema_Validation/pydantic_structured_output.py --demo
    python3 03_Day_Structured_Outputs_and_Schema_Validation/pydantic_structured_output.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError:
    print("`pydantic` is not installed in this Python environment.")
    print("Install it with:  pip install pydantic   (or: uv pip install -r requirements.txt)")
    print(
        "Tip: You can run the zero-dependency version right now:\n"
        "  python3 03_Day_Structured_Outputs_and_Schema_Validation/schema_validator_stdlib.py"
    )
    raise SystemExit(0)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class ExtractedItem(BaseModel):
    """Single item mentioned in a customer support ticket."""

    model_config = ConfigDict(extra="forbid")

    sku: str = Field(..., min_length=3, description="Product SKU, e.g. SKU-404")
    quantity: int = Field(..., ge=1, le=100, description="Affected unit count")


class SupportTicketExtraction(BaseModel):
    """Strict output contract for support ticket triage."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(..., pattern=r"^ORD-\d{4,}$", description="Order ID like ORD-1042")
    category: Literal["damaged_item", "late_delivery", "billing", "other"]
    priority: Literal["low", "medium", "high"]
    refund_requested: bool
    summary: str = Field(..., min_length=10, max_length=200)
    items: list[ExtractedItem] = Field(
        ..., description="Affected items mentioned in the ticket, or an empty list if none."
    )


def build_openai_response_format(model_cls: type[BaseModel], schema_name: str) -> dict[str, Any]:
    """Wrap a Pydantic JSON Schema in the OpenAI `response_format` structure."""
    schema = model_cls.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def validate_ticket_json(raw_json: str) -> tuple[SupportTicketExtraction | None, list[dict[str, Any]]]:
    """Validate raw JSON text against SupportTicketExtraction.

    Returns (validated_model, errors). Never raises on bad model output.
    """
    try:
        parsed = SupportTicketExtraction.model_validate_json(raw_json)
        return parsed, []
    except ValidationError as exc:
        compact_errors = [
            {
                "loc": ".".join(str(part) for part in err["loc"]),
                "type": err["type"],
                "msg": err["msg"],
            }
            for err in exc.errors()
        ]
        return None, compact_errors


def is_local_endpoint(base_url: str) -> bool:
    lowered = base_url.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered


def should_use_demo(force_demo: bool) -> bool:
    env_demo = os.getenv("AGENT_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    missing_key = (not api_key) or api_key == "your-api-key-here"
    return force_demo or env_demo or (missing_key and not is_local_endpoint(base_url))


def call_live_structured(ticket_text: str) -> str:
    """Request structured JSON matching SupportTicketExtraction from a live endpoint."""
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or "ollama"

    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract structured ticket details as JSON strictly matching the schema. "
                    "Do not invent fields."
                ),
            },
            {"role": "user", "content": ticket_text},
        ],
        "response_format": build_openai_response_format(
            SupportTicketExtraction, "support_ticket_extraction"
        ),
    }

    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Structured API call failed: {exc}") from exc
    return str(data["choices"][0]["message"].get("content", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Pydantic v2 structured output demo.")
    parser.add_argument("--demo", action="store_true", help="Use offline mock outputs.")
    args = parser.parse_args()

    print("=== Generated JSON Schema (for OpenAI response_format) ===")
    response_format = build_openai_response_format(
        SupportTicketExtraction, "support_ticket_extraction"
    )
    print(json.dumps(response_format["json_schema"]["schema"], indent=2))
    print()

    sample_ticket = (
        "Hi support, order ORD-4490 arrived today with 2 units of SKU-908 cracked. "
        "Please refund these items right away—we need replacements for an event Friday."
    )

    if should_use_demo(args.demo):
        print("=== Mode: DEMO (offline mock outputs) ===\n")
        raw_valid = json.dumps(
            {
                "order_id": "ORD-4490",
                "category": "damaged_item",
                "priority": "high",
                "refund_requested": True,
                "summary": "Two units of SKU-908 from ORD-4490 arrived cracked; refund requested.",
                "items": [{"sku": "SKU-908", "quantity": 2}],
            }
        )
    else:
        print("=== Mode: LIVE API ===\n")
        try:
            raw_valid = call_live_structured(sample_ticket)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    print("1) Validating model output:")
    ticket, errors = validate_ticket_json(raw_valid)
    if ticket is not None:
        print("   VALID:", ticket.model_dump())
    else:
        print("   INVALID:", errors)
    print()

    # Demonstrate catching bad outputs even if a provider ignores response_format.
    malformed_samples = [
        (
            "Wrong enum + bad order_id + extra key",
            json.dumps(
                {
                    "order_id": "4490",
                    "category": "broken_box",
                    "priority": "urgent",
                    "refund_requested": True,
                    "summary": "Broken item",
                    "unexpected_field": "should fail",
                }
            ),
        ),
        (
            "Truncated / invalid JSON",
            '{"order_id": "ORD-4490", "category": "damaged_item", "priority": ',
        ),
    ]

    print("2) Rejecting malformed outputs:")
    for label, payload in malformed_samples:
        _, errs = validate_ticket_json(payload)
        print(f"   Case [{label}]:")
        for err in errs:
            print(f"     - {err['loc'] or '<root>'}: {err['msg']} ({err['type']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
