"""Day 3: Zero-dependency JSON extraction and strict schema validation.

Models frequently wrap JSON in markdown fences, add explanatory prose before or
after the object, return wrong types, omit required keys, or invent extra fields.
This script separates safe formatting cleanup (extracting the JSON object) from
strict schema validation (rejecting missing fields, wrong types, or extra keys).

Run:
    python3 03_Day_Structured_Outputs_and_Schema_Validation/schema_validator_stdlib.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# Strict JSON Schema subset for a support ticket extraction payload.
TICKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["order_id", "category", "priority", "refund_requested", "summary", "items"],
    "properties": {
        "order_id": {"type": "string", "pattern": r"^ORD-\d{4,}$"},
        "category": {
            "type": "string",
            "enum": ["damaged_item", "late_delivery", "billing", "other"],
        },
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "refund_requested": {"type": "boolean"},
        "summary": {"type": "string", "minLength": 10, "maxLength": 200},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sku", "quantity"],
                "properties": {
                    "sku": {"type": "string", "minLength": 3},
                    "quantity": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    data: dict[str, Any] | None
    repaired_formatting: bool
    errors: list[str]


def extract_json_object(raw_text: str) -> tuple[str | None, bool]:
    """Extract a single JSON object string from raw LLM output.

    Returns (json_str, was_repaired):
      - If raw_text is already a clean JSON object, was_repaired is False.
      - If markdown fences or surrounding prose were stripped, was_repaired is True.
      - If no balanced JSON object can be found, returns (None, False).
    """
    stripped = raw_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return stripped, False
        except json.JSONDecodeError:
            pass

    # Step 1: Check for fenced ```json ... ``` or ``` ... ``` blocks.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return candidate, True
        except json.JSONDecodeError:
            pass

    # Step 2: Scan for the first balanced {...} JSON object, respecting quoted strings.
    in_string = False
    escape = False
    depth = 0
    start_idx: int | None = None

    for idx, ch in enumerate(raw_text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidate = raw_text[start_idx : idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return candidate, True
                except json.JSONDecodeError:
                    start_idx = None

    return None, False


def _matches_type(value: Any, expected_type: str) -> bool:
    # Important Python quirk: `bool` is a subclass of `int`, so `isinstance(True, int)` is True!
    # Always guard boolean vs integer explicitly in schema validators.
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return False


def validate_against_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Recursively validate a decoded JSON value against a strict JSON Schema subset."""
    errors: list[str] = []

    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        errors.append(f"{path}: expected type '{expected_type}', got '{type(value).__name__}'")
        return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} not in allowed enum {schema['enum']}")

    if expected_type == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: string length {len(value)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: string length {len(value)} > maxLength {schema['maxLength']}")
        if "pattern" in schema and re.match(schema["pattern"], value) is None:
            errors.append(f"{path}: string {value!r} does not match pattern {schema['pattern']!r}")

    if expected_type in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value {value} > maximum {schema['maximum']}")

    if expected_type == "object":
        properties = schema.get("properties", {})
        for req_key in schema.get("required", []):
            if req_key not in value:
                errors.append(f"{path}.{req_key}: missing required field")

        if schema.get("additionalProperties") is False:
            extra_keys = sorted(set(value.keys()) - set(properties.keys()))
            for extra in extra_keys:
                errors.append(f"{path}.{extra}: unexpected extra field")

        for key, prop_schema in properties.items():
            if key in value:
                errors.extend(validate_against_schema(value[key], prop_schema, f"{path}.{key}"))

    if expected_type == "array" and "items" in schema:
        item_schema = schema["items"]
        for idx, item in enumerate(value):
            errors.extend(validate_against_schema(item, item_schema, f"{path}[{idx}]"))

    return errors


def parse_and_validate(raw_output: str, schema: dict[str, Any]) -> ValidationResult:
    """Full pipeline: extract JSON safely, decode, then enforce strict schema rules."""
    extracted, repaired = extract_json_object(raw_output)
    if extracted is None:
        return ValidationResult(
            ok=False,
            data=None,
            repaired_formatting=False,
            errors=["$: could not extract a valid JSON object from model output"],
        )

    decoded = json.loads(extracted)
    errors = validate_against_schema(decoded, schema)
    if errors:
        return ValidationResult(
            ok=False,
            data=None,
            repaired_formatting=repaired,
            errors=errors,
        )

    return ValidationResult(ok=True, data=decoded, repaired_formatting=repaired, errors=[])


def main() -> None:
    test_cases: list[tuple[str, str]] = [
        (
            "1. Clean valid JSON",
            json.dumps(
                {
                    "order_id": "ORD-1042",
                    "category": "late_delivery",
                    "priority": "medium",
                    "refund_requested": True,
                    "summary": "Express package ORD-1042 is five days late and customer asks for refund.",
                    "items": [{"sku": "SKU-110", "quantity": 1}],
                }
            ),
        ),
        (
            "2. Markdown fence + surrounding prose (safe to repair formatting)",
            (
                "Sure! Here is the extracted ticket in JSON format:\n"
                "```json\n"
                "{\n"
                '  "order_id": "ORD-4490",\n'
                '  "category": "damaged_item",\n'
                '  "priority": "high",\n'
                '  "refund_requested": true,\n'
                '  "summary": "Two units of SKU-908 arrived with cracked screens.",\n'
                '  "items": [{"sku": "SKU-908", "quantity": 2}]\n'
                "}\n"
                "```\n"
                "Let me know if you need anything else!"
            ),
        ),
        (
            "3. Missing required field + unexpected extra field (must reject)",
            json.dumps(
                {
                    "order_id": "ORD-5501",
                    "category": "billing",
                    "refund_requested": False,
                    "summary": "Customer asked for a PDF copy of last month invoice.",
                    "items": [],
                    "approved_refund_usd": 500,
                }
            ),
        ),
        (
            "4. Wrong types (bool passed for integer, string passed for boolean)",
            json.dumps(
                {
                    "order_id": "ORD-7710",
                    "category": "damaged_item",
                    "priority": "high",
                    "refund_requested": "yes",
                    "summary": "Box arrived crushed on the right corner.",
                    "items": [{"sku": "SKU-301", "quantity": True}],
                }
            ),
        ),
        (
            "5. Truncated output cut off by max_tokens (must reject)",
            '{"order_id": "ORD-8820", "category": "other", "priority": "lo',
        ),
    ]

    print("=== Zero-Dependency Strict Schema Validation Demo ===\n")
    for name, raw in test_cases:
        result = parse_and_validate(raw, TICKET_SCHEMA)
        status = "PASS" if result.ok else "REJECT"
        repair_note = " (formatting repaired)" if result.repaired_formatting else ""
        print(f"{name}")
        print(f"  Status: {status}{repair_note}")
        if result.ok and result.data is not None:
            print(f"  Order : {result.data['order_id']} | Category: {result.data['category']}")
        else:
            for err in result.errors:
                print(f"  Error : {err}")
        print()


if __name__ == "__main__":
    main()
