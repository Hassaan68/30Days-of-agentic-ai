"""Input and output guardrails for prompt injection defense and secret redaction."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailOutcome:
    """Result of an input or output guardrail check."""

    allowed: bool
    sanitized_text: str
    reason: str
    triggered_rules: tuple[str, ...] = ()


class InputGuardrail:
    """Rejects obvious prompt injection overrides and oversized payloads."""

    INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "ignore_instructions",
            re.compile(
                r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|policies)",
                re.IGNORECASE,
            ),
        ),
        (
            "system_override",
            re.compile(r"(\[?\s*system\s+override\s*\]?|bypass\s+approval)", re.IGNORECASE),
        ),
        (
            "role_hijack",
            re.compile(r"you\s+are\s+now\s+(root|unrestricted|dan)", re.IGNORECASE),
        ),
        (
            "secret_exfiltration",
            re.compile(r"(print|dump|reveal)\s+(all\s+)?(env|api\s*keys|secrets|system\s+prompt)", re.IGNORECASE),
        ),
    )

    def __init__(self, max_chars: int = 4000) -> None:
        self.max_chars = max_chars

    def check(self, text: str) -> GuardrailOutcome:
        cleaned = text.strip()
        if not cleaned:
            return GuardrailOutcome(False, "", "Empty user request", ("empty_input",))
        if len(cleaned) > self.max_chars:
            return GuardrailOutcome(
                False,
                cleaned[: self.max_chars],
                f"Input exceeds max length ({len(cleaned)} > {self.max_chars})",
                ("max_length",),
            )

        hits = [
            rule_name
            for rule_name, pattern in self.INJECTION_PATTERNS
            if pattern.search(cleaned)
        ]
        if hits:
            return GuardrailOutcome(
                False,
                cleaned,
                f"Blocked potential prompt injection ({', '.join(hits)})",
                tuple(hits),
            )
        return GuardrailOutcome(True, cleaned, "input clean")


class OutputGuardrail:
    """Scans outgoing agent responses and redacts leaked credentials or keys."""

    SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
        ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE)),
        (
            "private_key_block",
            re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
        ),
    )

    def check_and_sanitize(self, text: str) -> GuardrailOutcome:
        sanitized = text
        triggered: list[str] = []

        for rule_name, pattern in self.SECRET_PATTERNS:
            if pattern.search(sanitized):
                triggered.append(rule_name)
                sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)

        if triggered:
            return GuardrailOutcome(
                True,
                sanitized,
                f"Redacted sensitive tokens ({', '.join(triggered)})",
                tuple(triggered),
            )
        return GuardrailOutcome(True, sanitized, "output clean")
