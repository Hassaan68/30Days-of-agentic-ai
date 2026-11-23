"""Tamper-evident SHA-256 hash-chained audit receipt ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import AuditReceipt, canonical_json

GENESIS_HASH = "0" * 64


def _compute_receipt_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only hash-chained audit log recording every agent decision."""

    def __init__(self, jsonl_path: Path | None = None) -> None:
        self.jsonl_path = jsonl_path
        self.receipts: list[AuditReceipt] = []

    def record(
        self,
        *,
        run_id: str,
        event_type: str,
        actor: str,
        outcome: str,
        tool_name: str | None = None,
        arguments_hash: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditReceipt:
        seq = len(self.receipts) + 1
        prev_hash = self.receipts[-1].receipt_hash if self.receipts else GENESIS_HASH
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "seq": seq,
            "timestamp": ts,
            "run_id": run_id,
            "event_type": event_type,
            "tool_name": tool_name,
            "arguments_hash": arguments_hash,
            "actor": actor,
            "outcome": outcome,
            "details": details or {},
            "prev_hash": prev_hash,
        }
        receipt_hash = _compute_receipt_hash(payload)
        receipt = AuditReceipt(**payload, receipt_hash=receipt_hash)
        self.receipts.append(receipt)

        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(receipt), sort_keys=True) + "\n")

        return receipt

    def verify_chain(self) -> tuple[bool, str]:
        """Recompute every receipt hash and previous-link pointer from genesis."""
        expected_prev = GENESIS_HASH
        for idx, receipt in enumerate(self.receipts, start=1):
            if receipt.seq != idx:
                return False, f"Sequence gap at index {idx}: found seq={receipt.seq}"
            if receipt.prev_hash != expected_prev:
                return (
                    False,
                    f"Broken chain link at seq={receipt.seq}: "
                    f"expected prev_hash={expected_prev[:12]}, got {receipt.prev_hash[:12]}",
                )
            body = asdict(receipt)
            stored_hash = body.pop("receipt_hash")
            recomputed = _compute_receipt_hash(body)
            if stored_hash != recomputed:
                return (
                    False,
                    f"Tampered receipt detected at seq={receipt.seq} ({receipt.event_type}): "
                    f"stored={stored_hash[:12]} != recomputed={recomputed[:12]}",
                )
            expected_prev = stored_hash

        return True, f"Verified {len(self.receipts)} hash-chained audit receipt(s) from genesis."
