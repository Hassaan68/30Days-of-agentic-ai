"""Day 7: Short-Term Memory and Conversation State Management.

Implements a typed ConversationState container with:
1. Pair-safe atomic grouping so assistant tool_calls and tool results are never split.
2. Token-budgeted sliding window with rolling summary compaction.
3. Structured working memory slots separate from raw chat history.
4. JSON checkpoint serialization and restore.

Run offline demo (no API key needed):
    python 07_Day_Short_Term_Memory_and_Conversation_State/conversation_state.py --demo

Run with live LLM rolling summarization (OpenAI-compatible endpoint):
    export OPENAI_API_KEY="your-key"
    python 07_Day_Short_Term_Memory_and_Conversation_State/conversation_state.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Typed message & state data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Represents a single function call requested by the assistant."""

    id: str
    name: str
    arguments: str  # JSON-encoded string matching OpenAI wire format

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(id=data["id"], name=data["name"], arguments=data["arguments"])


@dataclass
class Message:
    """Typed conversation message compatible with OpenAI chat completions."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        elif self.role == "assistant" and self.tool_calls:
            payload["content"] = None
        else:
            payload["content"] = ""
        if self.tool_calls:
            payload["tool_calls"] = [tc.to_openai_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None and self.role == "tool":
            payload["name"] = self.name
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class TokenBudget:
    """Controls how many tokens the prompt may use before compaction runs."""

    max_context_tokens: int = 650
    reserve_for_response: int = 150
    max_summary_tokens: int = 160

    @property
    def max_prompt_tokens(self) -> int:
        return max(100, self.max_context_tokens - self.reserve_for_response)


# ---------------------------------------------------------------------------
# Token estimation, atomic grouping, and tool-pair validation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Fast character-based token estimate (~3.5 chars/token, conservative for JSON/code)."""
    if not text:
        return 0
    return math.ceil(len(text) / 3.5)


def estimate_message_tokens(msg: Message) -> int:
    """Estimate token cost of one Message including role/formatting overhead."""
    tokens = 4  # per-message framing overhead
    if msg.content:
        tokens += estimate_tokens(msg.content)
    for tc in msg.tool_calls:
        tokens += 8 + estimate_tokens(tc.name) + estimate_tokens(tc.arguments)
    if msg.tool_call_id:
        tokens += estimate_tokens(msg.tool_call_id)
    return tokens


def group_atomic_blocks(messages: list[Message]) -> list[list[Message]]:
    """Group messages into indivisible blocks so tool calls & results stay together.

    Rules:
    - An assistant message containing tool_calls is bundled with all immediately
      following tool messages that answer those tool_call_ids.
    - Regular user or assistant messages form single-message blocks.
    """
    blocks: list[list[Message]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if msg.role == "assistant" and msg.tool_calls:
            expected_ids = {tc.id for tc in msg.tool_calls}
            block = [msg]
            i += 1
            while i < n and messages[i].role == "tool" and messages[i].tool_call_id in expected_ids:
                block.append(messages[i])
                i += 1
            blocks.append(block)
        else:
            blocks.append([msg])
            i += 1
    return blocks


def validate_tool_pairs(messages: list[Message]) -> list[str]:
    """Verify every tool_call has matching tool results and no orphan tool messages exist."""
    errors: list[str] = []
    pending_ids: set[str] = set()

    for idx, msg in enumerate(messages):
        if msg.role == "assistant" and msg.tool_calls:
            if pending_ids:
                errors.append(
                    f"Index {idx}: new assistant tool_calls started before resolving {sorted(pending_ids)}."
                )
            pending_ids = {tc.id for tc in msg.tool_calls}
        elif msg.role == "tool":
            if not msg.tool_call_id:
                errors.append(f"Index {idx}: tool message missing tool_call_id.")
            elif msg.tool_call_id not in pending_ids:
                errors.append(
                    f"Index {idx}: orphan tool message for tool_call_id='{msg.tool_call_id}' "
                    "with no preceding matching assistant tool_call."
                )
            else:
                pending_ids.remove(msg.tool_call_id)
        else:
            if pending_ids:
                errors.append(
                    f"Index {idx} ({msg.role}): unresolved tool_call_ids before next turn: "
                    f"{sorted(pending_ids)}."
                )
                pending_ids.clear()

    if pending_ids:
        errors.append(f"End of history: unresolved tool_call_ids: {sorted(pending_ids)}.")

    return errors


# ---------------------------------------------------------------------------
# Rolling summary compactors (offline deterministic + live OpenAI-compatible)
# ---------------------------------------------------------------------------

class SummaryCompactor(Protocol):
    """Interface for merging evicted messages into a compact rolling summary."""

    def compact(
        self,
        existing_summary: str,
        evicted_messages: list[Message],
        max_tokens: int,
    ) -> str:
        ...


class DeterministicCompactor:
    """Offline rule-based compactor that preserves IDs, facts, and tool outcomes."""

    _ID_PATTERN = re.compile(r"\b(?:ACC|INC|ORD|PR|DEPLOY|SVC|TICKET)-[A-Z0-9-]+\b|\b[a-z]{2}-[a-z]+-\d\b")

    def compact(
        self,
        existing_summary: str,
        evicted_messages: list[Message],
        max_tokens: int,
    ) -> str:
        bullets: list[str] = []
        if existing_summary.strip():
            for line in existing_summary.splitlines():
                cleaned = line.strip()
                if cleaned:
                    bullets.append(cleaned)

        for msg in evicted_messages:
            if msg.role == "user" and msg.content:
                snippet = self._shorten(msg.content, 110)
                bullets.append(f"- User asked: {snippet}")
            elif msg.role == "tool" and msg.content:
                tool_label = msg.name or "tool"
                snippet = self._shorten(msg.content, 110)
                bullets.append(f"- Tool {tool_label}: {snippet}")
            elif msg.role == "assistant" and msg.content and not msg.tool_calls:
                # Keep assistant conclusions that mention concrete IDs/decisions.
                if self._ID_PATTERN.search(msg.content):
                    snippet = self._shorten(msg.content, 110)
                    bullets.append(f"- Assistant noted: {snippet}")

        # Deduplicate while preserving order.
        deduped: list[str] = []
        seen: set[str] = set()
        for item in bullets:
            if item not in seen:
                seen.add(item)
                deduped.append(item)

        # Enforce max_summary_tokens while keeping lines containing important IDs.
        while len(deduped) > 1 and estimate_tokens("\n".join(deduped)) > max_tokens:
            # Prefer dropping the oldest bullet that does NOT contain a structured ID.
            drop_idx = next(
                (i for i, line in enumerate(deduped) if not self._ID_PATTERN.search(line)),
                0,
            )
            deduped.pop(drop_idx)

        return "\n".join(deduped)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        compact_ws = " ".join(text.split())
        if len(compact_ws) <= limit:
            return compact_ws
        return compact_ws[: limit - 3].rstrip() + "..."


class OpenAISummaryCompactor:
    """Calls an OpenAI-compatible chat model to roll evicted turns into a summary."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def compact(
        self,
        existing_summary: str,
        evicted_messages: list[Message],
        max_tokens: int,
    ) -> str:
        transcript_lines = []
        for msg in evicted_messages:
            if msg.role == "assistant" and msg.tool_calls:
                calls = ", ".join(f"{tc.name}({tc.arguments})" for tc in msg.tool_calls)
                transcript_lines.append(f"ASSISTANT_TOOL_CALLS: {calls}")
            elif msg.role == "tool":
                transcript_lines.append(f"TOOL[{msg.name or msg.tool_call_id}]: {msg.content}")
            else:
                transcript_lines.append(f"{msg.role.upper()}: {msg.content}")

        prompt = (
            "Update the running conversation summary using the evicted messages below. "
            "Keep concrete IDs (accounts, tickets, regions, error codes), decisions, and unresolved tasks. "
            f"Keep the summary under {max_tokens} tokens as concise bullet points.\n\n"
            f"Existing summary:\n{existing_summary or '(none)'}\n\n"
            f"Evicted messages:\n" + "\n".join(transcript_lines)
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You produce concise, factual conversation summaries."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as exc:
            # Fall back to deterministic compaction rather than crashing a live session.
            fallback = DeterministicCompactor()
            return fallback.compact(existing_summary, evicted_messages, max_tokens)


# ---------------------------------------------------------------------------
# Typed ConversationState manager
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """Manages short-term message window, rolling summary, and structured slots."""

    session_id: str
    system_prompt: str
    budget: TokenBudget = field(default_factory=TokenBudget)
    running_summary: str = ""
    working_memory: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    compaction_count: int = 0
    evicted_message_count: int = 0

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(
        self,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self.messages.append(
            Message(role="assistant", content=content, tool_calls=list(tool_calls or []))
        )

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append(
            Message(role="tool", tool_call_id=tool_call_id, name=name, content=content)
        )

    def update_working_memory(self, **slots: Any) -> None:
        """Persist explicit structured facts outside the rolling text window."""
        self.working_memory.update(slots)

    def _build_system_message(self) -> Message:
        sections = [self.system_prompt.strip()]
        if self.working_memory:
            slots_json = json.dumps(self.working_memory, sort_keys=True)
            sections.append(f"Structured session state:\n{slots_json}")
        if self.running_summary.strip():
            sections.append(f"Summary of earlier conversation:\n{self.running_summary.strip()}")
        return Message(role="system", content="\n\n".join(sections))

    def build_prompt_messages(self) -> list[dict[str, Any]]:
        """Assemble the OpenAI-ready message list with pinned system + summary."""
        system_msg = self._build_system_message()
        return [system_msg.to_openai_dict(), *[m.to_openai_dict() for m in self.messages]]

    def estimated_prompt_tokens(self) -> int:
        system_msg = self._build_system_message()
        return estimate_message_tokens(system_msg) + sum(
            estimate_message_tokens(m) for m in self.messages
        )

    def compact_if_needed(
        self,
        compactor: SummaryCompactor,
        min_recent_blocks: int = 1,
    ) -> int:
        """Evict oldest atomic blocks into running_summary until within token budget.

        Evicts blocks in a batch before calling the compactor so live LLM
        summarization only runs once per compaction pass instead of once per message.
        Returns the number of messages evicted during this call.
        """
        if self.estimated_prompt_tokens() <= self.budget.max_prompt_tokens:
            return 0

        blocks = group_atomic_blocks(self.messages)
        total_evicted = 0
        floor_blocks = max(1, min_recent_blocks)

        while len(blocks) > floor_blocks and self.estimated_prompt_tokens() > self.budget.max_prompt_tokens:
            batch_evicted: list[Message] = []
            # Pop oldest atomic blocks until remaining prompt (plus summary headroom) fits budget.
            target_tokens = self.budget.max_prompt_tokens - min(
                40, self.budget.max_summary_tokens // 4
            )
            while len(blocks) > floor_blocks and (
                not batch_evicted or self.estimated_prompt_tokens() > target_tokens
            ):
                oldest_block = blocks.pop(0)
                batch_evicted.extend(oldest_block)
                self.messages = [m for block in blocks for m in block]

            self.running_summary = compactor.compact(
                self.running_summary,
                batch_evicted,
                self.budget.max_summary_tokens,
            )
            total_evicted += len(batch_evicted)

        if total_evicted:
            self.compaction_count += 1
            self.evicted_message_count += total_evicted

        violations = validate_tool_pairs(self.messages)
        if violations:
            raise RuntimeError(f"Compaction broke tool-pair integrity: {violations}")

        return total_evicted

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "budget": {
                "max_context_tokens": self.budget.max_context_tokens,
                "reserve_for_response": self.budget.reserve_for_response,
                "max_summary_tokens": self.budget.max_summary_tokens,
            },
            "running_summary": self.running_summary,
            "working_memory": self.working_memory,
            "messages": [m.to_dict() for m in self.messages],
            "compaction_count": self.compaction_count,
            "evicted_message_count": self.evicted_message_count,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationState:
        b = data["budget"]
        return cls(
            session_id=data["session_id"],
            system_prompt=data["system_prompt"],
            budget=TokenBudget(
                max_context_tokens=b["max_context_tokens"],
                reserve_for_response=b["reserve_for_response"],
                max_summary_tokens=b["max_summary_tokens"],
            ),
            running_summary=data.get("running_summary", ""),
            working_memory=dict(data.get("working_memory", {})),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            compaction_count=data.get("compaction_count", 0),
            evicted_message_count=data.get("evicted_message_count", 0),
        )

    @classmethod
    def from_json(cls, raw: str) -> ConversationState:
        return cls.from_dict(json.loads(raw))


# ---------------------------------------------------------------------------
# Runnable demonstration
# ---------------------------------------------------------------------------

def _should_use_demo(force_demo: bool) -> bool:
    if force_demo or os.getenv("AGENT_DEMO_MODE", "").strip() == "1":
        return True
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    return not api_key and not is_local


def run_demo(compactor: SummaryCompactor, mode_label: str) -> None:
    print(f"=== Day 7: Short-Term Memory & Conversation State ({mode_label}) ===\n")

    # 1. Show why naive slicing messages[-3:] breaks tool_call/tool pairs.
    sample_history = [
        Message(role="user", content="Check status for ACC-884 in eu-west-1."),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="call_a1", name="get_account", arguments='{"account_id":"ACC-884"}'),
                ToolCall(id="call_a2", name="get_region_health", arguments='{"region":"eu-west-1"}'),
            ],
        ),
        Message(role="tool", tool_call_id="call_a1", name="get_account", content='{"tier":"enterprise"}'),
        Message(role="tool", tool_call_id="call_a2", name="get_region_health", content='{"status":"degraded"}'),
        Message(role="assistant", content="ACC-884 is on enterprise tier; eu-west-1 is degraded."),
    ]
    naive_slice = sample_history[-3:]
    naive_errors = validate_tool_pairs(naive_slice)
    print("1) Why naive slicing (`messages[-3:]`) fails:")
    for err in naive_errors:
        print(f"   VIOLATION: {err}")

    # 2. Run a multi-turn session with pair-safe rolling compaction.
    print("\n2) Bounded ConversationState across a 6-turn incident investigation:")
    state = ConversationState(
        session_id="sess_ops_2026_01",
        system_prompt=(
            "You are an incident response assistant. Track customer IDs, regions, "
            "and remediation steps accurately."
        ),
        budget=TokenBudget(
            max_context_tokens=560,
            reserve_for_response=140,
            max_summary_tokens=150,
        ),
    )

    scripted_turns: list[tuple[str, list[ToolCall], list[tuple[str, str, str]], str, dict[str, Any]]] = [
        (
            "Enterprise customer ACC-884 reports 502 errors on checkout in eu-west-1.",
            [
                ToolCall("call_1a", "get_account", '{"account_id":"ACC-884"}'),
                ToolCall("call_1b", "check_region", '{"region":"eu-west-1"}'),
            ],
            [
                ("call_1a", "get_account", '{"account_id":"ACC-884","tier":"enterprise","sla":"99.99%"}'),
                ("call_1b", "check_region", '{"region":"eu-west-1","error_rate_pct":14.2,"status":"degraded"}'),
            ],
            "Confirmed ACC-884 is enterprise tier and eu-west-1 checkout error rate is 14.2%.",
            {"account_id": "ACC-884", "region": "eu-west-1", "tier": "enterprise"},
        ),
        (
            "Open an incident ticket and pull the latest deployment ID for checkout-api.",
            [
                ToolCall("call_2a", "create_incident", '{"account":"ACC-884","severity":"sev1"}'),
                ToolCall("call_2b", "latest_deploy", '{"service":"checkout-api","region":"eu-west-1"}'),
            ],
            [
                ("call_2a", "create_incident", '{"incident_id":"INC-409","severity":"sev1"}'),
                ("call_2b", "latest_deploy", '{"deploy_id":"DEPLOY-772","commit":"9f21ac","minutes_ago":18}'),
            ],
            "Created INC-409 (sev1). Latest rollout was DEPLOY-772 18 minutes ago.",
            {"active_incident": "INC-409", "suspect_deploy": "DEPLOY-772"},
        ),
        (
            "Inspect error logs for DEPLOY-772 to see what is crashing.",
            [ToolCall("call_3a", "query_logs", '{"deploy_id":"DEPLOY-772","level":"ERROR"}')],
            [
                (
                    "call_3a",
                    "query_logs",
                    '{"top_error":"ConnectionPoolExhausted on db-eu-west-1","count":4182}',
                )
            ],
            "Logs show 4,182 ConnectionPoolExhausted errors on db-eu-west-1 introduced by DEPLOY-772.",
            {},
        ),
        (
            "Roll back checkout-api in eu-west-1 to the previous stable build DEPLOY-771.",
            [ToolCall("call_4a", "rollback_service", '{"service":"checkout-api","target":"DEPLOY-771"}')],
            [("call_4a", "rollback_service", '{"status":"rolled_back","active_deploy":"DEPLOY-771"}')],
            "Rollback to DEPLOY-771 succeeded for checkout-api in eu-west-1.",
            {"active_deploy": "DEPLOY-771"},
        ),
        (
            "Re-check error rates and verify checkout latency recovered.",
            [ToolCall("call_5a", "check_region", '{"region":"eu-west-1"}')],
            [("call_5a", "check_region", '{"region":"eu-west-1","error_rate_pct":0.08,"p95_ms":142}')],
            "Error rate dropped from 14.2% to 0.08% with p95 latency at 142ms.",
            {"status": "mitigated"},
        ),
        (
            "Draft a quick post-incident status note referencing the account, incident, and rollback.",
            [],
            [],
            (
                "Incident INC-409 for ACC-884 in eu-west-1 is mitigated after rolling back "
                "DEPLOY-772 to DEPLOY-771; error rate is back to 0.08%."
            ),
            {},
        ),
    ]

    budget_limit = state.budget.max_prompt_tokens
    print(f"   Token budget ceiling (max_prompt_tokens): {budget_limit}\n")

    for turn_idx, (user_text, tool_calls, tool_results, reply, slot_updates) in enumerate(
        scripted_turns, start=1
    ):
        state.add_user_message(user_text)
        if tool_calls:
            state.add_assistant_message(content=None, tool_calls=tool_calls)
            for call_id, name, payload in tool_results:
                state.add_tool_result(tool_call_id=call_id, name=name, content=payload)
        state.add_assistant_message(content=reply)
        if slot_updates:
            state.update_working_memory(**slot_updates)

        evicted = state.compact_if_needed(compactor)
        tokens_now = state.estimated_prompt_tokens()
        pair_ok = len(validate_tool_pairs(state.messages)) == 0
        blocks_now = len(group_atomic_blocks(state.messages))
        print(
            f"   Turn {turn_idx}: msgs={len(state.messages):2d} | blocks={blocks_now} | "
            f"tokens={tokens_now:3d}/{budget_limit} | evicted={evicted:2d} | "
            f"compactions={state.compaction_count} | pairs_valid={pair_ok}"
        )

    print("\n3) Preserved structured working memory:")
    print("   " + json.dumps(state.working_memory, sort_keys=True))

    print("\n4) Rolling summary retained after evicting older turns:")
    for line in state.running_summary.splitlines():
        print(f"   {line}")

    # 5. Checkpoint serialization round-trip check.
    serialized = state.to_json()
    restored = ConversationState.from_json(serialized)
    assert restored.to_dict() == state.to_dict(), "Round-trip serialization mismatch!"
    assert validate_tool_pairs(restored.messages) == [], "Restored state has invalid tool pairs!"
    print(
        f"\n5) Checkpoint round-trip OK ({len(serialized)} bytes JSON, "
        f"{restored.evicted_message_count} total messages compacted)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 7: Short-term memory & conversation state")
    parser.add_argument("--demo", action="store_true", help="Use offline deterministic compactor")
    args = parser.parse_args()

    if _should_use_demo(args.demo):
        compactor: SummaryCompactor = DeterministicCompactor()
        run_demo(compactor, mode_label="offline deterministic compactor")
    else:
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        compactor = OpenAISummaryCompactor(api_key=api_key, base_url=base_url, model=model)
        run_demo(compactor, mode_label=f"live summarizer: {model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
