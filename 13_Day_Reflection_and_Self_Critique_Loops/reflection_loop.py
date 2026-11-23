"""Day 13: Reflection and Self-Critique Loop (Generator + Deterministic Checks + Rubric Critic).

Demonstrates:
- Separating draft generation from structured critique.
- Running fast, objective deterministic checks (required sections, fact preservation,
  secret/internal-leak guard, JSON syntax validation, length bounds) before LLM critique.
- Evaluating drafts against a structured rubric (scores + typed blocking/minor defects).
- Enforcing clear stopping criteria (pass threshold + zero blocking defects + max rounds +
  best-draft retention) to avoid endless nitpicking or regression.
- Supporting both live OpenAI-compatible APIs and an offline `--demo` mode.

Usage:
    python3 13_Day_Reflection_and_Self_Critique_Loops/reflection_loop.py --demo
    python3 13_Day_Reflection_and_Self_Critique_Loops/reflection_loop.py --max-rounds 4 --min-score 4.3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Source specification & typed reflection structures
# ---------------------------------------------------------------------------

RAW_ENGINEERING_NOTES = """
Internal Ticket: JIRA-4821 (do not share externally)
Internal staging host: webhooks-v2.internal.corp (do not share externally)

Facts for customer advisory:
- Legacy Webhook API v1 (HMAC-SHA1 via X-Hub-Signature) is being retired on 2026-09-15.
- Customers must migrate to Webhook API v2 (api_version: "2026-09").
- Webhook API v2 signs requests using HMAC-SHA256 in the header `X-Webhook-Signature-256`.
- Customers should verify the signature against the raw request body before parsing JSON,
  test in the sandbox dashboard, and switch their endpoint version before 2026-09-15.
- After 2026-09-15, v1 deliveries will stop and return HTTP 410 Gone on legacy test hooks.
"""


@dataclass(frozen=True)
class DeterministicCheck:
    """Result of one objective, rule-based validation check."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Defect:
    """Concrete, actionable issue identified during critique."""

    severity: Literal["blocking", "minor"]
    area: str
    issue: str
    fix: str


@dataclass(frozen=True)
class RubricScore:
    """Numeric score (1-5) for one rubric criterion."""

    criterion: str
    score: float
    note: str


@dataclass
class CritiqueReport:
    """Combined deterministic + rubric evaluation for a single draft."""

    deterministic_checks: list[DeterministicCheck]
    rubric_scores: list[RubricScore]
    defects: list[Defect]

    @property
    def deterministic_passed(self) -> bool:
        return all(check.passed for check in self.deterministic_checks)

    @property
    def blocking_defects(self) -> list[Defect]:
        return [d for d in self.defects if d.severity == "blocking"]

    @property
    def average_rubric_score(self) -> float:
        if not self.rubric_scores:
            return 0.0
        return round(sum(r.score for r in self.rubric_scores) / len(self.rubric_scores), 2)

    def overall_rank(self) -> tuple[int, int, float]:
        """Higher tuple is better: deterministic pass, negative blocking count, rubric avg."""
        return (
            1 if self.deterministic_passed else 0,
            -len(self.blocking_defects),
            self.average_rubric_score,
        )

    def meets_acceptance(self, min_score: float) -> bool:
        return (
            self.deterministic_passed
            and len(self.blocking_defects) == 0
            and self.average_rubric_score >= min_score
        )


@dataclass
class RevisionRound:
    """Snapshot of one generate -> critique iteration."""

    round_number: int
    draft: str
    critique: CritiqueReport


@dataclass
class ReflectionResult:
    """Final output plus complete revision history and stop reason."""

    best_round: int
    final_draft: str
    accepted: bool
    stop_reason: str
    history: list[RevisionRound] = field(default_factory=list)
    mode: str = "demo"


# ---------------------------------------------------------------------------
# Deterministic checks (run before LLM critic: fast, free, exact)
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def run_deterministic_checks(draft: str) -> list[DeterministicCheck]:
    """Check objective format, fact, security, JSON syntax, and length constraints."""
    checks: list[DeterministicCheck] = []

    # 1. Required headings
    required_headings = ["## Summary", "## Required Action", "## Example Payload", "## Timeline"]
    missing_headings = [h for h in required_headings if h not in draft]
    checks.append(
        DeterministicCheck(
            name="required_sections",
            passed=not missing_headings,
            detail="All 4 required sections present."
            if not missing_headings
            else f"Missing section(s): {', '.join(missing_headings)}",
        )
    )

    # 2. Required factual tokens
    required_tokens = ["2026-09-15", "X-Webhook-Signature-256", "HMAC-SHA256"]
    missing_tokens = [tok for tok in required_tokens if tok not in draft]
    checks.append(
        DeterministicCheck(
            name="required_facts",
            passed=not missing_tokens,
            detail="Exact cutoff date and signature header preserved."
            if not missing_tokens
            else f"Missing required fact(s): {', '.join(missing_tokens)}",
        )
    )

    # 3. No leaked internal identifiers
    forbidden_patterns = ["JIRA-", "internal.corp", "webhooks-v2.internal"]
    leaked = [pat for pat in forbidden_patterns if pat.lower() in draft.lower()]
    checks.append(
        DeterministicCheck(
            name="no_internal_leaks",
            passed=not leaked,
            detail="No internal hostnames or ticket IDs leaked."
            if not leaked
            else f"Leaked internal identifier(s): {', '.join(leaked)}",
        )
    )

    # 4. Valid JSON code block containing api_version == "2026-09"
    json_match = _JSON_BLOCK_RE.search(draft)
    if not json_match:
        checks.append(
            DeterministicCheck(
                name="valid_json_example",
                passed=False,
                detail="Missing fenced ```json ... ``` example payload block.",
            )
        )
    else:
        try:
            payload = json.loads(json_match.group(1))
            if isinstance(payload, dict) and payload.get("api_version") == "2026-09":
                checks.append(
                    DeterministicCheck(
                        name="valid_json_example",
                        passed=True,
                        detail='Valid JSON block with "api_version": "2026-09".',
                    )
                )
            else:
                checks.append(
                    DeterministicCheck(
                        name="valid_json_example",
                        passed=False,
                        detail='JSON parsed, but missing top-level "api_version": "2026-09".',
                    )
                )
        except json.JSONDecodeError as exc:
            checks.append(
                DeterministicCheck(
                    name="valid_json_example",
                    passed=False,
                    detail=f"Invalid JSON syntax in example block: {exc.msg} (line {exc.lineno}).",
                )
            )

    # 5. Concise length bounds (80..260 words)
    word_count = len(re.findall(r"\b\w+\b", draft))
    in_bounds = 80 <= word_count <= 260
    checks.append(
        DeterministicCheck(
            name="word_count_bounds",
            passed=in_bounds,
            detail=f"Word count is {word_count} (allowed: 80-260).",
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Generator & Critic backends (Live OpenAI-compatible HTTP + Deterministic Demo)
# ---------------------------------------------------------------------------


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Expected JSON object in critic response, got: {raw_text[:200]}")
    return json.loads(cleaned[start : end + 1])


class OpenAIReflectionBackend:
    """Uses an OpenAI-compatible API for separate Generator and Critic roles."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 30) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach LLM endpoint {url}: {exc}") from exc
        return str(data["choices"][0]["message"]["content"]).strip()

    def generate(
        self,
        source_notes: str,
        previous_draft: str | None,
        previous_critique: CritiqueReport | None,
        round_number: int,
    ) -> str:
        system = (
            "You are a technical writer drafting a customer-facing API migration advisory.\n"
            "Requirements:\n"
            "- Include exact headings: ## Summary, ## Required Action, ## Example Payload, ## Timeline.\n"
            "- Keep total length between 80 and 260 words.\n"
            "- Include a valid fenced ```json code block with \"api_version\": \"2026-09\".\n"
            "- Include exact facts: cutoff 2026-09-15, HMAC-SHA256, header X-Webhook-Signature-256.\n"
            "- Never leak internal tickets or internal hostnames."
        )
        if previous_draft is None or previous_critique is None:
            user = f"Write the customer migration advisory from these notes:\n\n{source_notes}"
        else:
            failed_checks = [
                f"- [CHECK FAILED] {c.name}: {c.detail}"
                for c in previous_critique.deterministic_checks
                if not c.passed
            ]
            defects = [
                f"- [{d.severity.upper()}] ({d.area}) {d.issue} -> Fix: {d.fix}"
                for d in previous_critique.defects
            ]
            feedback_block = "\n".join(failed_checks + defects) or "- Polish clarity while keeping all checks green."
            user = (
                f"Source notes:\n{source_notes}\n\n"
                f"Previous draft (Round {round_number - 1}):\n{previous_draft}\n\n"
                f"Fix ALL of the following issues without introducing new regressions:\n{feedback_block}"
            )
        return self._chat(system, user, temperature=0.2)

    def critique(
        self,
        source_notes: str,
        draft: str,
        deterministic_checks: list[DeterministicCheck],
        round_number: int,
    ) -> CritiqueReport:
        system = (
            "You are a strict documentation reviewer. Grade the draft against the source notes and "
            "deterministic check results. Return ONLY valid JSON with:\n"
            '{"rubric_scores": [{"criterion": "accuracy"|"actionability"|"clarity_and_tone", "score": 1-5, "note": "..."}], '
            '"defects": [{"severity": "blocking"|"minor", "area": "...", "issue": "...", "fix": "..."}]}\n'
            "Do not invent nitpicks if the draft is accurate, complete, and clear."
        )
        det_summary = [
            {"name": c.name, "passed": c.passed, "detail": c.detail} for c in deterministic_checks
        ]
        user = json.dumps(
            {
                "round": round_number,
                "source_notes": source_notes,
                "deterministic_checks": det_summary,
                "draft": draft,
            },
            indent=2,
        )
        raw = self._chat(system, user, temperature=0.0)
        parsed = _extract_json_object(raw)
        scores = [
            RubricScore(
                criterion=str(item["criterion"]),
                score=float(item["score"]),
                note=str(item["note"]),
            )
            for item in parsed.get("rubric_scores", [])
        ]
        defects = [
            Defect(
                severity="blocking" if item.get("severity") == "blocking" else "minor",
                area=str(item.get("area", "general")),
                issue=str(item.get("issue", "")),
                fix=str(item.get("fix", "")),
            )
            for item in parsed.get("defects", [])
        ]
        return CritiqueReport(
            deterministic_checks=deterministic_checks,
            rubric_scores=scores,
            defects=defects,
        )


class DemoReflectionBackend:
    """Deterministic local generator + critic illustrating 3 rounds of targeted improvement."""

    def generate(
        self,
        source_notes: str,
        previous_draft: str | None,
        previous_critique: CritiqueReport | None,
        round_number: int,
    ) -> str:
        if round_number == 1:
            # Round 1: realistic flawed first draft (leaks internal info, bad JSON comma, missing Timeline)
            return (
                "## Summary\n"
                "As tracked in JIRA-4821, we are retiring Webhook API v1 on 2026-09-15. "
                "Please switch over to Webhook API v2 and test against webhooks-v2.internal.corp soon.\n\n"
                "## Required Action\n"
                "Update your endpoint to handle HMAC-SHA256 signatures and verify incoming events.\n\n"
                "## Example Payload\n"
                "```json\n"
                "{\n"
                '  "event_id": "evt_01J9X",\n'
                '  "api_version": "2026-09",\n'
                '  "type": "invoice.paid",\n'
                "}\n"
                "```"
            )

        if round_number == 2:
            # Round 2: fixes leaks, JSON syntax, and adds Timeline — but still omits exact header name
            return (
                "## Summary\n"
                "Legacy Webhook API v1 (HMAC-SHA1) is retiring on 2026-09-15. "
                "All integrations must move to Webhook API v2 before that date.\n\n"
                "## Required Action\n"
                "1. Update your webhook handler to verify HMAC-SHA256 signatures on the raw request body before parsing JSON.\n"
                "2. Validate delivery in the sandbox dashboard.\n"
                "3. Switch your production endpoint to API v2.\n\n"
                "## Example Payload\n"
                "```json\n"
                "{\n"
                '  "event_id": "evt_01J9X",\n'
                '  "api_version": "2026-09",\n'
                '  "type": "invoice.paid"\n'
                "}\n"
                "```\n\n"
                "## Timeline\n"
                "- **Now:** Webhook API v2 is available in sandbox and production.\n"
                "- **2026-09-15:** Webhook API v1 deliveries stop and legacy test hooks return HTTP 410 Gone."
            )

        # Round 3+: clean, accurate, customer-ready draft that passes all deterministic + rubric gates
        return (
            "## Summary\n"
            "Legacy Webhook API v1 (HMAC-SHA1 via `X-Hub-Signature`) is retiring on **2026-09-15**. "
            "Migrate your integration to Webhook API v2 (`2026-09`) to avoid interrupted event delivery.\n\n"
            "## Required Action\n"
            "1. Read the HMAC-SHA256 digest from the `X-Webhook-Signature-256` request header.\n"
            "2. Verify the signature against the **raw request body bytes** before parsing JSON.\n"
            "3. Send test events in the sandbox dashboard and confirm HTTP 2xx responses.\n"
            "4. Switch your production endpoint setting to Webhook API v2 before 2026-09-15.\n\n"
            "## Example Payload\n"
            "```json\n"
            "{\n"
            '  "event_id": "evt_01J9X",\n'
            '  "api_version": "2026-09",\n'
            '  "type": "invoice.paid"\n'
            "}\n"
            "```\n\n"
            "## Timeline\n"
            "- **Now:** Webhook API v2 is live in sandbox and production.\n"
            "- **2026-09-15:** Webhook API v1 deliveries end permanently, and legacy test hooks return HTTP 410 Gone."
        )

    def critique(
        self,
        source_notes: str,
        draft: str,
        deterministic_checks: list[DeterministicCheck],
        round_number: int,
    ) -> CritiqueReport:
        failed_map = {c.name: c.detail for c in deterministic_checks if not c.passed}
        defects: list[Defect] = []

        if "no_internal_leaks" in failed_map:
            defects.append(
                Defect(
                    severity="blocking",
                    area="security",
                    issue=failed_map["no_internal_leaks"],
                    fix="Remove internal ticket numbers and internal corporate hostnames completely.",
                )
            )
        if "valid_json_example" in failed_map:
            defects.append(
                Defect(
                    severity="blocking",
                    area="example_payload",
                    issue=failed_map["valid_json_example"],
                    fix="Fix JSON formatting (remove trailing comma) and keep api_version set to 2026-09.",
                )
            )
        if "required_sections" in failed_map:
            defects.append(
                Defect(
                    severity="blocking",
                    area="structure",
                    issue=failed_map["required_sections"],
                    fix="Add all required markdown sections including ## Timeline.",
                )
            )
        if "required_facts" in failed_map:
            defects.append(
                Defect(
                    severity="blocking",
                    area="accuracy",
                    issue=failed_map["required_facts"],
                    fix="State the exact header name `X-Webhook-Signature-256`, `HMAC-SHA256`, and `2026-09-15`.",
                )
            )
        if "word_count_bounds" in failed_map:
            defects.append(
                Defect(
                    severity="minor",
                    area="length",
                    issue=failed_map["word_count_bounds"],
                    fix="Adjust detail so the advisory stays within 80-260 words.",
                )
            )

        if not defects:
            scores = [
                RubricScore("accuracy", 5.0, "Matches all source facts with exact cutoff date and header."),
                RubricScore("actionability", 4.6, "Clear numbered migration checklist plus valid JSON example."),
                RubricScore("clarity_and_tone", 4.5, "Concise, direct customer language with no internal jargon."),
            ]
        elif len(failed_map) >= 3:
            scores = [
                RubricScore("accuracy", 2.5, "Missing key signature header and cutoff consequence."),
                RubricScore("actionability", 2.0, "Example JSON is broken and migration steps are too vague."),
                RubricScore("clarity_and_tone", 3.0, "Leaks internal ticket/host details unsuitable for customers."),
            ]
        else:
            scores = [
                RubricScore("accuracy", 3.5, "Mostly accurate, but missing exact required header name."),
                RubricScore("actionability", 4.0, "Clear checklist and valid JSON payload."),
                RubricScore("clarity_and_tone", 4.2, "Clean customer-facing structure and tone."),
            ]

        return CritiqueReport(
            deterministic_checks=deterministic_checks,
            rubric_scores=scores,
            defects=defects,
        )


# ---------------------------------------------------------------------------
# Reflection loop controller
# ---------------------------------------------------------------------------


def run_reflection_loop(
    source_notes: str,
    backend: OpenAIReflectionBackend | DemoReflectionBackend,
    max_rounds: int = 3,
    min_score: float = 4.2,
    mode: str = "demo",
) -> ReflectionResult:
    """Run bounded Generator -> Deterministic Checks -> Rubric Critic loop."""
    history: list[RevisionRound] = []
    previous_draft: str | None = None
    previous_critique: CritiqueReport | None = None

    best_round: RevisionRound | None = None

    for round_num in range(1, max_rounds + 1):
        draft = backend.generate(
            source_notes=source_notes,
            previous_draft=previous_draft,
            previous_critique=previous_critique,
            round_number=round_num,
        )
        det_checks = run_deterministic_checks(draft)
        critique = backend.critique(
            source_notes=source_notes,
            draft=draft,
            deterministic_checks=det_checks,
            round_number=round_num,
        )
        current = RevisionRound(round_number=round_num, draft=draft, critique=critique)
        history.append(current)

        # Retain the strongest draft so a later regression never degrades output quality
        if best_round is None or critique.overall_rank() > best_round.critique.overall_rank():
            best_round = current

        if critique.meets_acceptance(min_score=min_score):
            return ReflectionResult(
                best_round=current.round_number,
                final_draft=current.draft,
                accepted=True,
                stop_reason=(
                    f"Accepted on round {round_num}: all deterministic checks passed, "
                    f"0 blocking defects, rubric avg {critique.average_rubric_score:.2f} >= {min_score:.2f}."
                ),
                history=history,
                mode=mode,
            )

        previous_draft = draft
        previous_critique = critique

    assert best_round is not None
    return ReflectionResult(
        best_round=best_round.round_number,
        final_draft=best_round.draft,
        accepted=False,
        stop_reason=(
            f"Reached max_rounds={max_rounds}. Returning best draft from round "
            f"{best_round.round_number} (rubric avg {best_round.critique.average_rubric_score:.2f})."
        ),
        history=history,
        mode=mode,
    )


def select_backend(force_demo: bool) -> tuple[OpenAIReflectionBackend | DemoReflectionBackend, str]:
    env_demo = os.getenv("AGENT_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()

    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    if force_demo or env_demo or (not api_key and not is_local):
        return DemoReflectionBackend(), "demo"
    return OpenAIReflectionBackend(api_key=api_key or "ollama", base_url=base_url, model=model), f"live ({model})"


def print_reflection_result(result: ReflectionResult) -> None:
    print(f"Mode: {result.mode}")
    print("=" * 72)

    for item in result.history:
        c = item.critique
        det_pass_count = sum(1 for chk in c.deterministic_checks if chk.passed)
        print(
            f"\n[Round {item.round_number}] "
            f"Deterministic: {det_pass_count}/{len(c.deterministic_checks)} passed | "
            f"Rubric Avg: {c.average_rubric_score:.2f}/5.00 | "
            f"Blocking Defects: {len(c.blocking_defects)}"
        )
        for chk in c.deterministic_checks:
            mark = "PASS" if chk.passed else "FAIL"
            print(f"  - [{mark}] {chk.name}: {chk.detail}")
        for score in c.rubric_scores:
            print(f"  - Rubric ({score.criterion}): {score.score:.1f}/5.0 — {score.note}")
        if c.defects:
            print("  Defects to address:")
            for defect in c.defects:
                print(f"    * [{defect.severity.upper()}] ({defect.area}) {defect.issue} | Fix: {defect.fix}")

    print("\n" + "=" * 72)
    print(f"STOP REASON: {result.stop_reason}")
    print(f"SELECTED DRAFT: Round {result.best_round}")
    print("-" * 72)
    print(result.final_draft)


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 13: Reflection & Self-Critique Loop")
    parser.add_argument("--max-rounds", type=int, default=3, help="Hard cap on revision rounds.")
    parser.add_argument("--min-score", type=float, default=4.2, help="Minimum rubric average (1-5) to accept.")
    parser.add_argument("--demo", action="store_true", help="Run with deterministic offline generator/critic.")
    args = parser.parse_args()

    backend, mode = select_backend(force_demo=args.demo)
    result = run_reflection_loop(
        source_notes=RAW_ENGINEERING_NOTES,
        backend=backend,
        max_rounds=args.max_rounds,
        min_score=args.min_score,
        mode=mode,
    )
    print_reflection_result(result)


if __name__ == "__main__":
    main()
