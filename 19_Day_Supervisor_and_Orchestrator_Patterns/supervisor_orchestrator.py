"""Supervisor / Orchestrator pattern with bounded workers, validation, and retries.

Unlike Day 18's peer-to-peer handoffs where specialists pass control sideways,
a central Supervisor owns the plan from start to finish:
  1. Decomposes the user goal into explicit subtasks with dependencies and
     acceptance criteria.
  2. Runs specialized workers (`researcher`, `analyst`, `writer`) using ONLY
     the bounded context each subtask needs.
  3. Validates every worker result against acceptance criteria before accepting it.
  4. Retries failed subtasks with concrete validation feedback (bounded retries).
  5. Synthesizes a final report from verified artifacts.

Usage:
  # Offline deterministic demo (shows validation failure -> feedback -> retry pass)
  python3 19_Day_Supervisor_and_Orchestrator_Patterns/supervisor_orchestrator.py --demo

  # Live run against any OpenAI-compatible API
  OPENAI_API_KEY=... python3 19_Day_Supervisor_and_Orchestrator_Patterns/supervisor_orchestrator.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
import re
from typing import Any
import urllib.error
import urllib.request

DEFAULT_GOAL = (
    "Evaluate migrating our EU order-processing database from single-region Postgres "
    "to active-passive multi-region Postgres, including exact cost and RTO trade-offs "
    "and a recommendation memo for engineering leadership."
)

# ---------------------------------------------------------------------------
# Local knowledge corpus & deterministic worker tools
# ---------------------------------------------------------------------------

DOC_CORPUS: dict[str, dict[str, str]] = {
    "doc-arch-01": {
        "title": "EU Orders Postgres Architecture Baseline",
        "body": (
            "Current production runs single-region Postgres in eu-central-1. "
            "Observed Recovery Time Objective (RTO) during regional failover drills is 45 minutes. "
            "Asynchronous cross-region streaming adds ~18ms write p95 overhead with ~85ms replication lag; "
            "synchronous cross-region commits add ~95ms write p95 overhead."
        ),
    },
    "doc-cost-02": {
        "title": "Database Infrastructure Cost Model",
        "body": (
            "Current single-region primary + local standby costs $4,200/month. "
            "Adding a warm active-passive standby in eu-west-1 plus cross-region WAL transfer "
            "raises projected monthly spend to $7,350/month."
        ),
    },
    "doc-slo-03": {
        "title": "2026 Availability & Recovery Targets",
        "body": (
            "Tier-1 checkout and order-processing services must meet a target RTO of 5 minutes "
            "or less by Q3 while keeping checkout write latency p95 under +30ms regression."
        ),
    },
}


def tool_search_internal_docs(query: str) -> dict[str, Any]:
    terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    matches = []
    for doc_id, doc in DOC_CORPUS.items():
        haystack = f"{doc['title']} {doc['body']}".lower()
        score = sum(1 for term in terms if term in haystack)
        if score > 0 or not terms:
            matches.append(
                {
                    "doc_id": doc_id,
                    "title": doc["title"],
                    "body": doc["body"],
                    "match_score": score,
                }
            )
    matches.sort(key=lambda m: m["match_score"], reverse=True)
    return {"query": query, "results": matches}


def tool_compute_cost_and_slo(
    current_monthly_usd: float,
    projected_monthly_usd: float,
    current_rto_min: float,
    target_rto_min: float,
) -> dict[str, Any]:
    monthly_delta = round(projected_monthly_usd - current_monthly_usd, 2)
    annual_delta = round(monthly_delta * 12, 2)
    pct_increase = round((monthly_delta / current_monthly_usd) * 100, 1)
    rto_reduction_pct = round(
        ((current_rto_min - target_rto_min) / current_rto_min) * 100, 1
    )
    return {
        "current_monthly_usd": current_monthly_usd,
        "projected_monthly_usd": projected_monthly_usd,
        "monthly_delta_usd": monthly_delta,
        "annual_delta_usd": annual_delta,
        "cost_increase_pct": pct_increase,
        "current_rto_min": current_rto_min,
        "target_rto_min": target_rto_min,
        "rto_reduction_pct": rto_reduction_pct,
    }


# ---------------------------------------------------------------------------
# Subtask, Worker Output & Validation Data Models
# ---------------------------------------------------------------------------

@dataclass
class Subtask:
    id: str
    worker: str
    objective: str
    acceptance_criteria: list[str]
    depends_on: list[str] = field(default_factory=list)
    max_retries: int = 2


@dataclass
class WorkerArtifact:
    subtask_id: str
    worker: str
    attempt: int
    content: str
    citations: list[str] = field(default_factory=list)
    computed_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM Client (Live OpenAI-compatible + Offline Demo)
# ---------------------------------------------------------------------------

class OpenAICompatibleTextLLM:
    """Stdlib HTTP client for OpenAI-compatible chat completions."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        return str(data["choices"][0]["message"]["content"]).strip()


# ---------------------------------------------------------------------------
# Specialized Workers (Each runs with strictly bounded context)
# ---------------------------------------------------------------------------

def run_researcher_worker(
    subtask: Subtask,
    feedback: list[str],
    live_llm: OpenAICompatibleTextLLM | None,
    attempt: int,
) -> WorkerArtifact:
    # Researcher only sees its own subtask + docs returned by its scoped tool.
    docs_payload = tool_search_internal_docs("postgres multi region cost rto latency")
    docs = docs_payload["results"]
    doc_ids = [d["doc_id"] for d in docs]

    if live_llm is not None:
        context = "\n".join(f"[{d['doc_id']}] {d['title']}: {d['body']}" for d in docs)
        prompt = (
            f"Objective: {subtask.objective}\n"
            f"Acceptance criteria: {subtask.acceptance_criteria}\n"
            f"Previous validation feedback (fix all items): {feedback}\n\n"
            f"Internal documents:\n{context}\n\n"
            "Return concise factual findings citing doc IDs in brackets like [doc-arch-01]."
        )
        text = live_llm.complete(
            "You are a technical Researcher worker. Cite exact doc IDs and exact numbers.",
            prompt,
        )
    else:
        text = (
            "Verified internal facts:\n"
            "- Baseline architecture [doc-arch-01]: Single-region Postgres in eu-central-1 "
            "has 45-minute failover RTO. Async cross-region replication adds ~18ms write p95 "
            "overhead (~85ms lag); sync commits add ~95ms write p95.\n"
            "- Cost model [doc-cost-02]: Current spend is $4,200/month; active-passive "
            "multi-region in eu-west-1 is projected at $7,350/month.\n"
            "- SLO targets [doc-slo-03]: Tier-1 order processing requires <=5-minute RTO "
            "while keeping write latency regression under +30ms p95."
        )

    found_citations = sorted(set(re.findall(r"doc-[a-z]+-\d+", text))) or doc_ids
    return WorkerArtifact(
        subtask_id=subtask.id,
        worker="researcher",
        attempt=attempt,
        content=text,
        citations=found_citations,
    )


def run_analyst_worker(
    subtask: Subtask,
    dependency_artifacts: dict[str, WorkerArtifact],
    feedback: list[str],
    live_llm: OpenAICompatibleTextLLM | None,
    attempt: int,
) -> WorkerArtifact:
    # Analyst receives ONLY the researcher's verified summary, never full history.
    research_text = dependency_artifacts["task_research"].content
    metrics = tool_compute_cost_and_slo(
        current_monthly_usd=4200.0,
        projected_monthly_usd=7350.0,
        current_rto_min=45.0,
        target_rto_min=5.0,
    )

    if live_llm is not None:
        prompt = (
            f"Objective: {subtask.objective}\n"
            f"Acceptance criteria: {subtask.acceptance_criteria}\n"
            f"Validation feedback to fix: {feedback}\n\n"
            f"Verified research summary:\n{research_text}\n\n"
            f"Exact calculator output:\n{json.dumps(metrics, indent=2)}\n\n"
            "Write a short quantitative analysis including monthly delta ($3,150), "
            "annual delta ($37,800), cost increase percentage (75.0%), RTO reduction "
            "(88.9%), and async vs sync latency trade-off."
        )
        text = live_llm.complete(
            "You are a quantitative Analyst worker. Use only verified calculator figures.",
            prompt,
        )
    else:
        if attempt == 1 and not feedback:
            # Deliberately produce an incomplete attempt 1 in --demo mode so the
            # learner sees the Supervisor reject it and trigger a targeted retry!
            text = (
                "Initial quantitative check: Monthly spend increases by $3,150/month "
                "and RTO drops from 45 minutes to 5 minutes. Async replication keeps "
                "write overhead around +18ms p95."
            )
        else:
            text = (
                "Quantitative trade-off analysis:\n"
                f"- Cost impact: +${metrics['monthly_delta_usd']:,.0f}/month "
                f"(+${metrics['annual_delta_usd']:,.0f}/year, a +{metrics['cost_increase_pct']}% increase "
                "from $4,200 to $7,350/month).\n"
                f"- Reliability impact: Failover RTO drops from {metrics['current_rto_min']:.0f} min "
                f"to {metrics['target_rto_min']:.0f} min ({metrics['rto_reduction_pct']}% reduction), "
                "meeting the Q3 Tier-1 SLO.\n"
                "- Replication mode decision: Asynchronous streaming (+18ms p95) stays within the "
                "+30ms latency budget, whereas synchronous commits (+95ms p95) violate it."
            )

    return WorkerArtifact(
        subtask_id=subtask.id,
        worker="analyst",
        attempt=attempt,
        content=text,
        citations=list(dependency_artifacts["task_research"].citations),
        computed_metrics=metrics,
    )


def run_writer_worker(
    subtask: Subtask,
    dependency_artifacts: dict[str, WorkerArtifact],
    feedback: list[str],
    live_llm: OpenAICompatibleTextLLM | None,
    attempt: int,
) -> WorkerArtifact:
    # Writer has NO external tools; it can only synthesize vetted upstream artifacts.
    research = dependency_artifacts["task_research"]
    analysis = dependency_artifacts["task_analysis"]

    if live_llm is not None:
        prompt = (
            f"Objective: {subtask.objective}\n"
            f"Acceptance criteria: {subtask.acceptance_criteria}\n"
            f"Validation feedback to fix: {feedback}\n\n"
            f"Research artifact:\n{research.content}\n\n"
            f"Analysis artifact:\n{analysis.content}\n\n"
            "Write a concise leadership memo containing the exact headings:\n"
            "## Executive Summary\n## Cost & Reliability Trade-offs\n"
            "## Recommendation\n## Sources\n"
            "Cite doc-arch-01, doc-cost-02, and doc-slo-03."
        )
        text = live_llm.complete(
            "You are an executive Writer worker. Stay strictly grounded in the provided artifacts.",
            prompt,
        )
    else:
        text = (
            "## Executive Summary\n"
            "We recommend migrating EU order-processing Postgres to an active-passive "
            "multi-region topology (eu-central-1 primary, eu-west-1 warm standby) using "
            "asynchronous WAL streaming. This meets the Q3 <=5-minute RTO target while "
            "staying within the +30ms write latency budget.\n\n"
            "## Cost & Reliability Trade-offs\n"
            "- Spend increases from $4,200/month to $7,350/month (+$3,150/month, "
            "+$37,800/year, +75.0%).\n"
            "- Failover RTO improves from 45 minutes to <=5 minutes (88.9% reduction).\n"
            "- Async replication adds ~18ms write p95 (~85ms lag), avoiding the +95ms "
            "regression of synchronous cross-region commits.\n\n"
            "## Recommendation\n"
            "Approve the $37,800/year infrastructure budget and roll out asynchronous "
            "eu-west-1 standby replication with automated quarterly failover drills.\n\n"
            "## Sources\n"
            "- [doc-arch-01] EU Orders Postgres Architecture Baseline\n"
            "- [doc-cost-02] Database Infrastructure Cost Model\n"
            "- [doc-slo-03] 2026 Availability & Recovery Targets"
        )

    citations = sorted(set(re.findall(r"doc-[a-z]+-\d+", text)))
    return WorkerArtifact(
        subtask_id=subtask.id,
        worker="writer",
        attempt=attempt,
        content=text,
        citations=citations,
    )


# ---------------------------------------------------------------------------
# Deterministic Acceptance-Criteria Validator
# ---------------------------------------------------------------------------

def validate_artifact(subtask: Subtask, artifact: WorkerArtifact) -> ValidationResult:
    failures: list[str] = []
    text_lower = artifact.content.lower()

    if subtask.worker == "researcher":
        valid_cites = [c for c in artifact.citations if c in DOC_CORPUS]
        if len(valid_cites) < 2:
            failures.append("Must cite at least 2 valid internal doc IDs.")
        if "$4,200" not in artifact.content and "4200" not in artifact.content:
            failures.append("Must include current baseline monthly spend ($4,200).")
        if "45" not in artifact.content:
            failures.append("Must include baseline 45-minute RTO figure.")

    elif subtask.worker == "analyst":
        if "37,800" not in artifact.content and "37800" not in artifact.content:
            failures.append("Must report exact annual cost delta ($37,800/year).")
        if "75" not in artifact.content:
            failures.append("Must report percentage cost increase (+75.0%).")
        if "88.9" not in artifact.content:
            failures.append("Must report RTO percentage reduction (88.9%).")
        if "sync" not in text_lower or "async" not in text_lower:
            failures.append("Must compare async vs. sync replication latency trade-offs.")

    elif subtask.worker == "writer":
        for heading in [
            "## Executive Summary",
            "## Cost & Reliability Trade-offs",
            "## Recommendation",
            "## Sources",
        ]:
            if heading.lower() not in text_lower:
                failures.append(f"Missing required section heading: '{heading}'.")
        valid_cites = [c for c in artifact.citations if c in DOC_CORPUS]
        if len(valid_cites) < 2:
            failures.append("Memo must cite at least 2 verified source document IDs.")
        if len(artifact.content.split()) > 350:
            failures.append("Memo exceeds concise 350-word executive limit.")

    return ValidationResult(passed=not failures, failures=failures)


# ---------------------------------------------------------------------------
# Central Supervisor Orchestrator
# ---------------------------------------------------------------------------

class SupervisorOrchestrator:
    """Decomposes work, delegates to bounded workers, validates, retries, synthesizes."""

    def __init__(self, live_llm: OpenAICompatibleTextLLM | None = None) -> None:
        self.live_llm = live_llm

    def build_plan(self, goal: str) -> list[Subtask]:
        return [
            Subtask(
                id="task_research",
                worker="researcher",
                objective=f"Gather verified architecture, cost, and SLO facts for: {goal}",
                acceptance_criteria=[
                    "Cite at least 2 internal document IDs",
                    "Include current spend ($4,200/mo) and baseline RTO (45 min)",
                ],
                depends_on=[],
                max_retries=2,
            ),
            Subtask(
                id="task_analysis",
                worker="analyst",
                objective=(
                    "Calculate monthly/annual cost delta, percentage increase, RTO reduction, "
                    "and compare async vs sync replication latency."
                ),
                acceptance_criteria=[
                    "Include exact annual delta ($37,800/year) and +75.0% increase",
                    "Include 88.9% RTO reduction and async vs sync latency comparison",
                ],
                depends_on=["task_research"],
                max_retries=2,
            ),
            Subtask(
                id="task_writing",
                worker="writer",
                objective="Draft a concise leadership recommendation memo from verified artifacts.",
                acceptance_criteria=[
                    "Include Executive Summary, Cost & Reliability Trade-offs, Recommendation, Sources",
                    "Cite verified document IDs and stay under 350 words",
                ],
                depends_on=["task_research", "task_analysis"],
                max_retries=2,
            ),
        ]

    def _dispatch_worker(
        self,
        subtask: Subtask,
        artifacts: dict[str, WorkerArtifact],
        feedback: list[str],
        attempt: int,
    ) -> WorkerArtifact:
        scoped_deps = {dep_id: artifacts[dep_id] for dep_id in subtask.depends_on}
        if subtask.worker == "researcher":
            return run_researcher_worker(subtask, feedback, self.live_llm, attempt)
        if subtask.worker == "analyst":
            return run_analyst_worker(subtask, scoped_deps, feedback, self.live_llm, attempt)
        if subtask.worker == "writer":
            return run_writer_worker(subtask, scoped_deps, feedback, self.live_llm, attempt)
        raise ValueError(f"Unknown worker: {subtask.worker}")

    def run(self, goal: str) -> str:
        plan = self.build_plan(goal)
        print(f"Goal: {goal}\n")
        print(f"[SUPERVISOR] Created {len(plan)}-step delegation plan:")
        for st in plan:
            deps = ", ".join(st.depends_on) if st.depends_on else "none"
            print(f"  - {st.id} -> worker='{st.worker}' (depends_on: {deps})")

        verified_artifacts: dict[str, WorkerArtifact] = {}
        audit_summary: list[str] = []

        for subtask in plan:
            print(f"\n[SUPERVISOR] Delegating '{subtask.id}' to worker '{subtask.worker}'...")
            feedback: list[str] = []
            max_attempts = subtask.max_retries + 1

            for attempt in range(1, max_attempts + 1):
                artifact = self._dispatch_worker(
                    subtask, verified_artifacts, feedback, attempt
                )
                validation = validate_artifact(subtask, artifact)
                if validation.passed:
                    print(
                        f"  [PASS] {subtask.worker} attempt {attempt}/{max_attempts} "
                        f"passed validation (citations={artifact.citations})."
                    )
                    verified_artifacts[subtask.id] = artifact
                    audit_summary.append(
                        f"  - {subtask.id} ({subtask.worker}): PASSED on attempt {attempt}"
                    )
                    break

                print(
                    f"  [FAIL] {subtask.worker} attempt {attempt}/{max_attempts} rejected:"
                )
                for reason in validation.failures:
                    print(f"         * {reason}")
                feedback = validation.failures
                if attempt < max_attempts:
                    print(
                        f"  [RETRY] Sending targeted feedback back to '{subtask.worker}'..."
                    )
            else:
                raise RuntimeError(
                    f"Subtask '{subtask.id}' failed validation after {max_attempts} attempts: {feedback}"
                )

        print("\n[SUPERVISOR] All subtasks verified. Synthesizing final report...\n")
        memo = verified_artifacts["task_writing"].content
        scorecard = "\n".join(audit_summary)
        return f"{memo}\n\n---\nSupervisor Verification Scorecard:\n{scorecard}"


def should_use_demo(force_demo: bool) -> bool:
    if force_demo or os.getenv("AGENT_DEMO_MODE") == "1":
        return True
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervisor/orchestrator multi-agent delegation demo."
    )
    parser.add_argument("--demo", action="store_true", help="Run offline deterministic demo.")
    parser.add_argument("--goal", type=str, default=DEFAULT_GOAL, help="Top-level goal.")
    args = parser.parse_args()

    if should_use_demo(args.demo):
        print("=== Day 19: Supervisor & Orchestrator Pattern [DEMO MODE] ===\n")
        supervisor = SupervisorOrchestrator(live_llm=None)
    else:
        llm = OpenAICompatibleTextLLM(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        )
        print(f"=== Day 19: Supervisor & Orchestrator Pattern [LIVE: {llm.model}] ===\n")
        supervisor = SupervisorOrchestrator(live_llm=llm)

    final_report = supervisor.run(args.goal)
    print(final_report)


if __name__ == "__main__":
    main()
