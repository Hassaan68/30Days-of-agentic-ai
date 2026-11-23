"""Day 12: Plan-and-Execute Agent with Typed Plans and Dynamic Replanning.

Demonstrates:
- Breaking a multi-step goal into an explicit, typed `Plan` of `PlanStep` objects.
- Enforcing dependency order (`depends_on`) and validating against cycles.
- Inspecting intermediate observations and dynamically revising the remaining plan
  when a step uncovers unexpected facts (without throwing away completed work).
- Supporting both live OpenAI-compatible APIs and an offline `--demo` mode.

Usage:
    python3 12_Day_Planning_and_Task_Decomposition/plan_and_execute.py --demo
    python3 12_Day_Planning_and_Task_Decomposition/plan_and_execute.py --goal "Assess release readiness for checkout-api v2.14.0"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

StepStatus = Literal["pending", "in_progress", "completed", "failed", "skipped"]

# ---------------------------------------------------------------------------
# Typed Plan & Step data structures
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """Single unit of work in an execution plan."""

    id: str
    title: str
    tool: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class Plan:
    """Versioned multi-step execution plan with dependency validation."""

    goal: str
    steps: list[PlanStep]
    version: int = 1
    revision_notes: list[str] = field(default_factory=list)

    def step_map(self) -> dict[str, PlanStep]:
        return {step.id: step for step in self.steps}

    def validate(self) -> None:
        """Verify unique IDs, valid dependency references, and acyclic graph."""
        by_id: dict[str, PlanStep] = {}
        for step in self.steps:
            if step.id in by_id:
                raise ValueError(f"Duplicate step id '{step.id}' in plan.")
            by_id[step.id] = step

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in by_id:
                    raise ValueError(f"Step '{step.id}' depends on unknown step '{dep}'.")

        # Cycle check with 3-color DFS (0=unvisited, 1=visiting, 2=done)
        state: dict[str, int] = {step.id: 0 for step in self.steps}

        def dfs(node_id: str) -> None:
            state[node_id] = 1
            for dep in by_id[node_id].depends_on:
                if state[dep] == 1:
                    raise ValueError(f"Dependency cycle detected involving '{node_id}' -> '{dep}'.")
                if state[dep] == 0:
                    dfs(dep)
            state[node_id] = 2

        for step in self.steps:
            if state[step.id] == 0:
                dfs(step.id)

    def next_ready_step(self) -> PlanStep | None:
        """Return the first pending step whose dependencies are all completed."""
        by_id = self.step_map()
        for step in self.steps:
            if step.status != "pending":
                continue
            if all(by_id[dep].status == "completed" for dep in step.depends_on):
                return step
        return None

    def is_complete(self) -> bool:
        return all(step.status in {"completed", "skipped"} for step in self.steps)

    def format_summary(self) -> str:
        lines = [f"Plan v{self.version} — Goal: {self.goal}"]
        for step in self.steps:
            deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
            lines.append(f"  [{step.status.upper():11}] {step.id}: {step.title} -> tool={step.tool}{deps}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulated release engineering tools
# ---------------------------------------------------------------------------


def fetch_release_metadata(args: dict[str, Any]) -> dict[str, Any]:
    service = str(args.get("service", "checkout-api"))
    version = str(args.get("version", "v2.14.0"))
    return {
        "service": service,
        "version": version,
        "commit_sha": "8f3c21a",
        "pr_count": 6,
        "changed_areas": ["payments/idempotency.py", "migrations/2026_05_add_idempotency_idx.sql"],
        "ci_tests_passed": True,
        "code_owners_approved": True,
    }


def check_staging_health(args: dict[str, Any]) -> dict[str, Any]:
    service = str(args.get("service", "checkout-api"))
    # Unexpected observation: staging has an unapplied migration flagged for lock risk.
    return {
        "service": service,
        "http_5xx_rate_pct": 0.04,
        "p95_latency_ms": 142,
        "pending_migrations": ["2026_05_add_idempotency_idx.sql"],
        "staging_db_lock_warning": True,
        "warning_detail": "Migration 2026_05_add_idempotency_idx.sql has not been validated for table-locking DDL.",
    }


def inspect_migration(args: dict[str, Any]) -> dict[str, Any]:
    filename = str(args.get("migration_file", "2026_05_add_idempotency_idx.sql"))
    return {
        "migration_file": filename,
        "target_table": "orders",
        "approx_row_count": 48_000_000,
        "sql_preview": "CREATE INDEX idx_orders_idempotency_key ON orders (idempotency_key);",
        "uses_concurrently": False,
        "blocking_lock_risk": "HIGH",
        "recommended_fix": "Rewrite as CREATE INDEX CONCURRENTLY idx_orders_idempotency_key ON orders (idempotency_key);",
    }


def check_rollback_artifact(args: dict[str, Any]) -> dict[str, Any]:
    service = str(args.get("service", "checkout-api"))
    return {
        "service": service,
        "previous_stable_version": "v2.13.4",
        "image_digest_verified": True,
        "rollback_tested_in_staging": True,
    }


TOOLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "fetch_release_metadata": fetch_release_metadata,
    "check_staging_health": check_staging_health,
    "inspect_migration": inspect_migration,
    "check_rollback_artifact": check_rollback_artifact,
}

TOOL_DESCRIPTIONS = {
    "fetch_release_metadata": "Fetch release PRs, changed files, CI status. Args: {service, version}.",
    "check_staging_health": "Check staging error rate, latency, and pending DB migrations. Args: {service}.",
    "inspect_migration": "Inspect a SQL migration file for table lock risks and row counts. Args: {migration_file}.",
    "check_rollback_artifact": "Verify previous stable build exists and rollback is tested. Args: {service}.",
    "synthesize_report": "Built-in final step: compile completed step results into a Go/No-Go readiness report. Args: {}.",
}


# ---------------------------------------------------------------------------
# Planner / Replanner / Synthesizer backends (Live HTTP + Deterministic Demo)
# ---------------------------------------------------------------------------


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Expected JSON object in model output, got: {raw_text[:200]}")
    return json.loads(cleaned[start : end + 1])


class OpenAIPlanBackend:
    """Uses an OpenAI-compatible endpoint to create, revise, and synthesize plans."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 30) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat(self, system: str, user: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.0,
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
        return str(data["choices"][0]["message"]["content"])

    def create_initial_plan(self, goal: str) -> Plan:
        system = (
            "You are a release engineering planner. Return ONLY a JSON object with a 'steps' list.\n"
            "Each step must have: id (string like 'S1'), title, tool, tool_args (object), depends_on (list of step ids).\n"
            f"Available tools:\n{json.dumps(TOOL_DESCRIPTIONS, indent=2)}\n"
            "End the plan with a step using tool 'synthesize_report' that depends on prior checks."
        )
        raw = self._chat(system, f"Create an initial 3-to-4 step plan for goal: {goal}")
        data = _extract_json_object(raw)
        steps = [
            PlanStep(
                id=str(item["id"]),
                title=str(item["title"]),
                tool=str(item["tool"]),
                tool_args=dict(item.get("tool_args", {})),
                depends_on=[str(d) for d in item.get("depends_on", [])],
            )
            for item in data["steps"]
        ]
        plan = Plan(goal=goal, steps=steps, version=1)
        plan.validate()
        return plan

    def maybe_replan(self, plan: Plan, last_step: PlanStep) -> tuple[bool, str]:
        system = (
            "You are a dynamic replanner. Inspect the completed step's result and the remaining pending steps.\n"
            "If an unexpected risk or prerequisite was discovered and is not yet covered by existing steps, return:\n"
            '{"should_replan": true, "reason": "...", "insert_steps": [{"id": "...", "title": "...", "tool": "...", "tool_args": {...}, "depends_on": [...]}], "add_deps_to_step": {"<existing_step_id>": ["<new_step_id>"]}}\n'
            'Otherwise return: {"should_replan": false, "reason": "No changes needed."}\n'
            f"Available tools:\n{json.dumps(TOOL_DESCRIPTIONS, indent=2)}"
        )
        state = {
            "goal": plan.goal,
            "last_completed_step": {
                "id": last_step.id,
                "tool": last_step.tool,
                "result": last_step.result,
            },
            "existing_steps": [
                {
                    "id": s.id,
                    "title": s.title,
                    "tool": s.tool,
                    "status": s.status,
                    "depends_on": s.depends_on,
                }
                for s in plan.steps
            ],
        }
        raw = self._chat(system, json.dumps(state, indent=2))
        decision = _extract_json_object(raw)
        if not decision.get("should_replan"):
            return False, str(decision.get("reason", "No changes needed."))

        existing_ids = {s.id for s in plan.steps}
        new_steps: list[PlanStep] = []
        for item in decision.get("insert_steps", []):
            if item["id"] in existing_ids:
                continue
            new_steps.append(
                PlanStep(
                    id=str(item["id"]),
                    title=str(item["title"]),
                    tool=str(item["tool"]),
                    tool_args=dict(item.get("tool_args", {})),
                    depends_on=[str(d) for d in item.get("depends_on", [])],
                )
            )

        if not new_steps:
            return False, "Replanner proposed no new unique steps."

        # Insert new steps right before the final synthesis step
        synth_idx = next(
            (i for i, s in enumerate(plan.steps) if s.tool == "synthesize_report"),
            len(plan.steps),
        )
        plan.steps[synth_idx:synth_idx] = new_steps

        for target_id, extra_deps in decision.get("add_deps_to_step", {}).items():
            for step in plan.steps:
                if step.id == target_id:
                    for dep in extra_deps:
                        if dep not in step.depends_on:
                            step.depends_on.append(dep)

        plan.validate()
        plan.version += 1
        reason = str(decision.get("reason", "Adjusted plan after new observation."))
        plan.revision_notes.append(f"v{plan.version}: {reason}")
        return True, reason

    def synthesize(self, plan: Plan) -> str:
        system = (
            "Write a concise, grounded Release Readiness Report (Go or No-Go) based strictly on the "
            "completed step observations. Include key findings, blocking risks, and exact next actions."
        )
        evidence = [
            {"id": s.id, "title": s.title, "tool": s.tool, "result": s.result}
            for s in plan.steps
            if s.status == "completed" and s.tool != "synthesize_report"
        ]
        return self._chat(system, json.dumps({"goal": plan.goal, "evidence": evidence}, indent=2))


class DemoPlanBackend:
    """Deterministic local planner/replanner illustrating a concrete plan revision."""

    def create_initial_plan(self, goal: str) -> Plan:
        # Notice the planner initially assumes standard release checks are enough.
        plan = Plan(
            goal=goal,
            version=1,
            steps=[
                PlanStep(
                    id="S1",
                    title="Fetch release metadata and CI status",
                    tool="fetch_release_metadata",
                    tool_args={"service": "checkout-api", "version": "v2.14.0"},
                    depends_on=[],
                ),
                PlanStep(
                    id="S2",
                    title="Check staging service health and DB state",
                    tool="check_staging_health",
                    tool_args={"service": "checkout-api"},
                    depends_on=["S1"],
                ),
                PlanStep(
                    id="S3",
                    title="Verify rollback artifact for previous stable release",
                    tool="check_rollback_artifact",
                    tool_args={"service": "checkout-api"},
                    depends_on=["S1"],
                ),
                PlanStep(
                    id="S4",
                    title="Compile final Go/No-Go release readiness report",
                    tool="synthesize_report",
                    tool_args={},
                    depends_on=["S2", "S3"],
                ),
            ],
        )
        plan.validate()
        return plan

    def maybe_replan(self, plan: Plan, last_step: PlanStep) -> tuple[bool, str]:
        # Concrete trigger: Step S2 reveals an unvalidated pending migration with a lock warning.
        if last_step.tool == "check_staging_health" and last_step.result:
            pending = last_step.result.get("pending_migrations", [])
            lock_warning = bool(last_step.result.get("staging_db_lock_warning"))
            already_scheduled = any(s.tool == "inspect_migration" for s in plan.steps)

            if pending and lock_warning and not already_scheduled:
                migration_file = pending[0]
                new_step = PlanStep(
                    id="S2b",
                    title=f"Inspect pending migration '{migration_file}' for table-lock risk",
                    tool="inspect_migration",
                    tool_args={"migration_file": migration_file},
                    depends_on=[last_step.id],
                )
                synth_idx = next(
                    (i for i, s in enumerate(plan.steps) if s.tool == "synthesize_report"),
                    len(plan.steps),
                )
                plan.steps.insert(synth_idx, new_step)
                for step in plan.steps:
                    if step.tool == "synthesize_report" and new_step.id not in step.depends_on:
                        step.depends_on.append(new_step.id)

                plan.validate()
                plan.version += 1
                reason = (
                    f"Step {last_step.id} reported pending migration '{migration_file}' with a DB lock warning. "
                    f"Inserted step {new_step.id} before final synthesis."
                )
                plan.revision_notes.append(f"v{plan.version}: {reason}")
                return True, reason

        return False, "No plan changes needed."

    def synthesize(self, plan: Plan) -> str:
        by_tool = {s.tool: s.result or {} for s in plan.steps if s.status == "completed"}
        meta = by_tool.get("fetch_release_metadata", {})
        health = by_tool.get("check_staging_health", {})
        mig = by_tool.get("inspect_migration", {})
        rollback = by_tool.get("check_rollback_artifact", {})

        verdict = "NO-GO (HOLD RELEASE)" if mig.get("blocking_lock_risk") == "HIGH" else "GO"
        lines = [
            f"RELEASE READINESS DECISION: {verdict}",
            f"Service / Version: {meta.get('service')} {meta.get('version')} (commit {meta.get('commit_sha')})",
            "",
            "Verified Signals:",
            f"- CI & Reviews: tests_passed={meta.get('ci_tests_passed')}, code_owners_approved={meta.get('code_owners_approved')}",
            f"- Staging Health: 5xx={health.get('http_5xx_rate_pct')}%, p95={health.get('p95_latency_ms')}ms",
            f"- Rollback Readiness: {rollback.get('previous_stable_version')} verified={rollback.get('image_digest_verified')}, tested={rollback.get('rollback_tested_in_staging')}",
        ]
        if mig:
            lines.extend(
                [
                    "",
                    "Blocking Finding (uncovered via dynamic replanning):",
                    f"- Migration `{mig.get('migration_file')}` targets `{mig.get('target_table')}` (~{mig.get('approx_row_count'):,} rows).",
                    f"- Lock risk: {mig.get('blocking_lock_risk')} (`uses_concurrently={mig.get('uses_concurrently')}`).",
                    f"- Required fix before deploy: {mig.get('recommended_fix')}",
                ]
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan-and-Execute Controller
# ---------------------------------------------------------------------------


def run_plan_and_execute(
    goal: str,
    backend: OpenAIPlanBackend | DemoPlanBackend,
    max_replans: int = 3,
    max_iterations: int = 12,
) -> tuple[Plan, str]:
    """Run Planner -> Executor -> Replanner loop with bounded safety limits."""
    plan = backend.create_initial_plan(goal)
    print(plan.format_summary())
    print("-" * 72)

    replan_count = 0
    final_report = ""

    for _ in range(max_iterations):
        step = plan.next_ready_step()
        if step is None:
            if plan.is_complete():
                break
            raise RuntimeError("Deadlock: pending steps remain but none have satisfied dependencies.")

        step.status = "in_progress"
        print(f"\n>>> Executing {step.id}: {step.title} (tool={step.tool})")

        if step.tool == "synthesize_report":
            final_report = backend.synthesize(plan)
            step.result = {"report_length_chars": len(final_report)}
            step.status = "completed"
            print(f"    Completed {step.id}: generated final report ({len(final_report)} chars).")
            continue

        handler = TOOLS.get(step.tool)
        if handler is None:
            step.status = "failed"
            step.error = f"Unknown tool '{step.tool}'."
            print(f"    FAILED {step.id}: {step.error}")
            break

        try:
            step.result = handler(step.tool_args)
            step.status = "completed"
            print(f"    Observation: {json.dumps(step.result)}")
        except Exception as exc:
            step.status = "failed"
            step.error = str(exc)
            print(f"    FAILED {step.id}: {step.error}")
            break

        # Replanning checkpoint after each tool observation
        if replan_count < max_replans and not plan.is_complete():
            revised, reason = backend.maybe_replan(plan, step)
            if revised:
                replan_count += 1
                print(f"\n!!! REPLAN TRIGGERED (#{replan_count}): {reason}")
                print(plan.format_summary())

    return plan, final_report


def select_backend(force_demo: bool) -> tuple[OpenAIPlanBackend | DemoPlanBackend, str]:
    env_demo = os.getenv("AGENT_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()

    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    if force_demo or env_demo or (not api_key and not is_local):
        return DemoPlanBackend(), "demo"
    return OpenAIPlanBackend(api_key=api_key or "ollama", base_url=base_url, model=model), f"live ({model})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 12: Plan-and-Execute with Dynamic Replanning")
    parser.add_argument(
        "--goal",
        default="Assess production release readiness for checkout-api v2.14.0.",
        help="Multi-step goal to plan and execute.",
    )
    parser.add_argument("--max-replans", type=int, default=3, help="Maximum plan revisions allowed.")
    parser.add_argument("--demo", action="store_true", help="Run with deterministic offline mock planner.")
    args = parser.parse_args()

    backend, mode = select_backend(force_demo=args.demo)
    print(f"Mode: {mode}")
    print("=" * 72)

    plan, report = run_plan_and_execute(
        goal=args.goal,
        backend=backend,
        max_replans=args.max_replans,
    )

    print("\n" + "=" * 72)
    print("FINAL PLAN STATE:")
    print(plan.format_summary())
    if plan.revision_notes:
        print("\nRevision History:")
        for note in plan.revision_notes:
            print(f"  - {note}")
    print("\n" + "=" * 72)
    print(report)


if __name__ == "__main__":
    main()
