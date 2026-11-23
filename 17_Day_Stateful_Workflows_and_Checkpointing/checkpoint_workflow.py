"""SQLite-backed stateful agent workflow with crash recovery and time-travel replay.

Demonstrates how production workflows avoid losing work across crashes, timeouts,
or human-approval pauses:
  1. Every workflow run gets a durable `run_id` and versioned `WorkflowState`.
  2. Each completed step writes an immutable snapshot to SQLite in a transaction.
  3. On restart with the same `run_id`, already-checkpointed steps are skipped
     rather than re-executed.
  4. Any historical checkpoint can be inspected or forked into a new `run_id`
     for time-travel debugging and alternate-branch replay.

Usage:
  # Run the full crash -> recovery -> inspection -> time-travel demo (temp DB)
  python3 17_Day_Stateful_Workflows_and_Checkpointing/checkpoint_workflow.py --demo

  # Persist the SQLite file locally so you can inspect it with `sqlite3`
  python3 17_Day_Stateful_Workflows_and_Checkpointing/checkpoint_workflow.py --db ./day17_checkpoints.db
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Callable
import urllib.error
import urllib.request

STATE_SCHEMA_VERSION = 1


class SimulatedCrashError(RuntimeError):
    """Injected mid-workflow failure used to test crash recovery."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class WorkflowState:
    """Versioned, JSON-serializable state passed between workflow steps."""

    topic: str
    audience: str = "platform engineering leads"
    schema_version: int = STATE_SCHEMA_VERSION
    facts: list[str] = field(default_factory=list)
    draft: str = ""
    verification_notes: list[str] = field(default_factory=list)
    final_brief: str = ""
    executed_steps: list[str] = field(default_factory=list)

    def to_canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def state_hash(self) -> str:
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_json(cls, raw: str) -> WorkflowState:
        data = json.loads(raw)
        data = migrate_state_dict(data)
        return cls(**data)


def migrate_state_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade older persisted state snapshots when the schema evolves."""
    version = int(data.get("schema_version", 1))
    if version > STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema_version={version} is newer than supported "
            f"version={STATE_SCHEMA_VERSION}."
        )
    # Example migration hook for future versions (e.g., v1 -> v2 adding defaults):
    data.setdefault("audience", "platform engineering leads")
    data["schema_version"] = STATE_SCHEMA_VERSION
    return data


@dataclass
class CheckpointRecord:
    """Immutable snapshot of workflow state persisted at a specific step."""

    checkpoint_id: int
    run_id: str
    step_index: int
    step_name: str
    state_version: int
    state_hash: str
    state: WorkflowState
    created_at: str


class SQLiteCheckpointer:
    """Durable append-only step checkpointer backed by SQLite (WAL mode)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    workflow_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_completed_step INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    parent_run_id TEXT,
                    forked_from_step INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    state_hash TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, step_index),
                    FOREIGN KEY(run_id) REFERENCES workflow_runs(run_id)
                );
                """
            )

    def create_or_get_run(
        self,
        run_id: str,
        workflow_name: str,
        initial_state: WorkflowState,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is not None:
                return dict(row)

            conn.execute(
                """
                INSERT INTO workflow_runs (
                    run_id, workflow_name, status, last_completed_step, created_at, updated_at
                ) VALUES (?, ?, 'running', 0, ?, ?)
                """,
                (run_id, workflow_name, now, now),
            )
            conn.execute(
                """
                INSERT INTO checkpoints (
                    run_id, step_index, step_name, state_version, state_hash, state_json, created_at
                ) VALUES (?, 0, 'init', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    initial_state.schema_version,
                    initial_state.state_hash(),
                    initial_state.to_canonical_json(),
                    now,
                ),
            )
            created = conn.execute(
                "SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            assert created is not None
            return dict(created)

    def save_checkpoint(
        self,
        run_id: str,
        step_index: int,
        step_name: str,
        state: WorkflowState,
        status: str = "running",
    ) -> CheckpointRecord:
        now = utc_now()
        state_json = state.to_canonical_json()
        state_hash = state.state_hash()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO checkpoints (
                    run_id, step_index, step_name, state_version, state_hash, state_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step_index,
                    step_name,
                    state.schema_version,
                    state_hash,
                    state_json,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE workflow_runs
                SET status = ?, last_completed_step = ?, error_message = NULL, updated_at = ?
                WHERE run_id = ?
                """,
                (status, step_index, now, run_id),
            )
            checkpoint_id = int(cursor.lastrowid or 0)
        return CheckpointRecord(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            step_index=step_index,
            step_name=step_name,
            state_version=state.schema_version,
            state_hash=state_hash,
            state=state,
            created_at=now,
        )

    def mark_status(
        self, run_id: str, status: str, error_message: str | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE workflow_runs
                SET status = ?, error_message = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (status, error_message, utc_now(), run_id),
            )

    def get_run_info(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def get_latest_checkpoint(self, run_id: str) -> CheckpointRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE run_id = ?
                ORDER BY step_index DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def get_checkpoint(self, run_id: str, step_index: int) -> CheckpointRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE run_id = ? AND step_index = ?
                """,
                (run_id, step_index),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def list_checkpoints(self, run_id: str) -> list[CheckpointRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE run_id = ?
                ORDER BY step_index ASC
                """,
                (run_id,),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def fork_run(
        self,
        source_run_id: str,
        from_step_index: int,
        new_run_id: str,
        state_mutator: Callable[[WorkflowState], WorkflowState] | None = None,
    ) -> CheckpointRecord:
        """Create a new run branched from a historical checkpoint (time-travel replay)."""
        source_run = self.get_run_info(source_run_id)
        if source_run is None:
            raise ValueError(f"Unknown source_run_id: {source_run_id}")
        source_cp = self.get_checkpoint(source_run_id, from_step_index)
        if source_cp is None:
            raise ValueError(
                f"No checkpoint found for {source_run_id} at step {from_step_index}"
            )

        branched_state = WorkflowState.from_json(source_cp.state.to_canonical_json())
        if state_mutator is not None:
            branched_state = state_mutator(branched_state)

        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_runs (
                    run_id, workflow_name, status, last_completed_step,
                    parent_run_id, forked_from_step, created_at, updated_at
                ) VALUES (?, ?, 'paused', ?, ?, ?, ?, ?)
                """,
                (
                    new_run_id,
                    source_run["workflow_name"],
                    from_step_index,
                    source_run_id,
                    from_step_index,
                    now,
                    now,
                ),
            )
            cursor = conn.execute(
                """
                INSERT INTO checkpoints (
                    run_id, step_index, step_name, state_version, state_hash, state_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_run_id,
                    from_step_index,
                    f"fork:{source_cp.step_name}",
                    branched_state.schema_version,
                    branched_state.state_hash(),
                    branched_state.to_canonical_json(),
                    now,
                ),
            )
            cp_id = int(cursor.lastrowid or 0)

        return CheckpointRecord(
            checkpoint_id=cp_id,
            run_id=new_run_id,
            step_index=from_step_index,
            step_name=f"fork:{source_cp.step_name}",
            state_version=branched_state.schema_version,
            state_hash=branched_state.state_hash(),
            state=branched_state,
            created_at=now,
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            checkpoint_id=int(row["checkpoint_id"]),
            run_id=str(row["run_id"]),
            step_index=int(row["step_index"]),
            step_name=str(row["step_name"]),
            state_version=int(row["state_version"]),
            state_hash=str(row["state_hash"]),
            state=WorkflowState.from_json(str(row["state_json"])),
            created_at=str(row["created_at"]),
        )


def optional_live_completion(prompt: str, use_live_llm: bool) -> str | None:
    """Call an OpenAI-compatible model only when --live is requested and configured."""
    if not use_live_llm:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()
    if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
        return None

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'ollama'}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"]).strip()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class WorkflowStep:
    index: int
    name: str
    handler: Callable[[WorkflowState, bool], WorkflowState]


def step_gather_facts(state: WorkflowState, use_live_llm: bool) -> WorkflowState:
    del use_live_llm
    state.facts = [
        f"Topic scope: {state.topic}",
        "p99 checkout latency climbed from 210ms to 1,450ms after enabling synchronous audit writes.",
        "Switching audit writes to a durable write-ahead queue reduced p99 latency to 240ms in staging.",
        "Recovery target requires zero dropped receipts during process restarts.",
    ]
    state.executed_steps.append("1_gather_facts")
    return state


def step_draft_brief(state: WorkflowState, use_live_llm: bool) -> WorkflowState:
    live_text = optional_live_completion(
        (
            f"Write a concise 3-sentence engineering brief for {state.audience} "
            f"based strictly on these facts:\n" + "\n".join(f"- {f}" for f in state.facts)
        ),
        use_live_llm,
    )
    if live_text:
        state.draft = live_text
    else:
        state.draft = (
            f"Audience: {state.audience}.\n"
            f"Summary for {state.topic}: Synchronous audit writes pushed p99 checkout "
            "latency from 210ms to 1,450ms. Staging benchmarks confirm that moving audit "
            "persistence to a durable write-ahead queue restores p99 latency to 240ms "
            "while preserving zero dropped receipts across process restarts."
        )
    state.executed_steps.append("2_draft_brief")
    return state


def step_verify_claims(state: WorkflowState, use_live_llm: bool) -> WorkflowState:
    del use_live_llm
    checks: list[str] = []
    for marker, label in [
        ("1,450ms", "Latency regression figure cited"),
        ("240ms", "Staging mitigation latency cited"),
        ("zero dropped receipts", "Durability requirement cited"),
    ]:
        passed = marker.lower() in state.draft.lower()
        checks.append(f"[{'PASS' if passed else 'WARN'}] {label}")
    state.verification_notes = checks
    state.executed_steps.append("3_verify_claims")
    return state


def step_publish_report(state: WorkflowState, use_live_llm: bool) -> WorkflowState:
    del use_live_llm
    checks_block = "\n".join(f"  {note}" for note in state.verification_notes)
    state.final_brief = (
        f"=== ENGINEERING BRIEF: {state.topic} ===\n"
        f"{state.draft}\n\n"
        f"Verification Checks:\n{checks_block}"
    )
    state.executed_steps.append("4_publish_report")
    return state


DEFAULT_STEPS: list[WorkflowStep] = [
    WorkflowStep(1, "gather_facts", step_gather_facts),
    WorkflowStep(2, "draft_brief", step_draft_brief),
    WorkflowStep(3, "verify_claims", step_verify_claims),
    WorkflowStep(4, "publish_report", step_publish_report),
]


class CheckpointedWorkflow:
    """Executes multi-step workflows with step-level SQLite checkpointing."""

    def __init__(
        self,
        checkpointer: SQLiteCheckpointer,
        workflow_name: str = "engineering_brief_pipeline",
        steps: list[WorkflowStep] | None = None,
        use_live_llm: bool = False,
    ) -> None:
        self.checkpointer = checkpointer
        self.workflow_name = workflow_name
        self.steps = steps or DEFAULT_STEPS
        self.use_live_llm = use_live_llm
        self.steps_run_in_this_instance: list[str] = []

    def run(
        self,
        run_id: str,
        initial_state: WorkflowState | None = None,
        fail_on_step: int | None = None,
        pause_after_step: int | None = None,
    ) -> WorkflowState:
        existing = self.checkpointer.get_latest_checkpoint(run_id)
        if existing is None:
            if initial_state is None:
                raise ValueError(
                    f"Run '{run_id}' does not exist yet; provide initial_state."
                )
            self.checkpointer.create_or_get_run(run_id, self.workflow_name, initial_state)
            latest = self.checkpointer.get_latest_checkpoint(run_id)
            assert latest is not None
        else:
            latest = existing
            print(
                f"[RESUME] Loaded run='{run_id}' from checkpoint step={latest.step_index} "
                f"({latest.step_name}, hash={latest.state_hash})"
            )

        state = latest.state
        last_done = latest.step_index

        for step in self.steps:
            if step.index <= last_done:
                print(
                    f"  [SKIP] Step {step.index} ({step.name}) already completed in checkpoint."
                )
                continue

            print(f"  [RUN ] Step {step.index} ({step.name})...")
            if fail_on_step == step.index:
                err_msg = f"Simulated process crash during step {step.index} ({step.name})"
                self.checkpointer.mark_status(run_id, "failed", error_message=err_msg)
                raise SimulatedCrashError(err_msg)

            state = step.handler(state, self.use_live_llm)
            self.steps_run_in_this_instance.append(step.name)

            is_final = step.index == self.steps[-1].index
            next_status = "completed" if is_final else "running"
            if pause_after_step == step.index and not is_final:
                next_status = "paused"

            cp = self.checkpointer.save_checkpoint(
                run_id=run_id,
                step_index=step.index,
                step_name=step.name,
                state=state,
                status=next_status,
            )
            print(
                f"  [SAVE] Checkpoint #{cp.checkpoint_id}: step={cp.step_index} "
                f"({cp.step_name}) state_hash={cp.state_hash} status={next_status}"
            )

            if next_status == "paused":
                print(f"[PAUSED] Run '{run_id}' paused cleanly after step {step.index}.")
                return state

        return state


def print_checkpoint_history(checkpointer: SQLiteCheckpointer, run_id: str) -> None:
    run_info = checkpointer.get_run_info(run_id)
    assert run_info is not None
    print(
        f"\n--- Checkpoint History for run_id='{run_id}' "
        f"(status={run_info['status']}, last_step={run_info['last_completed_step']}) ---"
    )
    if run_info.get("parent_run_id"):
        print(
            f"  Forked from parent_run_id='{run_info['parent_run_id']}' "
            f"at step {run_info['forked_from_step']}"
        )
    for cp in checkpointer.list_checkpoints(run_id):
        print(
            f"  cp_id={cp.checkpoint_id:<2} | step={cp.step_index} ({cp.step_name:<18}) | "
            f"v{cp.state_version} | hash={cp.state_hash} | "
            f"facts={len(cp.state.facts)} | draft_chars={len(cp.state.draft)}"
        )


def run_demo(db_path: Path, use_live_llm: bool = False) -> None:
    print(f"=== Day 17: Stateful Workflows & Checkpointing (SQLite: {db_path}) ===\n")
    checkpointer = SQLiteCheckpointer(db_path)
    run_id = "run-checkout-latency-001"

    # 1. Start workflow and simulate a crash mid-way through step 2.
    print("Phase 1: Starting fresh workflow instance (will crash on step 2)...")
    worker_a = CheckpointedWorkflow(checkpointer, use_live_llm=use_live_llm)
    try:
        worker_a.run(
            run_id=run_id,
            initial_state=WorkflowState(
                topic="Checkout Audit Queue Migration",
                audience="platform engineering leads",
            ),
            fail_on_step=2,
        )
    except SimulatedCrashError as exc:
        print(f"[CRASH] {exc}")
        print(
            f"        Worker A executed steps before crash: {worker_a.steps_run_in_this_instance}\n"
        )

    # 2. Restart with a brand-new workflow instance and the exact same run_id.
    print("Phase 2: Restarting in a BRAND-NEW workflow instance with the same run_id...")
    worker_b = CheckpointedWorkflow(checkpointer, use_live_llm=use_live_llm)
    final_state = worker_b.run(run_id=run_id)
    print(
        f"        Worker B executed ONLY remaining steps: {worker_b.steps_run_in_this_instance}"
    )
    assert "gather_facts" not in worker_b.steps_run_in_this_instance, (
        "Step 1 should NOT have been re-run!"
    )

    # 3. Inspect persisted snapshots in SQLite.
    print_checkpoint_history(checkpointer, run_id)
    print("\n--- Final Output from Recovered Run ---")
    print(final_state.final_brief)

    # 4. Time-travel debugging: fork from step 1 with a different audience and replay!
    print(
        "\nPhase 3: Time-travel replay — forking from step 1 into a new run_id "
        "with audience='security & compliance reviewers'..."
    )
    fork_id = "run-checkout-latency-001-security-fork"
    checkpointer.fork_run(
        source_run_id=run_id,
        from_step_index=1,
        new_run_id=fork_id,
        state_mutator=lambda s: WorkflowState(
            **{**asdict(s), "audience": "security & compliance reviewers"}
        ),
    )
    worker_c = CheckpointedWorkflow(checkpointer, use_live_llm=use_live_llm)
    fork_state = worker_c.run(run_id=fork_id)
    print_checkpoint_history(checkpointer, fork_id)
    print("\n--- Forked Replay Output (step 1 reused, steps 2-4 re-run) ---")
    print(fork_state.final_brief)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SQLite-backed checkpointed agent workflow demo."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the offline deterministic crash-recovery and time-travel demo.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use configured OpenAI-compatible LLM for step 2 drafting if available.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite checkpoint DB. Uses a clean temporary DB if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.db is not None:
        if args.db.exists():
            args.db.unlink()
        run_demo(args.db, use_live_llm=args.live and not args.demo)
        return

    with tempfile.TemporaryDirectory(prefix="day17_checkpoints_") as tmpdir:
        run_demo(Path(tmpdir) / "checkpoints.db", use_live_llm=args.live and not args.demo)


if __name__ == "__main__":
    main()
