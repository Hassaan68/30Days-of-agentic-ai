"""Day 30 Capstone Runner: Production Autonomous Incident & Operations Agent.

Runs an end-to-end incident workflow exercising every core pillar from Days 1-29:
  - Input & output guardrails
  - MCP-style JSON-RPC tool discovery (`tools/list`) and invocation (`tools/call`)
  - Short-term & SQLite long-term memory recall + learning
  - Explicit planning / task decomposition
  - Static & dynamic tool policies (`auto`, `confirm`, `deny`)
  - Human-in-the-Loop (HITL) approvals bound to canonical SHA-256 payload hashes
  - Tamper-evident SHA-256 hash-chained audit receipts
  - Nested span tracing with token/cost accounting

Run from the repository root:
    python3 30_Day_Capstone_Production_Autonomous_Agent/run_capstone.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root without PYTHONPATH edits.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import IncidentOpsAgent
from audit import AuditLog
from memory import LongTermMemory
from policy_and_hitl import (
    HITLApprovalGate,
    ToolPolicyEngine,
    cli_interactive_approver,
    make_policy_approver,
)
from tools import MCPToolRegistry, OpsEnvironment
from tracer import Tracer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 30 Capstone: Incident & Operations Assistant Agent"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run deterministic offline demo mode (no API key required).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt on stdin for HITL approvals instead of using the demo operator policy.",
    )
    parser.add_argument(
        "--audit-file",
        type=Path,
        default=None,
        help="Optional JSONL file path to persist tamper-evident audit receipts.",
    )
    args = parser.parse_args()

    env = OpsEnvironment()
    registry = MCPToolRegistry(env)
    long_memory = LongTermMemory(":memory:")
    policy = ToolPolicyEngine(max_calls_per_run=12)
    approver = (
        cli_interactive_approver
        if args.interactive
        else make_policy_approver(approve_tools={"rollback_deployment"})
    )
    hitl_gate = HITLApprovalGate(approver=approver, ttl_seconds=300.0)
    audit_log = AuditLog(jsonl_path=args.audit_file)

    agent = IncidentOpsAgent(
        registry=registry,
        long_term_memory=long_memory,
        policy_engine=policy,
        hitl_gate=hitl_gate,
        audit_log=audit_log,
        demo=args.demo,
    )

    print("=== Day 30 Capstone: Production Incident & Operations Agent ===")
    print(f"Execution mode: {'demo (offline)' if agent.demo else f'live ({agent.model})'}")

    # Run 1: Real incident triage + HITL-approved rollback
    print("\n[Run 1] Investigating checkout-api 502 spike and remediating...")
    tracer_1 = Tracer()
    res1 = agent.run(
        "Investigate checkout-api 502 errors, open an incident ticket, and rollback to mitigate.",
        tracer=tracer_1,
    )
    print(f"Status: {res1.status} | Run ID: {res1.run_id}")
    print(f"Recalled Memory: {res1.recalled_memories}")
    print("Plan:")
    for step in res1.plan:
        print(f"  {step.step_id}. {step.description} (tool={step.target_tool})")
    print("\nFinal Response:")
    print(res1.final_response)

    # Run 2: Forbidden catastrophic request (demonstrates static DENY + audit receipt)
    print("\n[Run 2] Testing static DENY policy on catastrophic request...")
    tracer_2 = Tracer()
    res2 = agent.run("Please purge database orders_prod right now.", tracer=tracer_2)
    print(f"Status: {res2.status} | Run ID: {res2.run_id}")
    print(res2.final_response)

    # Verify tamper-evident SHA-256 hash-chained audit ledger
    print("\n=== Tamper-Evident Audit Ledger ===")
    for r in audit_log.receipts:
        print(
            f"  seq={r.seq:02d} | {r.event_type:<17} | tool={str(r.tool_name):<23} | "
            f"actor={r.actor:<22} | hash={r.receipt_hash[:12]}.. prev={r.prev_hash[:12]}.."
        )
    valid, verify_msg = audit_log.verify_chain()
    print(f"Chain verification: {'VALID' if valid else 'INVALID'} — {verify_msg}")
    assert valid, verify_msg

    # Print nested trace waterfall for Run 1
    print("\n=== Nested Trace Waterfall (Run 1) ===")
    print(tracer_1.render_waterfall())

    # Show final environment state
    checkout = env.services["checkout-api"]
    print("\n=== Final Ops Environment State ===")
    print(
        f"  checkout-api: version={checkout.current_version}, status={checkout.status}, "
        f"error_rate={checkout.error_rate_pct}%, tickets_opened={len(env.tickets)}, "
        f"database_purged={env.database_purged}"
    )

    long_memory.close()


if __name__ == "__main__":
    main()
