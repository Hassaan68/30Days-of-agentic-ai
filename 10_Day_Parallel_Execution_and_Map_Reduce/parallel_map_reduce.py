"""Day 10: Parallel Execution and Map-Reduce.

Demonstrates:
1. Sequential vs. bounded concurrent execution using asyncio.gather + asyncio.Semaphore.
2. Per-task timeouts with asyncio.wait_for and partial failure handling via
   return_exceptions=True.
3. Map-Reduce aggregation that synthesizes structured map outputs while
   explicitly reporting coverage gaps from failed or timed-out chunks.

Run offline demo (no API key needed):
    python 10_Day_Parallel_Execution_and_Map_Reduce/parallel_map_reduce.py --demo

Run against a live OpenAI-compatible endpoint:
    export OPENAI_API_KEY="your-key"
    python 10_Day_Parallel_Execution_and_Map_Reduce/parallel_map_reduce.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Typed Map-Reduce data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """One unit of input work for the Map phase."""

    chunk_id: str
    service: str
    text: str
    simulated_delay_s: float = 0.25
    simulated_failure: str | None = None  # None | "timeout" | "upstream_503"


@dataclass
class MapOutput:
    """Structured extraction produced for a single chunk."""

    chunk_id: str
    service: str
    ok: bool
    severity: str = "unknown"
    error_count: int = 0
    summary: str = ""
    action_item: str = ""
    latency_s: float = 0.0
    error_message: str | None = None


@dataclass
class MapReduceReport:
    """Aggregate report from a full Map-Reduce run."""

    total_chunks: int
    succeeded: int
    failed: int
    coverage_pct: float
    map_duration_s: float
    reduce_duration_s: float
    total_duration_s: float
    map_results: list[MapOutput]
    final_synthesis: str


# ---------------------------------------------------------------------------
# Pluggable async backends (offline demo + live OpenAI-compatible via to_thread)
# ---------------------------------------------------------------------------

class AsyncMapReduceBackend(Protocol):
    """Interface for Map and Reduce model calls."""

    @property
    def label(self) -> str:
        ...

    async def map_chunk(self, chunk: DocumentChunk) -> MapOutput:
        ...

    async def reduce_outputs(
        self,
        succeeded: list[MapOutput],
        failed: list[MapOutput],
    ) -> str:
        ...


class DemoAsyncBackend:
    """Offline async backend that simulates realistic network latency and failures."""

    _ERR_COUNT_RE = re.compile(r"errors=(\d+)")
    _SEV_RE = re.compile(r"severity=(sev[1-4]|ok)", re.IGNORECASE)

    @property
    def label(self) -> str:
        return "offline async demo backend"

    async def map_chunk(self, chunk: DocumentChunk) -> MapOutput:
        t0 = time.perf_counter()
        if chunk.simulated_failure == "timeout":
            # Sleep much longer than per-task timeout so asyncio.wait_for cancels it.
            await asyncio.sleep(5.0)
        else:
            await asyncio.sleep(chunk.simulated_delay_s)

        if chunk.simulated_failure == "upstream_503":
            raise RuntimeError(f"HTTP 503 Service Unavailable while analyzing {chunk.service}")

        sev_match = self._SEV_RE.search(chunk.text)
        err_match = self._ERR_COUNT_RE.search(chunk.text)
        severity = sev_match.group(1).lower() if sev_match else "sev3"
        error_count = int(err_match.group(1)) if err_match else 0

        # Extract the concise finding line from the chunk text.
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        finding = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("finding:")), lines[0])
        action = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("action:")), "Monitor metrics")

        return MapOutput(
            chunk_id=chunk.chunk_id,
            service=chunk.service,
            ok=True,
            severity=severity,
            error_count=error_count,
            summary=finding,
            action_item=action,
            latency_s=time.perf_counter() - t0,
        )

    async def reduce_outputs(
        self,
        succeeded: list[MapOutput],
        failed: list[MapOutput],
    ) -> str:
        await asyncio.sleep(0.08)  # Simulate one short Reduce call.
        sev_order = {"sev1": 1, "sev2": 2, "sev3": 3, "sev4": 4, "ok": 5}
        ranked = sorted(succeeded, key=lambda item: (sev_order.get(item.severity, 9), -item.error_count))

        total_errors = sum(item.error_count for item in ranked)
        lines = [
            f"Executive Brief ({len(succeeded)}/{len(succeeded) + len(failed)} services analyzed, "
            f"{total_errors:,} total errors observed):",
            "",
            "Priority Findings:",
        ]
        for item in ranked:
            lines.append(
                f"  - [{item.severity.upper()}] {item.service} ({item.error_count} errors): "
                f"{item.summary} -> Action: {item.action_item}"
            )

        if failed:
            lines.append("")
            lines.append("Coverage Gaps (excluded from synthesis — manual check required):")
            for miss in failed:
                lines.append(f"  - {miss.service} ({miss.chunk_id}): {miss.error_message}")

        return "\n".join(lines)


class OpenAIAsyncBackend:
    """Live OpenAI-compatible async backend using stdlib urllib inside asyncio.to_thread."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: float = 25.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    @property
    def label(self) -> str:
        return f"live OpenAI-compatible ({self.model})"

    def _sync_chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
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
                parsed = json.loads(resp.read().decode("utf-8"))
            return parsed["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            if json_mode and exc.code == 400:
                return self._sync_chat(system_prompt, user_prompt, json_mode=False)
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error: {exc.reason}") from exc

    async def map_chunk(self, chunk: DocumentChunk) -> MapOutput:
        t0 = time.perf_counter()
        if chunk.simulated_failure == "timeout":
            await asyncio.sleep(5.0)
        if chunk.simulated_failure == "upstream_503":
            raise RuntimeError(f"Simulated upstream 503 on {chunk.service}")

        system = (
            "Extract incident details from the service log snippet. Return ONLY JSON with keys: "
            "severity (sev1|sev2|sev3|sev4|ok), error_count (int), summary (1 sentence), "
            "action_item (short concrete next step)."
        )
        raw = await asyncio.to_thread(self._sync_chat, system, chunk.text, True)
        data = json.loads(raw)
        return MapOutput(
            chunk_id=chunk.chunk_id,
            service=chunk.service,
            ok=True,
            severity=str(data.get("severity", "sev3")).lower(),
            error_count=int(data.get("error_count", 0)),
            summary=str(data.get("summary", "")).strip(),
            action_item=str(data.get("action_item", "")).strip(),
            latency_s=time.perf_counter() - t0,
        )

    async def reduce_outputs(
        self,
        succeeded: list[MapOutput],
        failed: list[MapOutput],
    ) -> str:
        payload = {
            "analyzed_services": [
                {
                    "service": m.service,
                    "severity": m.severity,
                    "error_count": m.error_count,
                    "summary": m.summary,
                    "action_item": m.action_item,
                }
                for m in succeeded
            ],
            "failed_chunks": [
                {"service": m.service, "chunk_id": m.chunk_id, "error": m.error_message}
                for m in failed
            ],
        }
        system = (
            "Synthesize the structured service summaries into a concise engineering incident brief "
            "ordered by severity. Explicitly list any failed_chunks under a 'Coverage Gaps' section "
            "without inventing data for them."
        )
        return await asyncio.to_thread(self._sync_chat, system, json.dumps(payload, indent=2), False)


# ---------------------------------------------------------------------------
# Core Map-Reduce execution engine (sequential vs. bounded async parallel)
# ---------------------------------------------------------------------------

async def _run_one_bounded(
    chunk: DocumentChunk,
    backend: AsyncMapReduceBackend,
    semaphore: asyncio.Semaphore,
    timeout_s: float,
) -> MapOutput:
    """Acquire concurrency slot and enforce per-task timeout."""
    async with semaphore:
        return await asyncio.wait_for(backend.map_chunk(chunk), timeout=timeout_s)


async def run_sequential_map(
    chunks: list[DocumentChunk],
    backend: AsyncMapReduceBackend,
    timeout_s: float = 10.0,
) -> tuple[list[MapOutput], float]:
    """Run Map tasks one by one to establish a sequential baseline."""
    t0 = time.perf_counter()
    results: list[MapOutput] = []
    for chunk in chunks:
        try:
            res = await asyncio.wait_for(backend.map_chunk(chunk), timeout=timeout_s)
            results.append(res)
        except TimeoutError:
            results.append(
                MapOutput(
                    chunk_id=chunk.chunk_id,
                    service=chunk.service,
                    ok=False,
                    error_message=f"Timed out after {timeout_s:.2f}s",
                )
            )
        except Exception as exc:
            results.append(
                MapOutput(
                    chunk_id=chunk.chunk_id,
                    service=chunk.service,
                    ok=False,
                    error_message=str(exc),
                )
            )
    return results, time.perf_counter() - t0


async def run_parallel_map_reduce(
    chunks: list[DocumentChunk],
    backend: AsyncMapReduceBackend,
    max_concurrency: int = 3,
    per_task_timeout_s: float = 0.50,
    min_coverage: float = 0.50,
) -> MapReduceReport:
    """Run bounded concurrent Map phase with return_exceptions=True, then Reduce."""
    total_t0 = time.perf_counter()
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    # Map phase: launch all tasks, bounded by semaphore, preserving partial results.
    map_t0 = time.perf_counter()
    tasks = [
        _run_one_bounded(chunk, backend, semaphore, per_task_timeout_s)
        for chunk in chunks
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    map_duration = time.perf_counter() - map_t0

    map_outputs: list[MapOutput] = []
    for chunk, raw in zip(chunks, raw_results):
        if isinstance(raw, MapOutput):
            map_outputs.append(raw)
        elif isinstance(raw, TimeoutError):
            map_outputs.append(
                MapOutput(
                    chunk_id=chunk.chunk_id,
                    service=chunk.service,
                    ok=False,
                    error_message=f"Timed out after {per_task_timeout_s:.2f}s",
                )
            )
        elif isinstance(raw, BaseException):
            map_outputs.append(
                MapOutput(
                    chunk_id=chunk.chunk_id,
                    service=chunk.service,
                    ok=False,
                    error_message=f"{type(raw).__name__}: {raw}",
                )
            )

    succeeded = [m for m in map_outputs if m.ok]
    failed = [m for m in map_outputs if not m.ok]
    coverage = (len(succeeded) / len(chunks)) if chunks else 0.0

    # Reduce phase: synthesize only if minimum coverage threshold is met.
    reduce_t0 = time.perf_counter()
    if coverage < min_coverage:
        synthesis = (
            f"Aborted Reduce step: only {len(succeeded)}/{len(chunks)} chunks succeeded "
            f"({coverage * 100:.1f}% < required {min_coverage * 100:.0f}% coverage)."
        )
    else:
        synthesis = await backend.reduce_outputs(succeeded, failed)
    reduce_duration = time.perf_counter() - reduce_t0

    return MapReduceReport(
        total_chunks=len(chunks),
        succeeded=len(succeeded),
        failed=len(failed),
        coverage_pct=coverage * 100.0,
        map_duration_s=map_duration,
        reduce_duration_s=reduce_duration,
        total_duration_s=time.perf_counter() - total_t0,
        map_results=map_outputs,
        final_synthesis=synthesis,
    )


# ---------------------------------------------------------------------------
# Sample incident log dataset & CLI runner
# ---------------------------------------------------------------------------

def build_sample_chunks(inject_failures: bool = False) -> list[DocumentChunk]:
    """Create 6 realistic service log chunks for benchmarking and resilience testing."""
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            service="checkout-api",
            simulated_delay_s=0.25,
            text=(
                "service=checkout-api severity=sev1 errors=4182\n"
                "finding: ConnectionPoolExhausted on db-eu-west-1 after DEPLOY-772.\n"
                "action: Roll back DEPLOY-772 and increase pool wait timeout alerts."
            ),
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            service="payments-worker",
            simulated_delay_s=0.22,
            text=(
                "service=payments-worker severity=sev2 errors=319\n"
                "finding: Stripe webhook retries spiked due to downstream checkout 502s.\n"
                "action: Replay dead-letter queue once checkout-api stabilizes."
            ),
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            service="auth-service",
            simulated_delay_s=0.20,
            text=(
                "service=auth-service severity=ok errors=0\n"
                "finding: Token issuance and p95 latency (28ms) remained healthy.\n"
                "action: No action required."
            ),
        ),
        DocumentChunk(
            chunk_id="chunk_4",
            service="search-indexer",
            simulated_delay_s=0.26,
            simulated_failure="timeout" if inject_failures else None,
            text=(
                "service=search-indexer severity=sev3 errors=41\n"
                "finding: Indexing lag rose to 4 minutes during bulk catalog refresh.\n"
                "action: Throttle background reindex batch size."
            ),
        ),
        DocumentChunk(
            chunk_id="chunk_5",
            service="notification-svc",
            simulated_delay_s=0.24,
            simulated_failure="upstream_503" if inject_failures else None,
            text=(
                "service=notification-svc severity=sev3 errors=18\n"
                "finding: Delayed order confirmation emails queued in retry bucket.\n"
                "action: Confirm retry worker drained queue."
            ),
        ),
        DocumentChunk(
            chunk_id="chunk_6",
            service="billing-cron",
            simulated_delay_s=0.23,
            text=(
                "service=billing-cron severity=ok errors=0\n"
                "finding: Nightly invoice reconciliation completed cleanly.\n"
                "action: No action required."
            ),
        ),
    ]


def _should_use_demo(force_demo: bool) -> bool:
    if force_demo or os.getenv("AGENT_DEMO_MODE", "").strip() == "1":
        return True
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    return not api_key and not is_local


async def async_main(args: argparse.Namespace) -> int:
    use_demo = _should_use_demo(args.demo)
    if use_demo:
        backend: AsyncMapReduceBackend = DemoAsyncBackend()
        timeout_s = args.timeout
    else:
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        backend = OpenAIAsyncBackend(api_key=api_key, base_url=base_url, model=model)
        timeout_s = max(args.timeout, 25.0)

    print(f"=== Day 10: Parallel Execution & Map-Reduce ({backend.label}) ===\n")

    # Part 1: Sequential vs. bounded concurrent timing benchmark on healthy chunks.
    healthy_chunks = build_sample_chunks(inject_failures=False)
    print(f"1) Timing benchmark across {len(healthy_chunks)} healthy chunks (max_concurrency={args.concurrency}):")
    _, seq_time = await run_sequential_map(healthy_chunks, backend, timeout_s=timeout_s)
    par_report = await run_parallel_map_reduce(
        healthy_chunks,
        backend,
        max_concurrency=args.concurrency,
        per_task_timeout_s=timeout_s,
    )
    speedup = seq_time / par_report.map_duration_s if par_report.map_duration_s > 0 else 1.0
    print(f"   Sequential Map time:                {seq_time:.3f}s")
    print(f"   Parallel Map time (Semaphore={args.concurrency}):    {par_report.map_duration_s:.3f}s ({speedup:.2f}x faster)")
    print(f"   Reduce synthesis time:              {par_report.reduce_duration_s:.3f}s")

    # Part 2: Partial failures & timeout resilience with return_exceptions=True.
    print("\n2) Map-Reduce with injected timeout (chunk_4) and HTTP 503 failure (chunk_5):")
    faulty_chunks = build_sample_chunks(inject_failures=True)
    resilient_timeout = 0.45 if use_demo else timeout_s
    resilient_report = await run_parallel_map_reduce(
        faulty_chunks,
        backend,
        max_concurrency=args.concurrency,
        per_task_timeout_s=resilient_timeout,
        min_coverage=0.50,
    )
    print(
        f"   Succeeded: {resilient_report.succeeded}/{resilient_report.total_chunks} "
        f"({resilient_report.coverage_pct:.1f}% coverage) | "
        f"Failed: {resilient_report.failed} | "
        f"Total time: {resilient_report.total_duration_s:.3f}s"
    )
    print("\n   Per-chunk Map statuses:")
    for item in resilient_report.map_results:
        status = f"OK ({item.severity}, {item.latency_s:.2f}s)" if item.ok else f"FAILED ({item.error_message})"
        print(f"     - {item.chunk_id} [{item.service:16s}]: {status}")

    print("\n3) Final Reduced Synthesis:")
    print("-" * 72)
    print(resilient_report.final_synthesis)
    print("-" * 72)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 10: Async parallel execution & Map-Reduce")
    parser.add_argument("--demo", action="store_true", help="Run offline async benchmark & demo")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent Map tasks allowed by asyncio.Semaphore (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.60,
        help="Per-task timeout in seconds for demo mode (default: 0.60).",
    )
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
