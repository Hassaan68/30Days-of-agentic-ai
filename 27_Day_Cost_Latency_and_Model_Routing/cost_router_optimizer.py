"""Day 27: Cost, Latency & Model Routing Optimizer.

Demonstrates practical production techniques for cutting agent spend and
latency without sacrificing quality on hard tasks:
  1. Token cost math (input vs. output vs. prompt-cached input pricing).
  2. Complexity-based model routing (fast -> balanced -> frontier tiers).
  3. Exact-match TTL response caching with key normalization.
  4. Context and tool-output trimming to prevent quadratic token growth.
  5. Hard per-run token and USD budget circuit breakers (`BudgetGuard`).

Run from the repository root:
    python3 27_Day_Cost_Latency_and_Model_Routing/cost_router_optimizer.py --demo
    python3 27_Day_Cost_Latency_and_Model_Routing/cost_router_optimizer.py --demo --max-budget 0.005
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Pricing Table & Cost Math
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Pricing (USD per 1M tokens) and latency profile for a model tier."""

    tier: str
    model_id: str
    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float
    ttft_ms: float
    tokens_per_sec: float

    def calculate_cost(
        self,
        uncached_input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Compute exact USD cost for a single model call."""
        in_cost = (uncached_input_tokens / 1_000_000) * self.input_per_1m
        cached_cost = (cached_input_tokens / 1_000_000) * self.cached_input_per_1m
        out_cost = (output_tokens / 1_000_000) * self.output_per_1m
        return in_cost + cached_cost + out_cost

    def estimate_latency_ms(self, output_tokens: int) -> float:
        """Approximate end-to-end response latency (TTFT + generation time)."""
        return self.ttft_ms + (output_tokens / self.tokens_per_sec) * 1000.0


# Representative pricing figures (USD per 1M tokens) for illustration.
MODEL_CATALOG: dict[str, ModelSpec] = {
    "fast": ModelSpec(
        tier="fast",
        model_id="gpt-4o-mini",
        input_per_1m=0.15,
        cached_input_per_1m=0.075,
        output_per_1m=0.60,
        ttft_ms=180.0,
        tokens_per_sec=110.0,
    ),
    "balanced": ModelSpec(
        tier="balanced",
        model_id="gpt-4.1-mini",
        input_per_1m=0.40,
        cached_input_per_1m=0.10,
        output_per_1m=1.60,
        ttft_ms=310.0,
        tokens_per_sec=80.0,
    ),
    "frontier": ModelSpec(
        tier="frontier",
        model_id="gpt-4o",
        input_per_1m=2.50,
        cached_input_per_1m=1.25,
        output_per_1m=10.00,
        ttft_ms=650.0,
        tokens_per_sec=50.0,
    ),
}


def estimate_tokens(text: str) -> int:
    """Rough BPE token estimator (~4 characters per token for English text)."""
    return max(1, (len(text) + 3) // 4)


# ---------------------------------------------------------------------------
# Exact TTL Cache
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    value: str
    expires_at: float
    model_id: str


class ExactTTLCache:
    """In-memory exact-match cache with prompt normalization and TTL expiry."""

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 512) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(task_type: str, prompt: str) -> str:
        normalized = re.sub(r"\s+", " ", prompt.strip().lower())
        raw = f"{task_type}:{normalized}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, task_type: str, prompt: str) -> CacheEntry | None:
        key = self._make_key(task_type, prompt)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def set(self, task_type: str, prompt: str, value: str, model_id: str) -> None:
        if len(self._store) >= self.max_entries:
            # Evict the oldest inserted key (dict preserves insertion order in Python 3.7+).
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        key = self._make_key(task_type, prompt)
        self._store[key] = CacheEntry(
            value=value,
            expires_at=time.monotonic() + self.ttl_seconds,
            model_id=model_id,
        )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ---------------------------------------------------------------------------
# Context Trimming
# ---------------------------------------------------------------------------


def trim_tool_output(payload: str, max_chars: int = 1200) -> str:
    """Keep head and tail of verbose tool payloads so prompts do not balloon."""
    if len(payload) <= max_chars:
        return payload
    head = int(max_chars * 0.7)
    tail = max_chars - head
    omitted = len(payload) - max_chars
    return (
        f"{payload[:head]}\n"
        f"... [trimmed {omitted} chars; full artifact stored out-of-band] ...\n"
        f"{payload[-tail:]}"
    )


# ---------------------------------------------------------------------------
# Complexity-Based Model Router
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadItem:
    """One incoming request or agent sub-step in a benchmark."""

    request_id: str
    task_type: str  # "classify", "extract", "faq", "tool_step", "synthesis"
    prompt: str
    tool_output_blob: str = ""
    expected_output_tokens: int = 120
    cacheable: bool = True


class ComplexityRouter:
    """Routes each task to fast, balanced, or frontier models via lightweight signals."""

    FRONTIER_MARKERS = (
        "root cause",
        "trade-off",
        "architecture",
        "security audit",
        "postmortem",
        "migration plan",
        "multi-region",
    )
    BALANCED_MARKERS = (
        "compare",
        "summarize",
        "sql",
        "debug",
        "draft",
        "refund policy",
    )

    def select_model(self, item: WorkloadItem) -> ModelSpec:
        """Pick the cheapest model tier capable of handling the task reliably."""
        lower = item.prompt.lower()
        score = 0

        if item.task_type in ("classify", "extract", "faq"):
            score += 0
        elif item.task_type == "tool_step":
            score += 2
        elif item.task_type == "synthesis":
            score += 4

        if len(item.prompt) > 220 or len(item.tool_output_blob) > 1500:
            score += 2
        if any(m in lower for m in self.BALANCED_MARKERS):
            score += 2
        if any(m in lower for m in self.FRONTIER_MARKERS):
            score += 4

        if score <= 1:
            return MODEL_CATALOG["fast"]
        if score <= 4:
            return MODEL_CATALOG["balanced"]
        return MODEL_CATALOG["frontier"]


# ---------------------------------------------------------------------------
# Per-Run Budget Circuit Breaker
# ---------------------------------------------------------------------------


class BudgetExceededError(RuntimeError):
    """Raised before a model call that would exceed token or spend caps."""


@dataclass
class BudgetGuard:
    """Tracks cumulative spend and tokens across a run and halts overruns."""

    max_cost_usd: float
    max_total_tokens: int
    spent_usd: float = 0.0
    total_tokens: int = 0
    calls: int = 0

    def check_before_call(
        self,
        model: ModelSpec,
        projected_uncached_in: int,
        projected_cached_in: int,
        projected_out: int,
    ) -> None:
        projected_tokens = (
            self.total_tokens
            + projected_uncached_in
            + projected_cached_in
            + projected_out
        )
        projected_cost = self.spent_usd + model.calculate_cost(
            projected_uncached_in, projected_cached_in, projected_out
        )
        if projected_tokens > self.max_total_tokens:
            raise BudgetExceededError(
                f"Projected tokens ({projected_tokens:,}) exceed limit ({self.max_total_tokens:,})"
            )
        if projected_cost > self.max_cost_usd:
            raise BudgetExceededError(
                f"Projected spend (${projected_cost:.5f}) exceeds limit (${self.max_cost_usd:.5f})"
            )

    def record(self, tokens: int, cost_usd: float) -> None:
        self.total_tokens += tokens
        self.spent_usd += cost_usd
        self.calls += 1


# ---------------------------------------------------------------------------
# Benchmark Execution (Naive vs. Optimized)
# ---------------------------------------------------------------------------


STATIC_SYSTEM_PROMPT = (
    "You are an operations and support assistant. Follow company policy strictly. "
    "Tools available: lookup_order, query_metrics, query_logs, create_ticket, "
    "draft_postmortem, classify_intent, extract_invoice_fields. "
) * 28  # ~1,100 tokens of shared system instructions + tool schemas


def build_benchmark_workload() -> list[WorkloadItem]:
    """Create a realistic 18-request mix of easy, duplicate, medium, and hard work."""
    noisy_log_dump = (
        "[2026-11-23T14:01:02Z] INFO healthcheck ok status=200 latency=4ms\n" * 160
        + "[2026-11-23T14:04:19Z] ERROR db_pool exhausted waiting=5000ms service=checkout\n"
    )
    return [
        # Simple classification & extraction tasks (ideal for fast tier)
        WorkloadItem("req-01", "classify", "Classify intent: 'Where is my invoice?'", expected_output_tokens=20),
        WorkloadItem("req-02", "classify", "Classify intent: 'Cancel my subscription today'", expected_output_tokens=20),
        WorkloadItem("req-03", "classify", "Classify intent: 'API returns 429 rate limit'", expected_output_tokens=20),
        WorkloadItem("req-04", "extract", "Extract JSON {invoice_id, total}: Invoice INV-4410 Total $129.50", expected_output_tokens=45),
        WorkloadItem("req-05", "extract", "Extract JSON {invoice_id, total}: Invoice INV-4411 Total $89.00", expected_output_tokens=45),
        WorkloadItem("req-06", "faq", "What is our standard 30-day return window?", expected_output_tokens=70),
        # Repeated queries with whitespace/case differences (exact TTL cache hits)
        WorkloadItem("req-07", "faq", "  what is our standard 30-day return window? ", expected_output_tokens=70),
        WorkloadItem("req-08", "faq", "What is our standard 30-day return window?", expected_output_tokens=70),
        WorkloadItem("req-09", "classify", "classify intent: 'Where is my invoice?'", expected_output_tokens=20),
        WorkloadItem("req-10", "faq", "How do I rotate my workspace API key?", expected_output_tokens=85),
        WorkloadItem("req-11", "faq", "How do I rotate my workspace API key?", expected_output_tokens=85),
        # Medium tool-augmented tasks (balanced tier + trimmed logs)
        WorkloadItem(
            "req-12",
            "tool_step",
            "Summarize the checkout errors from these logs and suggest next debug step.",
            tool_output_blob=noisy_log_dump,
            expected_output_tokens=160,
            cacheable=False,
        ),
        WorkloadItem(
            "req-13",
            "tool_step",
            "Compare our enterprise and starter refund policy for damaged hardware.",
            expected_output_tokens=180,
        ),
        WorkloadItem(
            "req-14",
            "tool_step",
            "Write a read-only SQL query to count failed payments per hour over the last 24h.",
            expected_output_tokens=140,
        ),
        WorkloadItem(
            "req-15",
            "tool_step",
            "Debug why webhook retries are duplicating rows in the orders table.",
            tool_output_blob=noisy_log_dump,
            expected_output_tokens=200,
            cacheable=False,
        ),
        # High-stakes synthesis tasks (routed to frontier tier)
        WorkloadItem(
            "req-16",
            "synthesis",
            "Draft a customer-facing postmortem and root cause analysis for the checkout DB pool outage with concrete architectural trade-offs.",
            tool_output_blob=noisy_log_dump,
            expected_output_tokens=450,
            cacheable=False,
        ),
        WorkloadItem(
            "req-17",
            "synthesis",
            "Design a zero-downtime multi-region database migration plan and security audit checklist for moving 4TB of billing data.",
            expected_output_tokens=520,
            cacheable=False,
        ),
        WorkloadItem("req-18", "classify", "Classify intent: 'Need SOC2 report PDF'", expected_output_tokens=20),
    ]


@dataclass
class RunMetrics:
    strategy_name: str
    total_requests: int = 0
    cache_hits: int = 0
    total_input_tokens: int = 0
    cached_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    tier_counts: dict[str, int] = field(
        default_factory=lambda: {"fast": 0, "balanced": 0, "frontier": 0}
    )

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests else 0.0


def run_naive_strategy(workload: list[WorkloadItem]) -> RunMetrics:
    """Send every request directly to the frontier model with no cache or trimming."""
    metrics = RunMetrics("Naive (Single Frontier Model)")
    model = MODEL_CATALOG["frontier"]
    system_tokens = estimate_tokens(STATIC_SYSTEM_PROMPT)

    for item in workload:
        metrics.total_requests += 1
        user_tokens = estimate_tokens(item.prompt + "\n" + item.tool_output_blob)
        in_tokens = system_tokens + user_tokens
        out_tokens = item.expected_output_tokens

        cost = model.calculate_cost(
            uncached_input_tokens=in_tokens,
            cached_input_tokens=0,
            output_tokens=out_tokens,
        )
        metrics.total_input_tokens += in_tokens
        metrics.total_output_tokens += out_tokens
        metrics.total_cost_usd += cost
        metrics.total_latency_ms += model.estimate_latency_ms(out_tokens)
        metrics.tier_counts["frontier"] += 1

    return metrics


def run_optimized_strategy(
    workload: list[WorkloadItem],
    budget_usd: float,
) -> RunMetrics:
    """Combine complexity routing, exact TTL cache, prompt caching, and trimming."""
    metrics = RunMetrics("Optimized (Routed + Cached + Trimmed)")
    router = ComplexityRouter()
    cache = ExactTTLCache(ttl_seconds=300.0)
    guard = BudgetGuard(max_cost_usd=budget_usd, max_total_tokens=250_000)

    system_tokens = estimate_tokens(STATIC_SYSTEM_PROMPT)
    warmed_prefix_tiers: set[str] = set()

    for item in workload:
        metrics.total_requests += 1

        if item.cacheable:
            cached = cache.get(item.task_type, item.prompt)
            if cached is not None:
                metrics.cache_hits += 1
                metrics.total_latency_ms += 2.0  # local memory lookup
                continue

        model = router.select_model(item)
        trimmed_blob = trim_tool_output(item.tool_output_blob, max_chars=900)
        user_tokens = estimate_tokens(item.prompt + "\n" + trimmed_blob)

        # Prompt prefix caching applies once a tier has seen the static prefix.
        if model.tier in warmed_prefix_tiers:
            cached_in = system_tokens
            uncached_in = user_tokens
        else:
            cached_in = 0
            uncached_in = system_tokens + user_tokens
            warmed_prefix_tiers.add(model.tier)

        out_tokens = item.expected_output_tokens
        guard.check_before_call(model, uncached_in, cached_in, out_tokens)
        cost = model.calculate_cost(uncached_in, cached_in, out_tokens)
        guard.record(uncached_in + cached_in + out_tokens, cost)

        metrics.total_input_tokens += uncached_in + cached_in
        metrics.cached_input_tokens += cached_in
        metrics.total_output_tokens += out_tokens
        metrics.total_cost_usd += cost
        metrics.total_latency_ms += model.estimate_latency_ms(out_tokens)
        metrics.tier_counts[model.tier] += 1

        if item.cacheable:
            cache.set(
                item.task_type,
                item.prompt,
                value=f"[simulated answer for {item.request_id}]",
                model_id=model.model_id,
            )

    return metrics


def maybe_run_live_probe(prompt: str) -> None:
    """Optionally call an OpenAI-compatible API when not in --demo mode."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    if not api_key and not is_local:
        print("No OPENAI_API_KEY set; running offline benchmark (--demo).")
        return

    item = WorkloadItem("live-01", "classify", prompt, expected_output_tokens=60)
    selected = ComplexityRouter().select_model(item)
    target_model = model if is_local else selected.model_id
    print(f"\nLive routed call -> tier={selected.tier}, model={target_model}")

    body = json.dumps(
        {
            "model": target_model,
            "temperature": 0.0,
            "max_tokens": 80,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = data.get("usage", {})
        reply = data["choices"][0]["message"]["content"].strip()
        print(f"Live reply ({latency_ms:.0f}ms, usage={usage}): {reply}")
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        print(f"Live probe failed ({exc}); continuing with benchmark.")


def demo_budget_circuit_breaker() -> None:
    """Show BudgetGuard stopping a runaway agent loop before spend explodes."""
    print("\n=== BudgetGuard Circuit Breaker Demo ===")
    guard = BudgetGuard(max_cost_usd=0.015, max_total_tokens=20_000)
    model = MODEL_CATALOG["frontier"]

    for step in range(1, 20):
        # Simulate growing conversation context on every step of a stuck loop.
        in_tokens = 1500 + step * 800
        out_tokens = 350
        try:
            guard.check_before_call(model, in_tokens, 0, out_tokens)
            cost = model.calculate_cost(in_tokens, 0, out_tokens)
            guard.record(in_tokens + out_tokens, cost)
            print(
                f"  step {step:02d}: OK  | cumulative=${guard.spent_usd:.5f} "
                f"| tokens={guard.total_tokens:,}"
            )
        except BudgetExceededError as exc:
            print(f"  step {step:02d}: HALTED by BudgetGuard -> {exc}")
            break


def print_comparison(naive: RunMetrics, opt: RunMetrics) -> None:
    """Render side-by-side spend, latency, and token comparison."""
    cost_saving_pct = (1.0 - (opt.total_cost_usd / naive.total_cost_usd)) * 100.0
    latency_saving_pct = (1.0 - (opt.avg_latency_ms / naive.avg_latency_ms)) * 100.0
    token_saving_pct = (
        1.0
        - (
            (opt.total_input_tokens + opt.total_output_tokens)
            / (naive.total_input_tokens + naive.total_output_tokens)
        )
    ) * 100.0

    print("\n=== Benchmark: Naive Single Frontier vs. Routed + Cached + Trimmed ===")
    print(f"{'Metric':<28} {'Naive':>18} {'Optimized':>18} {'Delta':>14}")
    print("-" * 80)
    print(
        f"{'Total requests':<28} {naive.total_requests:>18} {opt.total_requests:>18} {'':>14}"
    )
    print(
        f"{'Exact cache hits':<28} {naive.cache_hits:>18} {opt.cache_hits:>18} "
        f"{f'{opt.cache_hits / opt.total_requests:.0%} hit':>14}"
    )
    print(
        f"{'Input tokens (total)':<28} {naive.total_input_tokens:>18,} "
        f"{opt.total_input_tokens:>18,} {'':>14}"
    )
    print(
        f"{'  - Prompt-cached input':<28} {naive.cached_input_tokens:>18,} "
        f"{opt.cached_input_tokens:>18,} {'':>14}"
    )
    print(
        f"{'Output tokens (total)':<28} {naive.total_output_tokens:>18,} "
        f"{opt.total_output_tokens:>18,} {'':>14}"
    )
    print(
        f"{'Avg latency / request':<28} {naive.avg_latency_ms:>15.0f} ms "
        f"{opt.avg_latency_ms:>15.0f} ms {f'-{latency_saving_pct:.1f}%':>14}"
    )
    print(
        f"{'Total USD spend':<28} {f'${naive.total_cost_usd:.5f}':>18} "
        f"{f'${opt.total_cost_usd:.5f}':>18} {f'-{cost_saving_pct:.1f}%':>14}"
    )
    print("-" * 80)
    print(
        f"Total token reduction: {token_saving_pct:.1f}% | "
        f"Optimized tier split (non-cached calls): {opt.tier_counts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 27 Cost, Latency & Model Routing")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the offline benchmark without making live API calls.",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=0.10,
        help="USD budget cap for the optimized benchmark run.",
    )
    args = parser.parse_args()

    if not args.demo:
        maybe_run_live_probe("Classify intent: 'Where is my monthly invoice?'")

    workload = build_benchmark_workload()
    naive = run_naive_strategy(workload)
    optimized = run_optimized_strategy(workload, budget_usd=args.max_budget)
    print_comparison(naive, optimized)
    demo_budget_circuit_breaker()


if __name__ == "__main__":
    main()
