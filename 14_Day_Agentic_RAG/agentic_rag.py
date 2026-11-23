"""Day 14: Agentic RAG (Retrieval Gating, Relevance Grading, Query Rewriting & Grounded Citations).

Demonstrates how Agentic RAG improves on naive static RAG:
- Decides whether retrieval is needed at all before touching the index.
- Retrieves candidate chunks using Okapi BM25 + token-overlap scoring (pure stdlib).
- Grades retrieved chunks for actual relevance before letting them into the prompt.
- Rewrites vague queries and retries retrieval when initial context is weak.
- Synthesizes answers with verified inline citations ([KB-XX]) or abstains cleanly
  when no supporting evidence exists.

Usage:
    python3 14_Day_Agentic_RAG/agentic_rag.py --demo
    python3 14_Day_Agentic_RAG/agentic_rag.py --demo --query "How long do old API keys stay valid after rotation?"
    python3 14_Day_Agentic_RAG/agentic_rag.py --query "What happens if our server is down when you ping us with events?"
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Mini internal engineering knowledge base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    title: str
    section: str
    tags: tuple[str, ...]
    content: str


KNOWLEDGE_BASE: list[DocumentChunk] = [
    DocumentChunk(
        doc_id="KB-01",
        title="API Rate Limits and Burst Quotas",
        section="Platform API",
        tags=("rate-limit", "429", "quota", "throttle", "retry-after"),
        content=(
            "Production API keys allow 120 requests per second with a short burst allowance of 200 requests. "
            "Exceeding the quota returns HTTP 429 Too Many Requests with a `Retry-After` header in seconds. "
            "Clients should honor `Retry-After` and apply jittered backoff."
        ),
    ),
    DocumentChunk(
        doc_id="KB-02",
        title="Webhook Delivery Retries and Dead-Letter Queue",
        section="Webhooks",
        tags=("webhook", "retries", "backoff", "dead-letter", "idempotency", "delivery"),
        content=(
            "When a customer webhook endpoint times out (>10s) or returns a non-2xx HTTP status, "
            "the platform retries delivery up to 5 times with exponential backoff: 1 minute, 5 minutes, "
            "15 minutes, 1 hour, and 6 hours. Events that still fail are placed in the Dead-Letter Queue (DLQ) "
            "for 24 hours where operators can manually replay them. Every delivery includes an `Idempotency-Key` header."
        ),
    ),
    DocumentChunk(
        doc_id="KB-03",
        title="API Key Rotation and Grace Windows",
        section="Security",
        tags=("api-key", "rotation", "secret", "grace-period", "auth"),
        content=(
            "Calling `POST /v2/keys/rotate` issues a new secret key while keeping the previous key active "
            "for a 48-hour grace period so zero-downtime rollouts can finish across worker fleets. "
            "For compromised keys, pass `{\"revoke_immediately\": true}` to skip the grace window."
        ),
    ),
    DocumentChunk(
        doc_id="KB-04",
        title="Production Deployment Freeze Windows",
        section="Release Engineering",
        tags=("deploy", "freeze", "release", "friday", "change-management"),
        content=(
            "Standard production deploys are frozen every Friday after 14:00 UTC through Monday 06:00 UTC, "
            "and during major public holidays. Emergency SEV-1 hotfixes during a freeze require sign-off "
            "from the on-call Incident Commander and VP of Engineering."
        ),
    ),
    DocumentChunk(
        doc_id="KB-05",
        title="Incident Severity Levels and Response SLAs",
        section="On-Call Operations",
        tags=("sev1", "sev2", "incident", "sla", "pager"),
        content=(
            "SEV-1 (full outage or data loss) requires a 15-minute acknowledgement SLA and status page updates "
            "every 30 minutes. SEV-2 (partial degradation affecting >5% of traffic) carries a 1-hour response SLA. "
            "SEV-3 (minor bug with workaround) is triaged on the next business day."
        ),
    ),
    DocumentChunk(
        doc_id="KB-06",
        title="Subscription Billing Proration Rules",
        section="Billing",
        tags=("billing", "proration", "upgrade", "downgrade", "invoice"),
        content=(
            "Mid-cycle plan upgrades take effect immediately and charge a daily prorated difference on the next invoice. "
            "Plan downgrades remain on the current tier until the end of the active billing period and do not issue "
            "partial-month cash refunds."
        ),
    ),
    DocumentChunk(
        doc_id="KB-07",
        title="Database Connection Pooling Standards",
        section="Infrastructure",
        tags=("postgres", "pgbouncer", "pooling", "connections"),
        content=(
            "All stateless API pods connect through PgBouncer in transaction pooling mode. "
            "Each worker pod is capped at 20 pooled connections to avoid exhausting PostgreSQL max_connections "
            "during traffic spikes."
        ),
    ),
    DocumentChunk(
        doc_id="KB-08",
        title="Sandbox Environment Limits",
        section="Developer Experience",
        tags=("sandbox", "testing", "staging", "keys"),
        content=(
            "Sandbox environments use isolated signing secrets and synthetic payment tokens. "
            "Sandbox rate limits are capped at 30 requests per minute and webhooks do not use the production DLQ."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Pure-stdlib BM25 + Token Overlap Retriever
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "happens", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "our",
    "that", "the", "their", "them", "they", "this", "to", "us", "we", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [tok for tok in raw_tokens if len(tok) > 1 and tok not in _STOPWORDS]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    bm25_score: float
    overlap_score: float
    combined_score: float


class HybridLexicalRetriever:
    """Combines Okapi BM25 with normalized query-token overlap over DocumentChunks."""

    def __init__(self, chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: Counter[str] = Counter()

        for chunk in chunks:
            # Weight title and tags more heavily than body text
            weighted_text = f"{chunk.title} {chunk.title} {' '.join(chunk.tags)} {chunk.content}"
            tokens = tokenize(weighted_text)
            self._doc_tokens.append(tokens)
            for unique_tok in set(tokens):
                self._doc_freqs[unique_tok] += 1

        self._doc_lengths = [len(toks) for toks in self._doc_tokens]
        self._avg_dl = sum(self._doc_lengths) / max(len(self._doc_lengths), 1)
        self._n_docs = len(chunks)

    def _idf(self, term: str) -> float:
        df = self._doc_freqs.get(term, 0)
        return math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_unique = set(q_tokens)

        scored: list[RetrievedChunk] = []
        for idx, chunk in enumerate(self.chunks):
            doc_toks = self._doc_tokens[idx]
            tf_counts = Counter(doc_toks)
            dl = self._doc_lengths[idx]

            bm25 = 0.0
            for term in q_tokens:
                tf = tf_counts.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self._avg_dl, 1.0)))
                bm25 += idf * ((tf * (self.k1 + 1.0)) / denom)

            overlap = len(q_unique & set(doc_toks)) / len(q_unique)
            # Blend BM25 (scaled) with lexical coverage
            combined = round(bm25 + 1.5 * overlap, 4)
            if combined > 0:
                scored.append(
                    RetrievedChunk(
                        chunk=chunk,
                        bm25_score=round(bm25, 4),
                        overlap_score=round(overlap, 4),
                        combined_score=combined,
                    )
                )

        scored.sort(key=lambda item: item.combined_score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Agentic RAG trace & decision structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradedChunk:
    retrieved: RetrievedChunk
    relevant: bool
    confidence: float
    reason: str


@dataclass
class RetrievalAttempt:
    attempt_number: int
    search_query: str
    candidates: list[RetrievedChunk]
    graded: list[GradedChunk]


@dataclass
class AgenticRAGResult:
    original_query: str
    used_retrieval: bool
    gate_reason: str
    attempts: list[RetrievalAttempt] = field(default_factory=list)
    cited_doc_ids: list[str] = field(default_factory=list)
    answer: str = ""
    abstained: bool = False
    mode: str = "demo"


# ---------------------------------------------------------------------------
# Safe arithmetic helper (for non-retrieval math questions)
# ---------------------------------------------------------------------------

_SAFE_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _try_simple_math(query: str) -> str | None:
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", query, re.IGNORECASE)
    if pct_match:
        pct = float(pct_match.group(1))
        base = float(pct_match.group(2))
        val = (pct / 100.0) * base
        return f"{pct:g}% of {base:g} is {val:g}."

    expr_match = re.fullmatch(r"\s*(?:what is\s+)?([0-9\.\+\-\*\/\(\)\s]+)\??\s*", query, re.IGNORECASE)
    if expr_match:
        expr = expr_match.group(1).strip()
        try:
            tree = ast.parse(expr, mode="eval")
            value = _eval_ast(tree.body)
            return f"{expr} = {value:g}."
        except Exception:
            return None
    return None


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError("Unsupported math expression")


# ---------------------------------------------------------------------------
# Backends (Live OpenAI-compatible HTTP + Deterministic Demo)
# ---------------------------------------------------------------------------


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Expected JSON object, got: {raw_text[:200]}")
    return json.loads(cleaned[start : end + 1])


class OpenAIRAGBackend:
    """OpenAI-compatible backend for routing, relevance grading, rewriting, and synthesis."""

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
        return str(data["choices"][0]["message"]["content"]).strip()

    def decide_retrieval(self, query: str) -> tuple[bool, str]:
        system = (
            "You are a RAG router for an internal engineering knowledge base (API limits, webhooks, "
            "key rotation, deploy freezes, incident SLAs, billing proration, DB pooling, sandbox).\n"
            'Return ONLY JSON: {"needs_retrieval": true|false, "reason": "..."}.\n'
            "Set needs_retrieval=false for pure arithmetic, greetings, or generic programming definitions."
        )
        data = _extract_json_object(self._chat(system, query))
        return bool(data["needs_retrieval"]), str(data.get("reason", ""))

    def direct_answer(self, query: str) -> str:
        return self._chat("Answer concisely and accurately without inventing company policies.", query)

    def grade_chunks(self, query: str, candidates: list[RetrievedChunk]) -> list[GradedChunk]:
        if not candidates:
            return []
        system = (
            "Grade each candidate document chunk for whether it directly helps answer the user query.\n"
            'Return ONLY JSON: {"grades": [{"doc_id": "KB-01", "relevant": true|false, "confidence": 0.0-1.0, "reason": "..."}]}'
        )
        payload = {
            "query": query,
            "candidates": [
                {"doc_id": c.chunk.doc_id, "title": c.chunk.title, "content": c.chunk.content}
                for c in candidates
            ],
        }
        data = _extract_json_object(self._chat(system, json.dumps(payload, indent=2)))
        grade_by_id = {g["doc_id"]: g for g in data.get("grades", [])}
        graded: list[GradedChunk] = []
        for cand in candidates:
            g = grade_by_id.get(cand.chunk.doc_id, {})
            graded.append(
                GradedChunk(
                    retrieved=cand,
                    relevant=bool(g.get("relevant", False)),
                    confidence=float(g.get("confidence", 0.0)),
                    reason=str(g.get("reason", "No grader note.")),
                )
            )
        return graded

    def rewrite_query(self, original_query: str, failed_query: str, attempt: int) -> str:
        system = (
            "Rewrite the user's question into a concise keyword-rich technical search query for an internal "
            "engineering knowledge base (webhooks, retries, DLQ, API keys, rate limits, deploys, SLAs, billing). "
            'Return ONLY JSON: {"rewritten_query": "..."}'
        )
        user = json.dumps(
            {
                "original_query": original_query,
                "previous_failed_query": failed_query,
                "attempt": attempt,
            }
        )
        data = _extract_json_object(self._chat(system, user))
        return str(data["rewritten_query"]).strip()

    def synthesize(self, original_query: str, relevant_chunks: list[GradedChunk]) -> str:
        system = (
            "Answer the user question using ONLY the provided knowledge-base chunks. "
            "Cite every factual claim inline using the exact doc ID in square brackets, e.g., [KB-02]. "
            "Do not add facts that are not in the chunks."
        )
        context = [
            {"doc_id": g.retrieved.chunk.doc_id, "title": g.retrieved.chunk.title, "content": g.retrieved.chunk.content}
            for g in relevant_chunks
        ]
        user = json.dumps({"question": original_query, "verified_chunks": context}, indent=2)
        return self._chat(system, user)


class DemoRAGBackend:
    """Deterministic offline backend demonstrating gating, grading, query rewrite, and citation."""

    def decide_retrieval(self, query: str) -> tuple[bool, str]:
        if _try_simple_math(query) is not None:
            return False, "Pure arithmetic calculation; internal knowledge base lookup is unnecessary."
        lowered = query.strip().lower()
        if lowered in {"hi", "hello", "hey", "thanks", "thank you"}:
            return False, "Simple greeting; no knowledge base retrieval needed."
        return True, "Query asks about operational policies or platform behavior covered by internal docs."

    def direct_answer(self, query: str) -> str:
        math_ans = _try_simple_math(query)
        if math_ans is not None:
            return math_ans
        return "Hello! Ask me any question about platform API limits, webhooks, deployments, or billing."

    def grade_chunks(self, query: str, candidates: list[RetrievedChunk]) -> list[GradedChunk]:
        q_tokens = set(tokenize(query))
        graded: list[GradedChunk] = []

        for cand in candidates:
            chunk = cand.chunk
            chunk_terms = set(tokenize(f"{chunk.title} {' '.join(chunk.tags)} {chunk.content}"))
            matched_terms = sorted(q_tokens & chunk_terms)

            # Require meaningful semantic/keyword alignment, not just a stray word like "events"
            strong_tag_hit = any(tok in set(chunk.tags) for tok in q_tokens)
            is_relevant = (
                (cand.bm25_score >= 2.2 and cand.overlap_score >= 0.25)
                or (strong_tag_hit and cand.bm25_score >= 1.4)
            )
            confidence = min(0.98, round(0.35 + 0.15 * cand.bm25_score + 0.35 * cand.overlap_score, 2))
            if is_relevant:
                reason = f"Matches key domain terms ({', '.join(matched_terms[:4])}); directly answers query."
            else:
                reason = (
                    f"Weak match (bm25={cand.bm25_score:.2f}, overlap={cand.overlap_score:.2f}, "
                    f"matched={matched_terms or ['none']}); insufficient to answer safely."
                )
            graded.append(
                GradedChunk(
                    retrieved=cand,
                    relevant=is_relevant,
                    confidence=confidence if is_relevant else round(confidence * 0.4, 2),
                    reason=reason,
                )
            )
        return graded

    def rewrite_query(self, original_query: str, failed_query: str, attempt: int) -> str:
        low = original_query.lower()
        if any(w in low for w in ("ping", "down", "events", "endpoint", "callback", "webhook")):
            return "webhook delivery retries exponential backoff dead-letter queue"
        if any(w in low for w in ("rotate", "old key", "secret", "expire", "token")):
            return "api-key rotation grace-period revoke_immediately"
        if any(w in low for w in ("friday", "deploy", "ship", "weekend", "holiday")):
            return "deploy freeze friday release window hotfix"
        if any(w in low for w in ("429", "too many", "throttle", "burst", "limit")):
            return "rate-limit 429 burst quota retry-after"
        if any(w in low for w in ("upgrade", "downgrade", "charge", "refund", "plan")):
            return "billing proration upgrade downgrade invoice"
        # Generic fallback expansion for unknown topics
        core = " ".join(tokenize(original_query))
        return f"{core} policy handbook" if attempt == 1 else f"{core} documentation"

    def synthesize(self, original_query: str, relevant_chunks: list[GradedChunk]) -> str:
        top = relevant_chunks[0].retrieved.chunk
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top.content) if s.strip()]
        cited_body = " ".join(f"{sent} [{top.doc_id}]" for sent in sentences)
        if len(relevant_chunks) > 1:
            second = relevant_chunks[1].retrieved.chunk
            first_sent = re.split(r"(?<=[.!?])\s+", second.content.strip())[0]
            cited_body += f" Additionally, {first_sent[0].lower() + first_sent[1:]} [{second.doc_id}]"
        return cited_body


# ---------------------------------------------------------------------------
# Agentic RAG controller
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(KB-\d+)\]")


def run_agentic_rag(
    query: str,
    retriever: HybridLexicalRetriever,
    backend: OpenAIRAGBackend | DemoRAGBackend,
    top_k: int = 3,
    max_retries: int = 2,
    mode: str = "demo",
) -> AgenticRAGResult:
    """Execute Retrieve-Gate -> Search -> Grade -> Rewrite/Retry -> Cited Synthesis."""
    needs_retrieval, gate_reason = backend.decide_retrieval(query)
    if not needs_retrieval:
        return AgenticRAGResult(
            original_query=query,
            used_retrieval=False,
            gate_reason=gate_reason,
            answer=backend.direct_answer(query),
            abstained=False,
            mode=mode,
        )

    attempts: list[RetrievalAttempt] = []
    current_query = query
    verified_chunks: list[GradedChunk] = []

    for attempt_num in range(1, max_retries + 2):
        candidates = retriever.search(current_query, top_k=top_k)
        graded = backend.grade_chunks(current_query, candidates)
        attempts.append(
            RetrievalAttempt(
                attempt_number=attempt_num,
                search_query=current_query,
                candidates=candidates,
                graded=graded,
            )
        )

        verified_chunks = [g for g in graded if g.relevant]
        if verified_chunks:
            break

        if attempt_num <= max_retries:
            rewritten = backend.rewrite_query(query, current_query, attempt=attempt_num)
            if rewritten.strip().lower() == current_query.strip().lower():
                break
            current_query = rewritten

    if not verified_chunks:
        return AgenticRAGResult(
            original_query=query,
            used_retrieval=True,
            gate_reason=gate_reason,
            attempts=attempts,
            cited_doc_ids=[],
            answer=(
                "I could not find verified information in the internal knowledge base to answer this question. "
                "Please check with the relevant team or provide additional documentation."
            ),
            abstained=True,
            mode=mode,
        )

    answer = backend.synthesize(query, verified_chunks)

    # Citation verification guard: only allow doc IDs that passed relevance grading
    allowed_ids = {g.retrieved.chunk.doc_id for g in verified_chunks}
    found_ids = _CITATION_RE.findall(answer)
    valid_cited = sorted(set(found_ids) & allowed_ids)
    hallucinated_ids = sorted(set(found_ids) - allowed_ids)
    if hallucinated_ids:
        answer += f"\n[Warning: removed unverified citation(s): {', '.join(hallucinated_ids)}]"

    return AgenticRAGResult(
        original_query=query,
        used_retrieval=True,
        gate_reason=gate_reason,
        attempts=attempts,
        cited_doc_ids=valid_cited,
        answer=answer,
        abstained=False,
        mode=mode,
    )


def select_backend(force_demo: bool) -> tuple[OpenAIRAGBackend | DemoRAGBackend, str]:
    env_demo = os.getenv("AGENT_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()

    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    if force_demo or env_demo or (not api_key and not is_local):
        return DemoRAGBackend(), "demo"
    return OpenAIRAGBackend(api_key=api_key or "ollama", base_url=base_url, model=model), f"live ({model})"


def print_result(result: AgenticRAGResult) -> None:
    print("=" * 72)
    print(f"QUERY : {result.original_query}")
    print(f"ROUTER: {'RETRIEVE' if result.used_retrieval else 'SKIP RETRIEVAL'} — {result.gate_reason}")

    for att in result.attempts:
        print(f"\n  [Retrieval Attempt {att.attempt_number}] search_query={att.search_query!r}")
        if not att.graded:
            print("    -> 0 candidate chunks matched.")
            continue
        for g in att.graded:
            chunk = g.retrieved.chunk
            tag = "RELEVANT" if g.relevant else "REJECTED"
            print(
                f"    -> [{tag}] {chunk.doc_id} ({chunk.title}) | "
                f"bm25={g.retrieved.bm25_score:.2f}, overlap={g.retrieved.overlap_score:.2f} | {g.reason}"
            )

    print("\nANSWER:")
    print(result.answer)
    if result.cited_doc_ids:
        print(f"VERIFIED CITATIONS: {', '.join(result.cited_doc_ids)}")
    elif result.abstained:
        print("STATUS: Abstained cleanly (no grounded evidence found).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 14: Agentic RAG Pipeline")
    parser.add_argument("--query", help="Single question to run. Omit to run all 3 demonstration scenarios.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per attempt.")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum query-rewrite retries.")
    parser.add_argument("--demo", action="store_true", help="Run with deterministic offline router/grader/rewriter.")
    args = parser.parse_args()

    backend, mode = select_backend(force_demo=args.demo)
    retriever = HybridLexicalRetriever(KNOWLEDGE_BASE)
    print(f"Mode: {mode} | Knowledge Base: {len(KNOWLEDGE_BASE)} chunks")

    queries = (
        [args.query]
        if args.query
        else [
            # Scenario 1: Vague wording -> weak initial match -> query rewrite -> retry -> cited answer
            "What happens if our server is down when you ping us with events?",
            # Scenario 2: Non-retrieval query -> skips RAG cleanly
            "What is 15% of 240?",
            # Scenario 3: Out-of-scope query -> retries, finds no supporting docs -> clean abstention
            "What is our company's parental leave policy in Germany?",
        ]
    )

    for q in queries:
        res = run_agentic_rag(
            query=q,
            retriever=retriever,
            backend=backend,
            top_k=args.top_k,
            max_retries=args.max_retries,
            mode=mode,
        )
        print_result(res)


if __name__ == "__main__":
    main()
