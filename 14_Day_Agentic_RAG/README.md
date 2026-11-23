[<< Day 13](../13_Day_Reflection_and_Self_Critique_Loops/README.md) | [Home](../README.md) | [Day 15 >>](../15_Day_Model_Context_Protocol_Basics/README.md)

# Day 14: Agentic RAG

Today we upgrade plain static Retrieval-Augmented Generation (RAG) into **Agentic RAG**: a pipeline that decides whether retrieval is needed, evaluates whether retrieved chunks are actually relevant, rewrites vague queries when search comes up weak, and cites verified sources (or abstains cleanly instead of hallucinating).

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `agentic_rag.py`](#walkthrough-agentic_ragpy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 14](#-exercises---day-14)

---

## Core Concept & Why It Matters

Standard (static) RAG is a fixed 3-step pipeline:

```text
User Query ──► Embed / Search Top-K ──► Stuff Chunks into Prompt ──► Generate Answer
```

In production, static RAG breaks in four familiar ways:

1. **Unnecessary retrieval:** A user asks *"What is 15% of 240?"* or *"Hi"* and the pipeline still queries the vector/keyword index, retrieves three semi-random chunks, and drags them into the prompt.
2. **Vocabulary mismatch on vague queries:** A user writes *"What happens if our server is down when you ping us with events?"* They never say `"webhook"`, `"exponential backoff"`, or `"DLQ"`. Static search either misses the right document or pulls weak partial matches.
3. **Context pollution:** Top-3 search always returns 3 chunks—even if 2 of them are irrelevant noise. Irrelevant chunks distract the generator and increase hallucination rates.
4. **Silent hallucination on missing docs:** When the user asks about a topic not in the knowledge base at all (like German parental leave policy in an engineering runbook), a static generator still tries to be "helpful" and fabricates an answer.

**Agentic RAG** puts decision-making and feedback loops around retrieval:

- **Retrieval Gate:** Skip retrieval completely when the query does not need internal documents.
- **Relevance Grader:** Inspect every candidate chunk before generation and drop noise.
- **Query Rewriter + Retry Loop:** If no retrieved chunk passes grading, rewrite the query with sharper domain terms and search again (up to `max_retries`).
- **Grounded Synthesis + Abstention:** Synthesize answers strictly from graded-relevant chunks with inline `[KB-XX]` citations, verify every cited ID against the allowed set, and abstain honestly when no supporting evidence exists.

### Cost & Latency Trade-Offs

| Dimension | Static RAG | Agentic RAG |
| :--- | :--- | :--- |
| **LLM calls per query** | 1 | 1 (if skipped) to 4+ (route + grade + rewrite + synthesize) |
| **Median latency** | Lower | Higher unless routing/grading use fast small models or batched calls |
| **Accuracy on vague queries** | Poor when phrasing differs from docs | Much higher due to query rewriting + retries |
| **Hallucination on out-of-scope queries** | High | Low when paired with strict grading and abstention |

You do not need a frontier model for every stage. Routing and batch chunk grading work well on fast, cheap models (or hybrid heuristics + small LLMs), reserving the stronger model for final synthesis.

---

## How It Works Under the Hood

Here is the control flow implemented in [`agentic_rag.py`](./agentic_rag.py):

```text
  User Query
      │
      ▼
┌──────────────────────────────┐   NO    ┌────────────────────────┐
│ 1. Retrieval Gate            ├────────►│ Direct Answer (0 index │
│    Needs internal KB?        │         │ calls, low latency)    │
└──────────────┬───────────────┘         └────────────────────────┘
               │ YES
               ▼
┌──────────────────────────────┐◄──────────────────────────────┐
│ 2. Hybrid Lexical Retriever  │                               │
│    Okapi BM25 + token overlap│                               │
│    (top_k = 3)               │                               │
└──────────────┬───────────────┘                               │
               │ candidates                                    │
               ▼                                               │
┌──────────────────────────────┐                               │
│ 3. Relevance Grader          │                               │
│    Keep only chunks that     │                               │
│    directly answer the query │                               │
└───────┬──────────────┬───────┘                               │
        │              │ 0 relevant & retries left             │
   >=1 relevant        ▼                                       │
        │      ┌──────────────────────────────┐                │
        │      │ 4. Query Rewriter            ├────────────────┘
        │      │    Expand vague phrasing into│
        │      │    technical search terms    │
        │      └──────────────────────────────┘
        │              │ 0 relevant & retries exhausted
        │              ▼
        │      Clean Abstention ("No verified docs found")
        ▼
┌──────────────────────────────┐
│ 5. Grounded Synthesizer      │
│    Cite inline [KB-XX] +     │
│    verify every citation ID  │
└──────────────────────────────┘
```

### How the Built-In Retriever Scores Chunks

[`agentic_rag.py`](./agentic_rag.py) implements a zero-dependency **Okapi BM25 + token-overlap** retriever (`HybridLexicalRetriever`) over an 8-chunk internal engineering knowledge base (`KB-01` through `KB-08`):

- **BM25 (`k1=1.5`, `b=0.75`)** rewards rare domain terms (high inverse document frequency like `idempotency` or `pgbouncer`) while saturating repeated terms and normalizing for chunk length.
- **Title and tag boosting** repeats `chunk.title` and includes structured `tags` so matches in headings carry more weight than incidental mentions in prose.
- **Token overlap ratio** (`|query_terms ∩ doc_terms| / |query_terms|`) rewards chunks that cover a higher share of the user's terms.

Even when you later add dense vector embeddings (for example, `text-embedding-3-small` + Qdrant/pgvector), keeping BM25 in a hybrid retrieval setup dramatically improves exact matching on error codes (`429`, `410`), endpoint paths (`/v2/keys/rotate`), and header names.

---

## Walkthrough: `agentic_rag.py`

When run with `--demo` and no `--query`, [`agentic_rag.py`](./agentic_rag.py) executes three contrasting scenarios so you can inspect every branch of the pipeline:

1. **Vague query requiring rewrite:** *"What happens if our server is down when you ping us with events?"*
   - Attempt 1 weakly matches only `"events"` (`bm25=1.59`) and gets rejected by the grader.
   - The rewriter translates it to `'webhook delivery retries exponential backoff dead-letter queue'`.
   - Attempt 2 hits `KB-02` (`bm25=18.59, overlap=1.00`), rejects noisy `KB-01`, and synthesizes a grounded `[KB-02]` answer.
2. **Non-retrieval query:** *"What is 15% of 240?"*
   - The retrieval gate skips the index entirely and computes `36` directly.
3. **Out-of-knowledge-base query:** *"What is our company's parental leave policy in Germany?"*
   - Runs initial search plus two rewrites, gets zero verified chunks, and abstains cleanly instead of inventing an HR policy.

### Running the Script

Run all three built-in scenarios in offline demo mode:

```bash
python3 14_Day_Agentic_RAG/agentic_rag.py --demo
```

Sample output:

```text
Mode: demo | Knowledge Base: 8 chunks
========================================================================
QUERY : What happens if our server is down when you ping us with events?
ROUTER: RETRIEVE — Query asks about operational policies or platform behavior covered by internal docs.

  [Retrieval Attempt 1] search_query='What happens if our server is down when you ping us with events?'
    -> [REJECTED] KB-02 (Webhook Delivery Retries and Dead-Letter Queue) | bm25=1.59, overlap=0.25 | Weak match (bm25=1.59, overlap=0.25, matched=['events']); insufficient to answer safely.

  [Retrieval Attempt 2] search_query='webhook delivery retries exponential backoff dead-letter queue'
    -> [RELEVANT] KB-02 (Webhook Delivery Retries and Dead-Letter Queue) | bm25=18.59, overlap=1.00 | Matches key domain terms (backoff, dead-letter, delivery, exponential); directly answers query.
    -> [REJECTED] KB-01 (API Rate Limits and Burst Quotas) | bm25=1.25, overlap=0.14 | Weak match (bm25=1.25, overlap=0.14, matched=['backoff']); insufficient to answer safely.

ANSWER:
When a customer webhook endpoint times out (>10s) or returns a non-2xx HTTP status, the platform retries delivery up to 5 times with exponential backoff: 1 minute, 5 minutes, 15 minutes, 1 hour, and 6 hours. [KB-02] Events that still fail are placed in the Dead-Letter Queue (DLQ) for 24 hours where operators can manually replay them. [KB-02] Every delivery includes an `Idempotency-Key` header. [KB-02]
VERIFIED CITATIONS: KB-02
========================================================================
QUERY : What is 15% of 240?
ROUTER: SKIP RETRIEVAL — Pure arithmetic calculation; internal knowledge base lookup is unnecessary.

ANSWER:
15% of 240 is 36.
========================================================================
QUERY : What is our company's parental leave policy in Germany?
ROUTER: RETRIEVE — Query asks about operational policies or platform behavior covered by internal docs.
...
ANSWER:
I could not find verified information in the internal knowledge base to answer this question. Please check with the relevant team or provide additional documentation.
STATUS: Abstained cleanly (no grounded evidence found).
```

Run a custom query against a live OpenAI-compatible model:

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
python3 14_Day_Agentic_RAG/agentic_rag.py --query "Can we ship a hotfix on Friday evening during a major outage?"
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Grading chunks one-by-one in serial LLM calls:** If `top_k=8` and you issue 8 sequential grading requests, latency explodes. Batch all candidate chunks into a single JSON grading call (as `OpenAIRAGBackend.grade_chunks()` does) or grade concurrently.
2. **Over-rewriting away from user intent:** An aggressive rewriter can turn *"Can I downgrade my plan mid-month and get a refund?"* into *"billing upgrade proration"* and retrieve the wrong policy. Always pass the `original_query` into the final synthesizer so the answer addresses what the user actually asked—not just the rewritten keywords.
3. **Fake or unverified citations:** Models frequently attach `[KB-01]` to a claim that is not actually in `KB-01`, or cite a chunk ID that was rejected during grading. Check every cited ID against `allowed_ids` in code, and for high-stakes apps run a post-generation entailment check.
4. **Infinite query-rewrite loops:** Always cap `max_retries` (usually 1–2) and stop early if the rewriter outputs the same query twice.
5. **Passing bad chunks straight to synthesis on retry exhaustion:** If all retries fail to find relevant context, **abstain**. Falling back to "just pass the top-3 rejected chunks anyway" defeats the entire point of grading.

---

## 💻 Exercises - Day 14

### Exercise: Level 1

1. Run `agentic_rag.py --demo --query "Can we deploy on Friday at 16:00 UTC?"` and inspect which chunks are retrieved, graded, and cited.
2. Add two new `DocumentChunk` entries (`KB-09` for OAuth token lifetimes and `KB-10` for audit log retention) to `KNOWLEDGE_BASE` and test both direct and vague queries against them.
3. Inspect the custom query `"How long do old API keys stay valid after rotation?"` in demo mode: notice `KB-01` also slips in because it shares `"api"` and `"keys"`. Tighten `DemoRAGBackend.grade_chunks()` so `KB-01` is rejected for that query while `KB-03` remains accepted.

### Exercise: Level 2

1. Implement **multi-query fan-out**: when a user asks a compound question (e.g., *"What are the sandbox rate limits and how do webhook retries work?"*), split it into two sub-queries, retrieve for each, deduplicate chunks by `doc_id` keeping the highest score, grade the combined set, and synthesize a unified cited answer.
2. Add a **citation completeness check** in `run_agentic_rag()`: verify that every sentence in a non-abstained retrieved answer ends with at least one valid `[KB-XX]` tag, and flag uncited sentences in a `warnings` list on `AgenticRAGResult`.
3. Add a lightweight `--eval` mode with 6 test cases (expected `used_retrieval`, expected `abstained`, and expected `doc_id`s) and print precision/recall and routing accuracy across the suite.

### Exercise: Level 3

1. Add **Reciprocal Rank Fusion (RRF)** combining `HybridLexicalRetriever` with a second retriever (for example, character n-gram cosine similarity or real embeddings via your OpenAI-compatible `/embeddings` endpoint), then compare top-3 recall on 10 colloquial user queries.
2. Implement **conversation-aware query contextualization**: allow passing `--history "Previous Q: ... Previous A: ..."` so follow-up questions with pronouns (*"And what if it's compromised?"* after asking about key rotation) are rewritten into standalone search queries before retrieval.
3. Build a **chunking + metadata ingestion loader** that reads markdown files from a folder, splits them by `##` headings with configurable overlap, assigns stable `doc_id`s, and indexes them into `HybridLexicalRetriever`.

---

[<< Day 13](../13_Day_Reflection_and_Self_Critique_Loops/README.md) | [Home](../README.md) | [Day 15 >>](../15_Day_Model_Context_Protocol_Basics/README.md)
