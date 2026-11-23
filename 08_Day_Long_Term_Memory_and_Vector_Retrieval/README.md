[<< Day 7](../07_Day_Short_Term_Memory_and_Conversation_State/README.md) | [Home](../README.md) | [Day 9 >>](../09_Day_Routing_and_Classification_Workflows/README.md)

# Day 8: Long-Term Memory and Vector Retrieval

Short-term conversation state disappears when a session ends. When an agent needs to remember a user's preferred stack, coding conventions, or active projects across days and separate sessions, you need durable long-term memory backed by embeddings and vector search. Today we build a persistent SQLite vector memory store with cosine similarity, metadata isolation, near-duplicate merging, and exponential time decay.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `vector_memory.py`](#walkthrough-vector_memorypy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 8](#-exercises---day-8)

---

## Core Concept & Why It Matters

On Day 7 we kept a single conversation within a token budget. Long-term memory solves a different problem: recalling a handful of relevant durable facts out of hundreds or thousands of past interactions without stuffing every historical note into every prompt.

Three design choices separate a useful memory system from one that injects noisy or wrong context:

1. **Embeddings + cosine similarity for fuzzy recall.** Exact SQL `LIKE` queries miss paraphrases ("what DB do I use?" vs. "migrated analytics store to PostgreSQL"). An embedding model maps text into a dense numeric vector where semantically similar statements point in similar directions, measured by cosine similarity:

   $$\text{cosine\_similarity}(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|}$$

2. **Metadata pre-filtering before vector ranking.** Vector search alone does not know about permissions or tenants. Always filter by `user_id` (and optionally `category` or `project`) *before* scoring vectors so User A's private facts can never bleed into User B's prompt.
3. **Deduplication and time decay.** Users repeat preferences across sessions and change their minds over months. Without near-duplicate detection (`similarity >= 0.88`), your top-3 retrieval slots fill up with three copies of the same sentence. Without recency decay, a 9-month-old preference ("we use MySQL") ties with or outranks yesterday's update ("we migrated to PostgreSQL").

### Do You Need a Dedicated Vector Database on Day 1?

Often, no. For a personal or single-tenant agent storing 1,000–50,000 memory rows, SQLite with a `user_id` index narrows the candidate set to a few hundred rows per user. Scanning a few hundred float vectors in Python takes a few milliseconds and requires zero external infrastructure. Reach for `pgvector`, Qdrant, or Pinecone when you need multi-node scale, approximate nearest neighbor (ANN) indexes over millions of vectors, or hybrid full-text + vector search across large corpora.

---

## How It Works Under the Hood

### Write Path (Ingest + Deduplication)

```text
  New fact: (user_id, category, content, importance, metadata)
                          │
                          ▼
               Compute L2-normalized embedding
                          │
                          ▼
       Scan existing rows for same (user_id, category)
                          │
            ┌─────────────┴─────────────┐
            │ max cosine >= 0.88?       │
            ▼ Yes                       ▼ No
   UPDATE existing row:          INSERT new row into
   - refresh content & vector    SQLite `memories` table
   - merge metadata JSON
   - max(importance)
   - access_count += 1
```

### Read Path (Pre-Filter + Multiplicative Recency Scoring)

```text
  Query: (user_id, query_text, category?, metadata_filter?)
                          │
                          ▼
     1. SQL WHERE user_id = ? [AND category = ?]   <-- strict tenant isolation
                          │
                          ▼
     2. Filter exact metadata key/value constraints
                          │
                          ▼
     3. Compute cosine_similarity(query_vec, row_vec)
        Drop rows below `min_similarity`
                          │
                          ▼
     4. Compute recency decay:  recency = 0.5 ^ (age_days / half_life_days)
        Combined score = sim * (0.65 + 0.25 * recency + 0.10 * importance)
                          │
                          ▼
     5. Sort descending, return top_k, update last_accessed_at & access_count
```

Notice why the combined score uses **multiplicative gating** (`sim * (0.65 + 0.25 * recency + 0.10 * importance)`) instead of a plain additive sum (`0.65 * sim + 0.25 * recency + 0.10 * importance`). With an additive sum, a brand-new completely unrelated memory (`sim = 0.17, recency = 1.0`) can outscore a genuinely relevant older memory (`sim = 0.39, recency = 0.18`). Multiplying by `sim` keeps relevance in charge while letting recency and importance break ties between related facts.

---

## Walkthrough: `vector_memory.py`

Open [`vector_memory.py`](./vector_memory.py). It uses only Python's standard library (`sqlite3`, `math`, `hashlib`, `urllib.request`) and supports two pluggable embedding backends through the `EmbeddingProvider` protocol:

- **`OpenAIEmbeddingProvider`**: Calls any OpenAI-compatible `/v1/embeddings` endpoint (configured via `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `EMBEDDING_MODEL`, defaulting to `text-embedding-3-small`).
- **`LocalHashEmbeddingProvider`**: Offline deterministic 256-dimensional embedding using signed feature hashing over word tokens, word bigrams, and character 3/4-grams, followed by L2 normalization. It lets you inspect cosine math, deduplication, persistence, and decay with zero network calls or package installs.
- **`SQLiteVectorMemory`**: Persists memories in a `memories` table indexed by `(user_id, category)`, implements `add_memory(...)` with duplicate merging, `search(...)` with tenant isolation and decay scoring, and `prune_stale(...)` to remove old low-importance clutter.

### Run the Offline Demo

From the repository root:

```bash
python 08_Day_Long_Term_Memory_and_Vector_Retrieval/vector_memory.py --demo
```

Sample output:

```text
=== Day 8: Long-Term Memory & Vector Retrieval (local-hash-256d) ===
SQLite store: /var/folders/.../agent_memory.db

1) Inserting initial memories:
   [inserted         ] mem_ffcb364ff3 user=alice cat=preference age= 10d
   [inserted         ] mem_36c6bf4526 user=alice cat=preference age= 75d
   [inserted         ] mem_6f61413ee8 user=alice cat=preference age=  1d
   [inserted         ] mem_5f5796d279 user=alice cat=project    age=  3d
   [inserted         ] mem_4e66339efd user=alice cat=note       age=120d
   [inserted         ] mem_a8a8d94fa5 user=bob   cat=preference age=  1d

2) Testing deduplication with near-identical preference phrasing:
   Result: updated_duplicate (mem_ffcb364ff3) | total alice memories: 5 (unchanged!)

3) Recalling Alice's database preference (recency decay ranks PostgreSQL above old MySQL):
   #1 score=0.355 (sim=0.364, recency=0.977, age=1d) -> Migrated analytics database preference to PostgreSQL with read replicas.
   #2 score=0.290 (sim=0.387, recency=0.177, age=75d) -> Preferred analytics database is MySQL for staging reports.
   #3 score=0.169 (sim=0.171, recency=1.000, age=0d) -> User prefers concise bullet points and Python type hints in code answers!

4) Reopened SQLite store from disk -> querying Alice's project memories:
   [mem_5f5796d279] score=0.240 -> Working on checkout-api migration in eu-west-1 targeting Python 3.12.

5) Tenant isolation check -> Bob query returned 1 hit(s) belonging solely to Bob:
   [bob] Bob prefers Rust examples and stores analytics database tables in ClickHouse.

6) Pruned stale low-importance memories: removed=1 (alice count: 5 -> 4)
```

### Persist to a Specific SQLite File or Use Live Embeddings

Keep the SQLite file on disk to inspect it with `sqlite3`:

```bash
python 08_Day_Long_Term_Memory_and_Vector_Retrieval/vector_memory.py --demo --db /tmp/my_agent_memory.db
```

Or run with live OpenAI-compatible embeddings:

```bash
export OPENAI_API_KEY="your-key"
export EMBEDDING_MODEL="text-embedding-3-small"
python 08_Day_Long_Term_Memory_and_Vector_Retrieval/vector_memory.py
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Mixing vectors from different embedding models in one table.** If you embed Monday's memories with `text-embedding-3-small` (1536 dims) and Tuesday's with a local Ollama model (768 dims), cosine similarity silently breaks or returns garbage. Always store `embedding_model` and `embedding_dim` alongside each row (or in a store metadata table) and refuse to compare mismatched models.
2. **Storing every raw chat turn as a "memory".** Dumping full transcripts into long-term memory pollutes retrieval with greetings, half-finished guesses, and superseded errors. Run a selective write step (only save durable facts, explicit user preferences, or finalized decisions).
3. **Post-filtering instead of pre-filtering by `user_id`.** If you run `top_k=5` globally across all users and then filter out other users afterward, a user with fewer memories can get 0 results—and a bug in the post-filter leaks private data across tenants. Always enforce `WHERE user_id = ?` before similarity scoring.
4. **Never handling contradictions.** Decay helps a newer "uses PostgreSQL" preference outrank an older "uses MySQL" preference, but both remain in the top-2 results and can confuse the model unless you explicitly mark older conflicting entries as superseded (or include timestamps when injecting memories into the prompt).
5. **Forgetting to set a minimum similarity floor.** Without `min_similarity`, asking "What is the weather?" still returns `top_k=5` completely irrelevant memories about databases and projects, wasting tokens and nudging the model off-topic.

---

## 💻 Exercises - Day 8

### Exercise: Level 1

1. **Inspect the SQLite database directly.** Run `vector_memory.py --demo --db /tmp/day08_test.db`, then run `sqlite3 /tmp/day08_test.db "SELECT id, user_id, category, importance, access_count, content FROM memories;"` to examine the stored rows.
2. **Experiment with `half_life_days`.** Change `half_life_days` in step 3 from `30.0` to `365.0` and then to `5.0`. Observe how the scores and ranking between the 1-day-old PostgreSQL memory and the 75-day-old MySQL memory shift.
3. **Test `min_similarity` filtering.** Run a query for `"favorite pizza topping"` against Alice's store and tune `min_similarity` so irrelevant database and project memories are filtered out cleanly.

### Exercise: Level 2

1. **Store `embedding_model` per row and validate mismatches.** Add an `embedding_model TEXT NOT NULL` column to the `memories` table. During `search(...)`, raise a clear `ValueError` (or skip with a warning) if a stored row was embedded with a different model than `self.embedder.name`.
2. **Add explicit memory superseding.** Add a `superseded_by TEXT` column and a `supersede_memory(old_id, new_content, ...)` method that inserts the new fact, links `old_id -> new_id`, and excludes superseded rows from default `search(...)` results while preserving history for auditability.
3. **Store vectors compactly as binary `BLOB`s.** Replace `embedding_json` with Python's `struct.pack(f"{len(vec)}f", *vec)` stored in a SQLite `BLOB` column, and measure the disk-size and read-speed improvement over JSON arrays on 2,000 synthetic memories.

### Exercise: Level 3

1. **Build hybrid BM25/keyword + vector search.** Add a SQLite `FTS5` virtual table synced with `memories`, retrieve both lexical matches and vector cosine matches, and combine them with Reciprocal Rank Fusion (RRF) so exact error codes like `ERR_POOL_4182` recall reliably even with semantic embeddings.
2. **Add a memory extraction & injection bridge.** Connect `SQLiteVectorMemory` to a simple chat loop: before each LLM call, retrieve up to 3 relevant memories for `user_id` formatted with timestamps (`[2026-11-22] ...`) and inject them into the system prompt; after each turn, extract durable user preferences into `add_memory(...)`.
3. **Benchmark `LocalHashEmbeddingProvider` vs. a neural embedding model.** Write an evaluation script with 15 memory-and-query pairs (including true synonyms like `"DB engine"` vs. `"PostgreSQL"` and `"shipping address"` vs. `"delivery location"`) and measure Recall@1 and Recall@3 for both providers so you can quantify exactly where lexical hashing falls short.

---

[<< Day 7](../07_Day_Short_Term_Memory_and_Conversation_State/README.md) | [Home](../README.md) | [Day 9 >>](../09_Day_Routing_and_Classification_Workflows/README.md)
