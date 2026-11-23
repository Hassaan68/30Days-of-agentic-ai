[<< Day 6](../06_Day_Error_Handling_and_Self_Correction/README.md) | [Home](../README.md) | [Day 8 >>](../08_Day_Long_Term_Memory_and_Vector_Retrieval/README.md)

# Day 7: Short-Term Memory and Conversation State

Keeping a plain Python `list` of messages works for a 3-turn script, then fails fast in real sessions: input token costs grow quadratically, old tool dumps crowd out recent instructions, and naively trimming `messages[-10:]` breaks `tool_call`/`tool` pairs with HTTP 400 errors. Today we build a typed `ConversationState` manager that enforces a strict token budget using atomic pair-safe grouping, rolling summaries, and explicit working-memory slots.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `conversation_state.py`](#walkthrough-conversation_statepy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 7](#-exercises---day-7)

---

## Core Concept & Why It Matters

LLMs are stateless over HTTP. Every time you send a request to `/v1/chat/completions`, you resend the entire conversation history. That creates three practical engineering problems as sessions grow:

1. **Quadratic token cost and rising latency.** Sending 20 turns of raw JSON tool responses on every step means turn 20 re-bills all 19 earlier turns. A single 4 KB log query left in raw history costs tokens on every subsequent model call.
2. **Broken tool-call pairs from naive slicing.** Most developers try `messages = [system_msg] + messages[-8:]` as a quick fix. The moment that slice cuts between an `assistant` message containing `tool_calls` and the `tool` messages answering those `tool_call_id`s, OpenAI-compatible APIs reject the request with `400 Bad Request: messages with role 'tool' must be a response to a preceding message with 'tool_calls'`.
3. **Loss of durable facts during compaction.** Pure sliding windows forget facts stated in turn 1 (like which customer account, region, or incident ID you are working on). Summaries help, but repeated re-summarization drifts over time—small details like `"ACC-884"` or `"eu-west-1"` eventually get paraphrased away unless you store them in structured state.

### Three Complementary Layers of Short-Term Memory

| Layer | What Lives Here | Lifetime | Cost Profile |
| :--- | :--- | :--- | :--- |
| **Pinned System Prompt** | Role, rules, output format | Entire session | Fixed |
| **Structured Working Memory** | Explicit key-value slots (`account_id`, `region`, `active_incident`) | Updated explicitly on key events | Tiny (~30–80 tokens), zero drift |
| **Rolling Summary** | Compact bullets of evicted turns | Grows up to `max_summary_tokens` | Bounded |
| **Recent Atomic Message Window** | Verbatim recent turns + tool call/result blocks | Trimmed oldest-block-first | Bounded by remaining token budget |

Separating **working memory slots** from **rolling prose summary** is one of the highest-leverage improvements you can make to an agent. Exact identifiers stay exact in JSON; narrative context stays compact in the summary; immediate conversational flow stays verbatim in the recent window.

---

## How It Works Under the Hood

Before trimming anything, group raw messages into **atomic blocks** so an assistant's tool call request and all of its tool results move or get evicted together:

```text
  Raw message list:
  [user_1] [asst(tc_1a, tc_1b)] [tool(tc_1a)] [tool(tc_1b)] [asst_1] [user_2] ...
     │      └───────────────────────┬───────────────────────┘   │       │
     ▼                              ▼                           ▼       ▼
  Block 0                        Block 1                     Block 2  Block 3
  (1 msg)                        (3 msgs - indivisible!)     (1 msg)  (1 msg)

  When estimated_prompt_tokens() > max_prompt_tokens:

     1. Pop oldest atomic block(s) from the front of `blocks`
     2. Pass evicted batch ONCE to SummaryCompactor -> update `running_summary`
     3. Re-flatten remaining blocks into `state.messages`
     4. Run `validate_tool_pairs(state.messages)` as a safety assertion
```

### Final Prompt Layout Sent to the Model

```text
  ┌───────────────────────────────────────────────────────────┐
  │ role: "system"                                            │
  │   - Base system instructions                              │
  │   - Structured session state: {"account_id": "ACC-884", …}│
  │   - Summary of earlier conversation: (bounded bullets)    │
  ├───────────────────────────────────────────────────────────┤
  │ role: "user" / "assistant" / "tool"                       │
  │   - Recent atomic blocks fitting within max_prompt_tokens │
  └───────────────────────────────────────────────────────────┘
```

Evicting blocks in a batch before calling the compactor also keeps live LLM costs down: if one large tool response pushes the window over budget and 3 old blocks need to be evicted, the summarizer runs once for that batch instead of making 3 separate LLM calls.

---

## Walkthrough: `conversation_state.py`

Open [`conversation_state.py`](./conversation_state.py). It implements the full short-term state pipeline with zero third-party dependencies, supporting both an offline deterministic compactor (`--demo`) and a live OpenAI-compatible summarizer.

Key components in the file:

- **`ToolCall` and `Message` dataclasses**: Typed structures with `to_openai_dict()`, `to_dict()`, and `from_dict()` so you never pass loosely typed dicts around your app.
- **`group_atomic_blocks(messages)`**: Bundles an `assistant` message with `tool_calls` together with every immediately following `tool` message whose `tool_call_id` matches.
- **`validate_tool_pairs(messages)`**: Verifies two invariants—no orphan `tool` message without a preceding matching `tool_call`, and no dangling `tool_call_id` left unanswered before the next `user` or `assistant` turn.
- **`DeterministicCompactor` and `OpenAISummaryCompactor`**: Implement the `SummaryCompactor` protocol. The live compactor falls back to `DeterministicCompactor` on network errors so an ongoing user session does not crash if summarization temporarily fails.
- **`ConversationState`**: Tracks `session_id`, `system_prompt`, `working_memory`, `running_summary`, `messages`, `budget`, and compaction counters, plus `to_json()` / `from_json()` for checkpointing to disk or Redis.

### Run the Offline Demo

From the repository root:

```bash
python 07_Day_Short_Term_Memory_and_Conversation_State/conversation_state.py --demo
```

Sample output:

```text
=== Day 7: Short-Term Memory & Conversation State (offline deterministic compactor) ===

1) Why naive slicing (`messages[-3:]`) fails:
   VIOLATION: Index 0: orphan tool message for tool_call_id='call_a1' with no preceding matching assistant tool_call.
   VIOLATION: Index 1: orphan tool message for tool_call_id='call_a2' with no preceding matching assistant tool_call.

2) Bounded ConversationState across a 6-turn incident investigation:
   Token budget ceiling (max_prompt_tokens): 420

   Turn 1: msgs= 5 | blocks=3 | tokens=204/420 | evicted= 0 | compactions=0 | pairs_valid=True
   Turn 2: msgs=10 | blocks=6 | tokens=369/420 | evicted= 0 | compactions=0 | pairs_valid=True
   Turn 3: msgs= 5 | blocks=4 | tokens=361/420 | evicted= 9 | compactions=1 | pairs_valid=True
   Turn 4: msgs= 5 | blocks=4 | tokens=372/420 | evicted= 4 | compactions=2 | pairs_valid=True
   Turn 5: msgs= 5 | blocks=4 | tokens=342/420 | evicted= 4 | compactions=3 | pairs_valid=True
   Turn 6: msgs= 7 | blocks=6 | tokens=413/420 | evicted= 0 | compactions=3 | pairs_valid=True

3) Preserved structured working memory:
   {"account_id": "ACC-884", "active_deploy": "DEPLOY-771", "active_incident": "INC-409", "region": "eu-west-1", "status": "mitigated", "suspect_deploy": "DEPLOY-772", "tier": "enterprise"}

4) Rolling summary retained after evicting older turns:
   - User asked: Inspect error logs for DEPLOY-772 to see what is crashing.
   - Tool query_logs: {"top_error":"ConnectionPoolExhausted on db-eu-west-1","count":4182}
   - Assistant noted: Logs show 4,182 ConnectionPoolExhausted errors on db-eu-west-1 introduced by DEPLOY-772.
   - User asked: Roll back checkout-api in eu-west-1 to the previous stable build DEPLOY-771.
   - Tool rollback_service: {"status":"rolled_back","active_deploy":"DEPLOY-771"}

5) Checkpoint round-trip OK (2607 bytes JSON, 17 total messages compacted).
```

### Run with Live LLM Rolling Summarization

With an OpenAI-compatible API key (or local Ollama) configured, omit `--demo` to run `OpenAISummaryCompactor`:

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
python 07_Day_Short_Term_Memory_and_Conversation_State/conversation_state.py
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Forgetting tool-definition tokens in your budget.** Your system prompt and messages are not the only input tokens—the JSON `tools` schema sent alongside the request can easily consume 500–2,000 tokens. Subtract tool-schema overhead when configuring `TokenBudget.max_context_tokens`.
2. **Oversized single tool responses.** What if one `query_logs` call returns 25,000 tokens of raw JSON? Evicting older blocks cannot fix a single block that exceeds the entire budget. Truncate or summarize large tool outputs *at tool-return time* (for example, cap at 4,000 characters with a `truncated: true` flag and store the full output in an artifact/file the agent can page through).
3. **Summary telephone-game drift.** Re-summarizing a summary 15 times gradually drops subtle details (like an exact timestamp or ID) and sometimes hallucinates details that were never confirmed. Keep exact identifiers in `working_memory` slots rather than relying on prose summaries alone.
4. **Assuming character count equals exact tokens.** `len(text) / 3.5` is a safe conservative heuristic for English and JSON, but non-Latin scripts, dense symbols, and minified code can tokenize at 1.5–2 chars per token. Leave a comfortable `reserve_for_response` buffer, or swap `estimate_tokens` for `tiktoken` (or the provider's returned `usage.prompt_tokens`) in production.
5. **Mutating live state without checkpointing.** If a multi-turn web server keeps `ConversationState` only in a process-local dictionary, a container restart wipes active user sessions. Persist `state.to_json()` to SQLite, Postgres, or Redis after every completed turn.

---

## 💻 Exercises - Day 7

### Exercise: Level 1

1. **Test different token budgets.** Change `TokenBudget.max_context_tokens` in `run_demo` to `400` and `900`, run `--demo`, and observe how `compactions`, `evicted`, and final `running_summary` change.
2. **Trigger every `validate_tool_pairs` check.** Write a small snippet that constructs: (a) an orphan `tool` message, (b) an assistant `tool_calls` message followed directly by a `user` message with no `tool` result, and (c) an assistant message with 2 tool calls where only 1 `tool` result is present. Confirm `validate_tool_pairs` catches all three.
3. **Inspect the OpenAI wire payload.** Call `state.build_prompt_messages()` at the end of `run_demo` and print the formatted JSON so you can verify how `working_memory` and `running_summary` are injected into the pinned `system` message.

### Exercise: Level 2

1. **Add tool-output truncation at ingestion time.** Update `add_tool_result(...)` to accept `max_chars: int = 800`. When `content` exceeds `max_chars`, truncate cleanly, append `... [truncated N chars]`, and record the truncated character count in a counter on `ConversationState`.
2. **Persist checkpoints to disk.** Add `--checkpoint-file path/to/state.json` CLI support so running the script once saves the state to JSON and running with `--resume path/to/state.json` reloads `ConversationState`, appends one more turn, and compacts again.
3. **Account for tool-schema overhead in `TokenBudget`.** Add a `tool_schema_tokens: int = 0` field to `TokenBudget` and a helper `estimate_tools_tokens(tools_schema: list[dict[str, Any]]) -> int`, deducting it from `max_prompt_tokens`.

### Exercise: Level 3

1. **Implement a real token-counter adapter.** Create a `TokenCounter` protocol with two implementations: the existing heuristic counter and an optional `TiktokenCounter` (using lazy `import tiktoken` with graceful fallback if `tiktoken` is not installed). Inject it into `ConversationState`.
2. **Add priority-aware compaction.** Tag messages with an optional `pinned: bool = False` flag (for example, a user's explicit policy constraint or approval confirmation). Modify `compact_if_needed` so pinned atomic blocks are skipped during eviction until no unpinned blocks remain.
3. **Compare rolling summary vs. pure sliding window on fact recall.** Write a short benchmark script that runs 20 turns where key facts appear in turns 1–3, then checks how many original facts are still present in `build_prompt_messages()` at turn 20 under (a) naive sliding window, (b) pair-safe sliding window without summary, and (c) `ConversationState` with `working_memory` + `running_summary`.

---

[<< Day 6](../06_Day_Error_Handling_and_Self_Correction/README.md) | [Home](../README.md) | [Day 8 >>](../08_Day_Long_Term_Memory_and_Vector_Retrieval/README.md)
