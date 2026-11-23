[<< Day 8](../08_Day_Long_Term_Memory_and_Vector_Retrieval/README.md) | [Home](../README.md) | [Day 10 >>](../10_Day_Parallel_Execution_and_Map_Reduce/README.md)

# Day 9: Routing and Classification Workflows

Sending every user request to one giant system prompt loaded with 20 tools and your most expensive model is slow, costly, and hard to maintain. Today we build a two-stage intent router that handles obvious lookups in plain Python at zero LLM cost, classifies ambiguous queries into strict JSON with a confidence threshold, and dispatches each request to the cheapest specialized handler—or a safe clarification fallback—that can solve it reliably.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Walkthrough: `intent_router.py`](#walkthrough-intent_routerpy)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 9](#-exercises---day-9)

---

## Core Concept & Why It Matters

When teams ship their first agent, they usually put every instruction and tool into one monolithic prompt. As soon as traffic grows, three problems show up:

1. **You pay frontier-model prices for trivial lookups.** Asking "Where is order ORD-2088?" or "What is (125 * 12) + 450 / 3?" does not need a reasoning model—a 5-line regex and a database query (or AST evaluator) answers in under a millisecond for $0.00.
2. **Prompt interference lowers accuracy.** Stuffing billing policy, refund rules, webhook debugging guides, and order-tracking formats into one system prompt causes rules to compete for attention. Splitting into small, single-purpose prompts makes each route easier to test and tune independently.
3. **Silent wrong guesses hurt user trust.** When a user types `"Hey, something seems off since Tuesday, can you help?"`, forcing an agent to pick a tool produces wrong answers. A router with an explicit confidence threshold catches low-confidence inputs and asks one targeted clarifying question instead.

### Comparing Monolithic Agents vs. Routed Workflows

| Dimension | Single Monolithic Agent | Two-Stage Intent Router |
| :--- | :--- | :--- |
| Cost on routine lookups | Full prompt + model every time | $0 via deterministic rules |
| Prompt size per call | Large (all domains + tools) | Small (only matched domain) |
| Model selection | One model for everything | Cheap classifier + fast/specialist tiers |
| Handling vague queries | Often guesses a tool blindly | Explicit confidence gate -> clarification |
| Added failure mode | None (no router step) | Misclassification can send query to wrong handler |

Routing adds one decision point that can fail if your taxonomy overlaps or your classifier is poorly calibrated. Always measure **routing accuracy** on real user queries—a cheap router that misroutes 20% of traffic costs more in user frustration than it saves on tokens.

---

## How It Works Under the Hood

Requests pass through three stages: deterministic fast-path rules, a JSON intent classifier, and a confidence-gated dispatcher.

```text
                        Incoming User Query
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │ Stage 1: Deterministic Pre-Router   │  (<1 ms, $0)
              │  - Explicit ORD-XXXX + status word? │──Match──► order_status (Python DB)
              │  - Pure arithmetic expression?      │──Match──► calculator (safe AST)
              └──────────────────┬──────────────────┘
                                 │ No match
                                 ▼
              ┌─────────────────────────────────────┐
              │ Stage 2: Structured LLM Classifier  │  (small model, JSON mode)
              │  -> {intent, confidence, entities}  │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │ Stage 3: Confidence Gate (>= 0.72?) │
              └──────────┬────────────────┬─────────┘
                         │ No / unknown   │ Yes
                         ▼                ▼
                fallback_clarify    Dispatch by intent:
                (ask focused        - order_status      -> deterministic Python
                 clarification)     - refund_escalation -> ticket creator (Python)
                                    - billing_faq       -> fast/cheap LLM
                                    - tech_support      -> specialist LLM
```

### Why Safe AST Evaluation Instead of `eval()`?

When routing arithmetic to Python, never call `eval(expression)`. User input can easily contain `__import__('os').system(...)`. `safe_eval_math` parses the expression with `ast.parse(..., mode="eval")` and walks only numeric literals and whitelisted arithmetic operators (`+`, `-`, `*`, `/`, `**` with bounded exponent, `%`). Anything else raises `ValueError`.

---

## Walkthrough: `intent_router.py`

Open [`intent_router.py`](./intent_router.py). It runs standalone with zero third-party packages, supporting both live OpenAI-compatible models (`MODEL_NAME` for classifier/fast routes and optional `SPECIALIST_MODEL_NAME` for specialist routes) and an offline deterministic `--demo` mode.

Key components in the file:

- **`try_deterministic_route(query)`**: Runs before any LLM call. Catches explicit `ORD-XXXX` tracking requests (while explicitly excluding `"refund"` so `"Please refund order ORD-1042"` does not get misrouted to tracking) and pure arithmetic expressions.
- **`IntentRouter.classify(query)`**: Calls the classifier in JSON mode (with automatic retry without `response_format` if a local OpenAI-compatible server rejects `json_object`), validates `intent` against `VALID_INTENTS`, clamps `confidence` to `[0.0, 1.0]`, and degrades malformed JSON cleanly to `intent="unknown", confidence=0.0`.
- **`IntentRouter.route(query)`**: Enforces `confidence >= confidence_threshold` (default `0.72`), dispatches to deterministic Python, fast LLM, specialist LLM, or `fallback_clarify`, and records `latency_ms` and `estimated_cost_usd`.

### Run the Offline Demo

From the repository root:

```bash
python 09_Day_Routing_and_Classification_Workflows/intent_router.py --demo
```

Sample output:

```text
=== Day 9: Routing & Classification Workflow (offline deterministic demo) ===

[1] Query: Where is my package for order ORD-2088?
    Route: order_status (intent=order_status, conf=1.00, via=deterministic_rule, handler=deterministic_python)
    Cost: $0.00000 | Latency: 0.01ms
    Response: Order ORD-2088: status=in_transit, carrier=FedEx, ETA=Nov 25 by 18:00.

[2] Query: What is (125 * 12) + 450 / 3?
    Route: calculator (intent=calculator, conf=1.00, via=deterministic_rule, handler=deterministic_python)
    Cost: $0.00000 | Latency: 0.05ms
    Response: (125 * 12) + 450 / 3 = 1650

[3] Query: How do I download last month's invoice PDF with our VAT ID?
    Route: billing_faq (intent=billing_faq, conf=0.91, via=llm_classifier, handler=fast_llm)
    Cost: $0.00045 | Latency: 0.04ms
    Response: Invoices are generated on the 1st of each month and can be downloaded as PDFs under Settings -> Billing -> Invoice History. You can also add a VAT/Tax ID there.

[4] Query: Our webhook consumer keeps hitting HTTP 429 rate limits in production.
    Route: tech_support (intent=tech_support, conf=0.89, via=llm_classifier, handler=specialist_llm)
    Cost: $0.00265 | Latency: 0.02ms
    Response: HTTP 429 means your webhook client exceeded the per-minute rate limit (120 req/min). Read the `Retry-After` response header, apply exponential backoff with jitter, and batch non-urgent webhook events.

[5] Query: Please refund order ORD-1042, the package arrived damaged.
    Route: refund_escalation (intent=refund_escalation, conf=0.94, via=llm_classifier, handler=deterministic_python)
    Cost: $0.00015 | Latency: 0.05ms
    Response: Created priority refund triage ticket for order=ORD-1042. A billing specialist will review eligibility and follow up within 4 business hours.

[6] Query: Can you check my shipment status?
    Route: order_status (intent=order_status, conf=0.82, via=llm_classifier, handler=deterministic_python)
    Cost: $0.00015 | Latency: 0.02ms
    Response: Please share your order ID (format ORD-1234) so I can check live shipping status.

[7] Query: Hey, something seems off since Tuesday, can you help?
    Route: fallback_clarify (intent=unknown, conf=0.42, via=llm_classifier, handler=fallback)
    Cost: $0.00015 | Latency: 0.01ms
    Response: I want to make sure I route this correctly (classifier confidence was 0.42). Are you asking about (1) order tracking, (2) invoices/billing, (3) a refund request, or (4) API technical support?

--- Batch Summary ---
Queries routed:        7
Estimated routed cost: $0.00355
Monolithic baseline:   $0.03150 (always calling a large prompt/model)
Estimated cost saved:  88.7%
```

### Run Against Live Models

Route a single query with live models (optionally using a stronger specialist model for `tech_support`):

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
export SPECIALIST_MODEL_NAME="gpt-4o"
python 09_Day_Routing_and_Classification_Workflows/intent_router.py \
  --query "Why did our webhook consumer start returning HTTP 429?"
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Over-eager regex rules stealing nuanced queries.** Look at query #5 (`"Please refund order ORD-1042..."`). If the deterministic `order_status` rule checked only for `ORD-\d{4}`, it would swallow refund requests and reply with shipping status instead of escalating a refund. Keep Stage 1 deterministic rules narrow and high-precision; let Stage 2 handle nuance.
2. **Trusting raw LLM confidence numbers blindly.** LLMs are notoriously overconfident—a model often outputs `0.95` even on borderline queries. Calibrate your `confidence_threshold` against a labeled validation set (for example, 50 real queries including tricky edge cases) rather than picking `0.50` by guess.
3. **No fallback for multi-intent queries.** Users regularly send compound messages like `"Where is ORD-2088 and also how do I add my VAT ID to last month's invoice?"`. A single-label router picks only one and drops the second request unless you either support multi-intent classification (`intents: list[str]`) or detect multi-part prompts and route them to a general coordinator.
4. **Overlapping intent definitions.** Defining both `"billing_support"` and `"payment_issues"` without sharp boundaries guarantees unstable routing. Give every route a mutually exclusive description and 2–3 concrete boundary examples in the classifier prompt.
5. **Forgetting to log misroutes.** Without logging `(query, decision, executed_route, user_followup)`, you cannot tell which routes are misclassifying traffic. Log every routing decision so you can turn real production mistakes into regression tests.

---

## 💻 Exercises - Day 9

### Exercise: Level 1

1. **Experiment with the confidence threshold.** Run `intent_router.py --demo --threshold 0.90` and `--threshold 0.30`. See which queries get redirected to `fallback_clarify` at `0.90` and why `0.30` lets an ambiguous query slip through.
2. **Test safety of `safe_eval_math`.** Pass `--query "__import__('os').system('echo hi')"` and `--query "2 ** 10000"` in `--demo` mode and verify that neither executes arbitrary code or hangs the process.
3. **Add a deterministic health-status rule.** Add a Stage 1 rule matching queries like `"is the api up?"` or `"system status"` that returns a deterministic status response in Python with `$0.00` LLM cost.

### Exercise: Level 2

1. **Support multi-intent queries.** Upgrade `RouteDecision` to return `intents: list[str]` (up to 2 intents). When a user asks `"Where is ORD-2088 and how do I download my VAT invoice?"`, execute both matched handlers and combine their responses cleanly.
2. **Build a labeled routing evaluation harness.** Create a list of 15 `(query, expected_route)` pairs (including at least 3 ambiguous queries expecting `fallback_clarify`), run `IntentRouter.route` over all 15, and print overall accuracy plus a confusion matrix showing any misrouted examples.
3. **Add semantic caching for repeated FAQ routes.** Cache `(normalized_query -> response)` for `billing_faq` so identical repeat questions return in `<1ms` at `$0.00` without calling `fast_llm` again.

### Exercise: Level 3

1. **Add second-chance escalation on low-margin classification.** Have the classifier return the top-2 candidate intents (`primary_intent`, `primary_confidence`, `secondary_intent`, `secondary_confidence`). If `primary_confidence - secondary_confidence < 0.15`, automatically ask a targeted disambiguation question naming only those two candidates instead of listing all four queues.
2. **Build a fallback-to-human handoff payload.** When `refund_escalation` fires or when a user hits `fallback_clarify` twice in a row, emit a structured JSON handoff ticket (`ticket_id`, `priority`, `summary`, `extracted_entities`, `classifier_reasoning`) ready to post to a ticketing webhook.
3. **Measure live latency and token usage across tiers.** Extend `OpenAICompatibleLLM.complete` to parse `usage.prompt_tokens` and `usage.completion_tokens` from the API response, calculate exact dollar cost per model tier instead of fixed estimates, and compare p50/p95 latency and total spend against a single-model baseline on 20 test queries.

---

[<< Day 8](../08_Day_Long_Term_Memory_and_Vector_Retrieval/README.md) | [Home](../README.md) | [Day 10 >>](../10_Day_Parallel_Execution_and_Map_Reduce/README.md)
