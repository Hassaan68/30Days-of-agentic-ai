[Home](../README.md) | [Day 2 >>](../02_Day_Prompts_System_Instructions_and_Context/README.md)

# Day 1: AI, ML, Deep Learning, Generative AI & Agentic AI Foundations

Before writing multi-step agents, you need clear definitions of the stack you are building on and an honest sense of when an agent is the right tool—and when plain Python code is better. Today we map out how AI, Machine Learning, Deep Learning, Generative AI, LLMs, and Agentic AI fit together, review the core vocabulary you will use for the next 29 days, verify your local environment, and run two working scripts.

## Table of Contents

- [Core Concepts: From AI to Agentic AI](#core-concepts-from-ai-to-agentic-ai)
- [Deterministic Workflows vs. Plain LLM Calls vs. Agents](#deterministic-workflows-vs-plain-llm-calls-vs-agents)
- [Quick Reference Glossary of Agentic Terms](#quick-reference-glossary-of-agentic-terms)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
- [Setup Check & Code Walkthrough](#setup-check--code-walkthrough)
  - [1. Environment Verification & First Model Call](#1-environment-verification--first-model-call)
  - [2. Workflow vs. Single LLM Call vs. Minimal Agent Loop](#2-workflow-vs-single-llm-call-vs-minimal-agent-loop)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [Exercises - Day 1](#-exercises---day-1)

---

## Core Concepts: From AI to Agentic AI

Teams often throw these six terms around interchangeably, which leads to bad architecture decisions. Each layer builds directly on the one underneath it:

- **Artificial Intelligence (AI):** The broad engineering field of building software that performs tasks usually associated with human reasoning or perception—search, scheduling, chess engines, rule-based fraud checks, and modern neural networks all fall under AI.
- **Machine Learning (ML):** A subset of AI where instead of writing explicit `if/else` rules by hand, you train a statistical model on historical data to learn patterns (for example, predicting house prices, classifying spam, or ranking search results).
- **Deep Learning (DL):** A subset of ML built on multi-layer neural networks. Deep learning excels at messy, unstructured inputs like images, audio, and natural language because the network learns its own internal feature representations instead of relying on hand-engineered features.
- **Generative AI:** Deep learning models trained to produce new content—text, code, images, audio, or structured JSON—rather than only classifying or scoring existing inputs.
- **Large Language Models (LLMs):** Text-and-code generative models (such as GPT-4o, Claude, Llama, or Qwen) built on the Transformer architecture and trained on massive corpora to predict the next token given prior context. Instruction-tuned LLMs can follow directions, summarize text, and emit structured tool calls.
- **Agentic AI:** A software architecture that places an LLM inside a **control loop** with access to **tools, state/memory, and guardrails**. Rather than answering once and stopping, an agent observes its environment, decides which tool or step to run next, reads the tool's output, and iterates until the goal is achieved or a hard limit stops it.

Here is how those layers nest together:

```text
+-------------------------------------------------------------------------+
| Artificial Intelligence (AI)                                            |
|   Rule engines, search, planning, optimization, and learning systems    |
|   +-----------------------------------------------------------------+   |
|   | Machine Learning (ML)                                           |   |
|   |   Statistical models trained on data (trees, SVMs, regressions) |   |
|   |   +---------------------------------------------------------+   |   |
|   |   | Deep Learning (DL)                                      |   |   |
|   |   |   Multi-layer neural networks (CNNs, RNNs, Transformers)|   |   |
|   |   |   +-------------------------------------------------+   |   |   |
|   |   |   | Generative AI                                   |   |   |   |
|   |   |   |   Models that generate text, code, images, audio|   |   |   |
|   |   |   |   +-----------------------------------------+   |   |   |   |
|   |   |   |   | Large Language Models (LLMs)            |   |   |   |   |
|   |   |   |   |   Token-predicting text/code models     |   |   |   |   |
|   |   |   |   +-----------------------------------------+   |   |   |   |
|   |   |   +-------------------------------------------------+   |   |   |
|   |   +---------------------------------------------------------+   |   |
|   +-----------------------------------------------------------------+   |
+-------------------------------------------------------------------------+
                                    |
                                    |  Add: Control Loop + Tools + State +
                                    |       Validation + Guardrails + Limits
                                    v
+-------------------------------------------------------------------------+
| Agentic AI System                                                       |
|   Model decides actions -> App executes tools -> Model reads results -> |
|   Repeats until done, escalated to a human, or stopped by max_steps     |
+-------------------------------------------------------------------------+
```

The main engineering takeaway: **an LLM by itself is stateless and passive**. It only transforms input tokens into output tokens. "Agentic" behavior comes from the Python code you wrap around the model.

---

## Deterministic Workflows vs. Plain LLM Calls vs. Agents

One of the most common mistakes in applied AI is reaching for a multi-step agent when a deterministic workflow or a single structured LLM call would be faster, cheaper, and easier to test.

| Dimension | Deterministic Workflow | Single-Turn LLM Call | Agentic AI System |
| :--- | :--- | :--- | :--- |
| **Control flow** | Hard-coded in Python (`if/else`, DAGs) | One prompt in, one completion out | Dynamic loop; model chooses next tool/action |
| **External actions** | Direct API/DB calls written by you | None (unless your code pre-fetches data) | Calls tools iteratively based on intermediate results |
| **Latency** | Milliseconds | ~0.5–3 seconds | Seconds to minutes (multiple round-trips) |
| **Cost per task** | Near zero | Fixed (1 model call) | Variable (scales with steps and context growth) |
| **Predictability** | 100% reproducible | Moderate (probabilistic text) | Lower; requires validation, step caps, and evals |
| **Best fit** | Known rules, fixed inputs, strict SLAs | Summarizing, drafting, classifying | Open-ended tasks where the path depends on what you discover |

Use a **deterministic workflow** whenever the decision tree is already known and finite (e.g., billing calculations, order refund rules with fixed fields). Use a **single LLM call** when you need language understanding or formatting with a fixed set of inputs. Reach for an **agent** only when the system must inspect intermediate results to decide what to look up or do next (e.g., investigating a production incident across logs, metrics, and docs).

---

## Quick Reference Glossary of Agentic Terms

You will see these terms throughout the next 29 days. Keep this table handy as a quick map:

| Term | Plain-English Meaning |
| :--- | :--- |
| **Tool Calling** | A structured API contract where the model outputs JSON requesting a specific function name and arguments (e.g., `get_order(order_id="ORD-1042")`). Your application executes the function—not the model—and feeds the result back. Covered on Day 4. |
| **Context Window** | The maximum number of tokens (input prompt + conversation history + tool outputs + generated reply) a model can hold in one request. Anything beyond the limit fails the request or gets dropped. Covered on Day 2. |
| **RAG / Agentic RAG** | **Retrieval-Augmented Generation (RAG)** fetches relevant documents before calling the model so answers are grounded in your data. **Agentic RAG** lets the agent decide *whether* to search, rewrite weak queries, and grade retrieved chunks before answering. Covered on Day 14. |
| **MCP (Model Context Protocol)** | An open JSON-RPC protocol that standardizes how agents connect to external tools, databases, and resources—similar to how USB-C standardizes hardware peripherals—so you do not rewrite custom glue code for every tool provider. Covered on Days 15–16. |
| **Guardrails** | Deterministic or model-based checks placed around inputs, tool calls, and outputs to block prompt injection, PII leaks, malformed JSON, unauthorized actions, or runaway costs. Covered on Days 21–22. |
| **HITL (Human-in-the-Loop)** | Pausing an agent before high-risk actions (such as issuing a refund, deleting records, or sending external emails) so a human can inspect the proposed arguments and approve, edit, or reject them. Covered on Day 23. |

---

## How It Works Under the Hood

At the wire level, calling an LLM is just an HTTP `POST` request with a JSON body to `/v1/chat/completions`. Almost every modern provider—OpenAI, Ollama, Groq, OpenRouter, vLLM—implements this same shape.

```text
Single LLM Call:
  [Your Python Script] --HTTP POST (messages)--> [LLM API]
  [Your Python Script] <--HTTP 200 (1 reply)---- [LLM API]
  (Done)

Agentic Loop:
  [Your Python Script] --messages + tool schemas--> [LLM API]
  [Your Python Script] <--tool_call JSON----------- [LLM API]
         |
         v
  [Validate args & run local Python function]
         |
         v
  [Append tool observation to messages]
         |
         +---- loop until final answer or max_steps ----> [LLM API]
```

Because you control the loop in Python, you own the timeout, retry logic, step limits, tool permissions, and logging. Nothing runs on your machine unless your code explicitly executes it.

---

## Setup Check & Code Walkthrough

Both scripts in this folder run with the Python 3.11+ standard library and require zero paid API keys to test.

### 1. Environment Verification & First Model Call

Open [`first_model_call.py`](./first_model_call.py). It checks your Python version and environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_NAME`), then calls `/chat/completions` using `urllib.request`.

If you have not configured an API key yet—or if you pass `--demo`—it runs a deterministic offline mock so you can inspect the exact input/output contract right away.

Run in offline demo mode:

```bash
python3 01_Day_Agentic_AI_Foundations/first_model_call.py --demo
```

Sample output:

```text
=== Environment Check ===
Python version : 3.14.5
Base URL       : https://api.openai.com/v1
Model          : gpt-4o-mini
API key status : not set (demo fallback)
Execution mode : DEMO (offline mock)

Prompt: In two plain sentences, explain the difference between a plain LLM call and an agentic AI system.

=== Model Response ===
A plain LLM call takes a prompt and returns one generated response with no ability to act on external state. An agentic AI system places that model inside a loop where it can call tools, inspect observations, and decide what to do next until a goal is met or a safety limit is reached.

=== Call Metadata ===
Source            : demo
Model             : gpt-4o-mini (mock)
Prompt tokens     : 22
Completion tokens : 72
Latency           : 0.0 ms
```

To run against a live OpenAI-compatible API or local Ollama:

```bash
# Cloud OpenAI-compatible provider
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python3 01_Day_Agentic_AI_Foundations/first_model_call.py "Name two reasons to cap agent steps."

# Or local Ollama (free, no paid key)
export OPENAI_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="llama3.2"
python3 01_Day_Agentic_AI_Foundations/first_model_call.py "Name two reasons to cap agent steps."
```

### 2. Workflow vs. Single LLM Call vs. Minimal Agent Loop

Open [`workflow_vs_agent.py`](./workflow_vs_agent.py). It solves the same shipping-refund question three ways side by side:

1. `deterministic_workflow`: directly looks up `ORDERS` and `POLICIES` in Python. Zero model latency, exact output, but breaks the moment a user asks a question outside that hard-coded path.
2. `single_turn_llm_call`: produces polite text, but without access to live order data it can only give generic advice.
3. `minimal_agent_loop`: runs a bounded `decide -> act -> observe` loop, inspects the order first, learns the shipping tier is `express`, fetches the matching refund policy, and returns a grounded answer with a full step trace.

Run it:

```bash
python3 01_Day_Agentic_AI_Foundations/workflow_vs_agent.py
```

Sample output:

```text
User Request:
  My order ORD-1042 still hasn't arrived and I paid for express shipping. Am I eligible for a shipping refund?

1) Deterministic Workflow (hard-coded rules)
   Output: Order ORD-1042 exceeded the 2-day express window (5 days in transit). Eligible refund: $18.00.

2) Single-Turn Generative LLM Call (no tools)
   Output: I am sorry your order has been delayed. Express shipping delays are often eligible for a refund depending on our policy and how many days the package has been in transit. Please check your order tracking page or contact billing with your order ID so they can verify eligibility.

3) Minimal Agent Loop (dynamic tool selection + observations)
   Step 1: action=get_order args={'order_id': 'ORD-1042'}
            observation={'order_id': 'ORD-1042', 'status': 'in_transit', 'shipping_tier': 'express', 'days_in_transit': 5, 'shipping_fee_usd': 18.0}
   Step 2: action=get_policy args={'shipping_tier': 'express'}
            observation={'shipping_tier': 'express', 'max_days': 2, 'refund_percent': 100}
   Step 3: action=finish args={'answer': 'Yes. Order ORD-1042 uses express shipping (limit: 2 days) and has been in transit for 5 days. You are eligible for a $18.00 refund.'}
   Final Answer: Yes. Order ORD-1042 uses express shipping (limit: 2 days) and has been in transit for 5 days. You are eligible for a $18.00 refund.
```

In `workflow_vs_agent.py`, `decide_next_action()` stands in for an LLM so you can see the control flow clearly without network noise. Over Days 2–5, we will replace that stand-in with real prompts, validated schemas, tool definitions, and a live LLM loop.

---

## Common Pitfalls & Bugs Developers Run Into

1. **Building an agent for a deterministic problem.** If every request follows the exact same 3 database queries and an `if` check, putting an LLM in charge of the control flow adds latency, cost, and flakiness with no benefit.
2. **Assuming the model executes tools itself.** Models only output text (or JSON requesting a tool). If your script ignores tool calls or passes unvalidated arguments directly to a database or shell, bugs and security vulnerabilities follow immediately.
3. **Forgetting to set HTTP timeouts.** Default network calls can hang indefinitely when an upstream LLM provider stalls. Always pass an explicit `timeout` (like `timeout_s=30.0` in `first_model_call.py`).
4. **Hard-coding provider SDK specifics too early.** Sticking to standard `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `MODEL_NAME` environment variables lets you swap between cloud models and local Ollama models without rewriting your application.
5. **Leaking secrets in logs.** Print whether `OPENAI_API_KEY` is configured—never print the raw key value in startup banners or error tracebacks.

---

## 💻 Exercises - Day 1

### Exercise: Level 1

1. Run `first_model_call.py --demo` and pass your own custom prompt argument on the command line. Confirm that the estimated `Prompt tokens` count changes with your prompt length.
2. Open `workflow_vs_agent.py` and run `minimal_agent_loop` with `order_id="ORD-1099"` and with a nonexistent ID like `"ORD-9999"`. Observe how the step trace changes when the order is missing.
3. For each of the following tasks, write a 1–2 sentence justification choosing **Deterministic Workflow**, **Single-Turn LLM Call**, or **Agentic AI**:
   - Calculating monthly sales tax from a CSV of invoices.
   - Turning raw customer meeting notes into a 5-bullet summary.
   - Investigating a failing CI build by reading logs, inspecting recent commits, and checking open issues.

### Exercise: Level 2

1. Extend `first_model_call.py` with a `--system` CLI flag so callers can override the default system prompt, and a `--temperature` flag validated to stay between `0.0` and `2.0`.
2. Add a third tool (`tool_check_carrier_delay`) to `workflow_vs_agent.py` that returns whether a weather hold caused the delay, and update `decide_next_action` to include that observation in the final answer when an order is `in_transit`.
3. Configure either a free local Ollama model (`llama3.2` or `qwen2.5:7b`) or an OpenAI-compatible API key in `.env`, run `first_model_call.py` in live mode, and compare the real latency and token usage against `--demo`.

### Exercise: Level 3

1. Add simple retry-with-backoff logic to `call_live_model` in `first_model_call.py` that retries transient HTTP errors (`429`, `500`, `502`, `503`, `504`) up to 3 times with exponential delay, while failing fast on client errors like `400` or `401`.
2. Add a `--json-log` flag to `first_model_call.py` that appends a structured JSONL record (timestamp, model, latency_ms, prompt_tokens, completion_tokens, status) to a local log file without storing sensitive API keys.
3. Modify `minimal_agent_loop` in `workflow_vs_agent.py` to support a strict `max_steps=2` test case and verify it returns a clean, structured failure status (instead of crashing or looping forever) when the budget is too small to finish the task.

---

[Home](../README.md) | [Day 2 >>](../02_Day_Prompts_System_Instructions_and_Context/README.md)
