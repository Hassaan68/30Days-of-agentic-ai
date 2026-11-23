# 30 Days of Agentic AI

A practical, 30-day hands-on guide to building agentic AI systems in Python—starting from plain model calls and tool loops, and ending with a tested, observable, production-ready autonomous agent.

Inspired by the clear, day-by-day structure of [30 Days of Python](https://github.com/asabeneh/30-days-of-python), every day in this repository focuses on one concrete engineering topic. You read how the mechanism works under the hood, run working Python code, learn what breaks in real deployments, and build muscle memory through three levels of exercises.

---

## Who This Curriculum Is For

- **Developers new to AI engineering** who want plain-English definitions (AI vs. ML vs. LLMs vs. Agents) and clean first implementations without hiding everything behind heavy frameworks.
- **Backend and full-stack engineers** who already call LLM APIs and want to build reliable multi-step loops, structured outputs, MCP integrations, and memory systems.
- **Engineers preparing agents for production** who need guardrails, tool policies, human-in-the-loop (HITL) approvals, tamper-evident audit trails, tracing, evals, and cost controls.

You only need comfortable working knowledge of Python (functions, classes, dictionaries, JSON, basic `async`/`await`, and CLI scripts).

---

## How to Work Through the 30 Days

1. **One folder per day.** Each folder (`01_Day_*` through `30_Day_*`) has a `README.md` lesson and runnable `.py` example(s).
2. **Run the code first, then read it line by line.** Most scripts support both live OpenAI-compatible models (OpenAI, Ollama, Groq, OpenRouter, vLLM) and an offline `--demo` mode so you can inspect control flow, tool calls, and failure handling even without spending API credits.
3. **Do not skip the exercises.** Every lesson ends with **Level 1** (fundamentals), **Level 2** (practical extensions), and **Level 3** (production hardening). Building the fixes yourself is where the concepts stick.
4. **Go at your own pace.** The first 10 days can be finished in an hour each; Phase 2 and Phase 3 topics (MCP, sandboxing, HITL approvals, evals, and the Day 30 capstone) reward taking extra time to experiment.

---

## Setup Instructions

### Prerequisites

- **Python 3.11 or newer** (`python3 --version`)
- **Git** to clone the repository
- Either an **OpenAI-compatible API key** (OpenAI, Anthropic via proxy, OpenRouter, Groq, etc.) **or** a local model runner like **Ollama** (free, runs on your machine)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/30-days-of-agentic-ai.git
cd 30-days-of-agentic-ai
```

### 2. Install Dependencies (`uv` or `pip`)

#### Option A: Using `uv` (recommended)

```bash
uv venv --python 3.11
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Or with `pyproject.toml`:

```bash
uv sync
```

#### Option B: Using standard `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your provider settings:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

Export them in your shell (or load them automatically via `python-dotenv` where supported):

```bash
set -a; source .env; set +a
```

### 4. Optional: Run Everything Locally with Ollama (No Paid API Required)

If you want to work through the curriculum without paying for API calls, install [Ollama](https://ollama.com/) and pull a tool-capable local model such as `llama3.2` or `qwen2.5:7b`:

```bash
ollama serve
ollama pull llama3.2
```

Then point your `.env` at Ollama's OpenAI-compatible endpoint:

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_NAME=llama3.2
```

Because almost all examples talk to the standard OpenAI-compatible `/v1/chat/completions` interface, switching between OpenAI, Ollama, vLLM, or OpenRouter only requires changing those three environment variables.

> **No model handy right now?** Most scripts accept `--demo` (or automatically fall back when `AGENT_DEMO_MODE=1` is set) to run against a deterministic local mock model. That lets you test loops, retries, memory, guardrails, approvals, and audit logs immediately.

---

## Curriculum Table of Contents

### Phase 1: Foundations (Days 1–10)

Build the core mechanics from scratch before reaching for frameworks: how models work, how to constrain outputs, how tool calling functions, and how to assemble a controlled agent loop with memory and concurrency.

| Day | Topic | What You Will Build & Learn |
| :-: | :--- | :--- |
| 01 | [AI, ML, Deep Learning, Generative AI & Agentic AI Foundations](./01_Day_Agentic_AI_Foundations/README.md) | Plain-English definitions, agentic glossary, agents vs. plain LLM calls vs. deterministic workflows, setup, and your first model call |
| 02 | [Prompts, System Instructions & Context Limits](./02_Day_Prompts_System_Instructions_and_Context/README.md) | System prompts, instruction hierarchy, context windows, token budgets, and common prompt failure modes |
| 03 | [Structured Outputs & Schema Validation](./03_Day_Structured_Outputs_and_Schema_Validation/README.md) | Strict JSON schemas, Pydantic validation, and safely handling malformed model outputs |
| 04 | [Tool & Function Calling Basics](./04_Day_Tool_and_Function_Calling_Basics/README.md) | Defining tool schemas, parsing tool calls, executing local Python functions, and returning observations |
| 05 | [Building Your First Agent Loop](./05_Day_Building_Your_First_Agent_Loop/README.md) | Implementing the `model -> tool call -> execution -> observation -> next step` loop with max-step limits |
| 06 | [Error Handling & Self-Correction Loops](./06_Day_Error_Handling_and_Self_Correction/README.md) | Feeding validation errors and tool failures back as structured observations so the model can recover |
| 07 | [Short-Term Memory & Conversation State](./07_Day_Short_Term_Memory_and_Conversation_State/README.md) | Managing message history, token-aware context trimming, rolling summaries, and clean state objects |
| 08 | [Long-Term Memory & Vector Retrieval](./08_Day_Long_Term_Memory_and_Vector_Retrieval/README.md) | Embeddings, cosine similarity vector search, and storing/recalling durable facts and user preferences |
| 09 | [Routing & Classification Workflows](./09_Day_Routing_and_Classification_Workflows/README.md) | Intent classification and routing requests between specialized prompts, models, or deterministic code |
| 10 | [Parallel Execution & Map-Reduce Patterns](./10_Day_Parallel_Execution_and_Map_Reduce/README.md) | Running concurrent tool/LLM calls with `asyncio`, bounding concurrency, and aggregating results cleanly |

---

### Phase 2: Patterns & Integrations (Days 11–20)

Move from basic loops to structured reasoning, retrieval, open tool protocols (MCP), crash-resilient state, multi-agent coordination, and safe code execution.

| Day | Topic | What You Will Build & Learn |
| :-: | :--- | :--- |
| 11 | [ReAct & Chain-of-Thought Reasoning](./11_Day_ReAct_and_Chain_of_Thought_Reasoning/README.md) | Structuring explicit Thought/Action/Observation steps before acting to avoid premature or wrong tool calls |
| 12 | [Planning & Task Decomposition](./12_Day_Planning_and_Task_Decomposition/README.md) | Breaking multi-step goals into explicit plans and adapting when intermediate observations invalidate assumptions |
| 13 | [Reflection & Self-Critique Loops](./13_Day_Reflection_and_Self_Critique_Loops/README.md) | Generator-critic loops that review drafts or code against clear rubrics before finishing |
| 14 | [Agentic RAG](./14_Day_Agentic_RAG/README.md) | Letting agents decide when to retrieve, rewrite weak search queries, and grade retrieved context |
| 15 | [Model Context Protocol (MCP) Basics](./15_Day_Model_Context_Protocol_Basics/README.md) | How MCP works over JSON-RPC, connecting an agent client to MCP servers, and discovering tools dynamically |
| 16 | [Building Custom MCP Servers](./16_Day_Building_Custom_MCP_Servers/README.md) | Writing a custom MCP server exposing tools and resources and wiring it to an agent client |
| 17 | [Stateful Workflows & Checkpointing](./17_Day_Stateful_Workflows_and_Checkpointing/README.md) | Persisting workflow state to SQLite across steps so long-running jobs can pause, resume, or recover after crashes |
| 18 | [Multi-Agent Handoffs & Specialization](./18_Day_Multi_Agent_Handoffs_and_Specialization/README.md) | Dividing work across specialized agents with strict role boundaries and explicit handoff state |
| 19 | [Supervisor & Orchestrator Patterns](./19_Day_Supervisor_and_Orchestrator_Patterns/README.md) | Building a central orchestrator that delegates subtasks to workers, validates outputs, and synthesizes final answers |
| 20 | [Code Execution & Sandboxing](./20_Day_Code_Execution_and_Sandboxing/README.md) | Running agent-generated Python safely using subprocess isolation, timeouts, memory limits, and restricted environments |

---

### Phase 3: Production, Safety & Mastery (Days 21–30)

Turn prototypes into systems you can actually ship: defend against untrusted input, enforce least-privilege tool policies, gate risky actions with tamper-proof human approvals, audit every decision, measure quality, control spend, and deploy a streaming API.

| Day | Topic | What You Will Build & Learn |
| :-: | :--- | :--- |
| 21 | [Input & Output Guardrails](./21_Day_Input_and_Output_Guardrails/README.md) | Prompt injection awareness, separating untrusted data from system instructions, and validating outgoing responses |
| 22 | [Tool Execution Policies & Least Privilege](./22_Day_Tool_Execution_Policies_and_Least_Privilege/README.md) | Static and dynamic tool policies (`auto`, `confirm`, `deny`), rate limits, and scoped resource permissions |
| 23 | [Human-in-the-Loop (HITL) Approvals](./23_Day_Human_in_the_Loop_Approvals/README.md) | Pausing execution before high-impact actions and binding approvals to canonical payload hashes to stop TOCTOU tampering |
| 24 | [Audit Logs & Tamper-Evident Receipts](./24_Day_Audit_Logs_and_Tamper_Evident_Receipts/README.md) | Recording structured execution receipts and SHA-256 hash-chained audit logs for approved, denied, and failed actions |
| 25 | [Observability, Tracing & Telemetry](./25_Day_Observability_Tracing_and_Telemetry/README.md) | Tracing LLM calls, tool invocations, span latency, token usage, and error paths across multi-step runs |
| 26 | [Evaluation & Benchmarking Agents](./26_Day_Evaluation_and_Benchmarking_Agents/README.md) | Writing deterministic checks and LLM-as-a-judge evaluations to test tool accuracy, task completion, and regressions |
| 27 | [Cost, Latency & Model Routing Optimization](./27_Day_Cost_Latency_and_Model_Routing/README.md) | Routing easy work to cheaper/faster models, trimming bloat, caching responses, and enforcing per-run spend budgets |
| 28 | [Handling Non-Determinism & Reliability](./28_Day_Handling_Non_Determinism_and_Reliability/README.md) | Idempotency keys for side-effecting tools, exponential backoff retries, loop detection, and graceful fallbacks |
| 29 | [Deploying Agents as APIs & Streaming UX](./29_Day_Deploying_Agents_as_APIs_and_Streaming_UX/README.md) | Exposing agents over HTTP and Server-Sent Events (SSE) so clients see live progress, tool calls, and streamed tokens |
| 30 | [Capstone Project — Production-Ready Autonomous Agent](./30_Day_Capstone_Production_Autonomous_Agent/README.md) | End-to-end production agent combining planning, tools/MCP, memory, policies, HITL approvals, hash-chained audit receipts, tracing, and evals |

---

## Repository Layout

```text
30-days-of-agentic-ai/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── 01_Day_Agentic_AI_Foundations/
├── 02_Day_Prompts_System_Instructions_and_Context/
├── ...
└── 30_Day_Capstone_Production_Autonomous_Agent/
```

---

## Quick Verification

After installing dependencies (or even before installing them, using `--demo` mode), try Day 1 right away:

```bash
python3 01_Day_Agentic_AI_Foundations/first_model_call.py --demo
```

Ready to start? Head straight to [Day 1: AI, ML, Deep Learning, Generative AI & Agentic AI Foundations](./01_Day_Agentic_AI_Foundations/README.md).
