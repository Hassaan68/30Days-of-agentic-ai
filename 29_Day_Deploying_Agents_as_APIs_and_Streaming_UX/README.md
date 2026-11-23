[<< Day 28](../28_Day_Handling_Non_Determinism_and_Reliability/README.md) | [Home](../README.md) | [Day 30 >>](../30_Day_Capstone_Production_Autonomous_Agent/README.md)

# Day 29: Deploying Agents as APIs & Streaming UX

Today we wrap an agent in an HTTP API that streams live progress over Server-Sent Events (SSE) instead of blocking until the entire run completes. You will see why 45-second synchronous `POST` endpoints fail in production web apps, how to design a clean typed SSE event protocol, and how to handle client disconnects and cancellation safely.

## Table of Contents

- [Why Blocking 45-Second POST Requests Fail in Production](#why-blocking-45-second-post-requests-fail-in-production)
- [How SSE Streaming Works Under the Hood](#how-sse-streaming-works-under-the-hood)
  - [The Wire Format (`text/event-stream`)](#the-wire-format-textevent-stream)
  - [Typed Agent Event Protocol](#typed-agent-event-protocol)
  - [Request IDs, Heartbeats & Reverse Proxies](#request-ids-heartbeats--reverse-proxies)
  - [Client Disconnects & Cancellation](#client-disconnects--cancellation)
- [Code Walkthrough](#code-walkthrough)
  - [1. `stdlib_sse_demo.py` (Zero Dependencies)](#1-stdlib_sse_demopy-zero-dependencies)
  - [2. `sse_agent_server.py` (FastAPI + Uvicorn)](#2-sse_agent_serverpy-fastapi--uvicorn)
- [Common Pitfalls & Bugs](#common-pitfalls--bugs)
- [💻 Exercises - Day 29](#-exercises---day-29)

---

## Why Blocking 45-Second POST Requests Fail in Production

A typical multi-step agent plans a task, calls 3–5 tools, runs guardrails, and generates a final answer—taking 10 to 60 seconds end to end. Exposing that behind a plain request/response `POST /v1/agent` endpoint creates four immediate production failures:

1. **Gateway and load-balancer timeouts:** Cloud load balancers (AWS ALB, Cloudflare, Nginx, API Gateway) routinely cut off idle HTTP connections after 30–60 seconds with a `504 Gateway Timeout` even while your agent is still running on the backend.
2. **Duplicate runs from impatient users:** Staring at a spinner for 25 seconds makes users click Submit again or refresh the tab—spawning parallel runs and duplicate tool calls.
3. **Zero visibility into tool actions or HITL pauses:** If an agent needs human approval halfway through a run (`approval_required`), a blocking POST has no way to pause and surface a confirmation dialog cleanly.
4. **Poor perceived latency:** Users perceive a 15-second run as fast when they see a plan in 200ms, live tool badges at 800ms, and streamed tokens at 3 seconds—and as broken when they see a blank screen for 15 seconds.

For unidirectional server-to-client streaming over standard HTTP, **Server-Sent Events (SSE)** is usually a better fit than WebSockets: it works over plain HTTP/1.1 and HTTP/2, traverses corporate firewalls cleanly, needs no custom upgrade handshake, and supports monotonic `id:` fields for replay.

---

## How SSE Streaming Works Under the Hood

```text
  Web / CLI Client                          Agent HTTP API (FastAPI / Stdlib)
         |                                                 |
         |  POST /v1/agent/stream                          |
         |  X-Request-ID: req-stream-full-01               |
         |  {"prompt": "Check checkout-api 502s..."}       |
         | ----------------------------------------------> |
         |                                                 |
         |  HTTP/1.1 200 OK                                |
         |  Content-Type: text/event-stream                |
         |  Cache-Control: no-cache                        |
         |  X-Accel-Buffering: no                          |
         | <---------------------------------------------- |
         |                                                 |
         |  id: 1 / event: plan                            |  (UI renders checklist)
         | <---------------------------------------------- |
         |  id: 2 / event: tool_start                      |  (UI shows running spinner)
         | <---------------------------------------------- |
         |  : keep-alive 2026-11-23T...                    |  (keeps proxies awake)
         | <---------------------------------------------- |
         |  id: 3 / event: tool_result                     |  (UI updates step result)
         | <---------------------------------------------- |
         |  id: 6 / event: approval_required               |  (UI shows Approve/Deny modal)
         | <---------------------------------------------- |
         |  id: 7..9 / event: token                        |  (UI appends text deltas)
         | <---------------------------------------------- |
         |  id: 10 / event: done                           |  (Stream closes cleanly)
         | <---------------------------------------------- |
```

### The Wire Format (`text/event-stream`)

SSE is plain UTF-8 text separated by double newlines (`\n\n`). Each frame contains optional `id:`, `event:`, and one or more `data:` lines:

```text
id: 2
event: tool_start
data: {"request_id":"req-01","seq":2,"tool":"get_service_health","arguments":{"service":"checkout-api"}}

: keep-alive 2026-11-23T22:41:53+00:00

id: 3
event: tool_result
data: {"request_id":"req-01","seq":3,"tool":"get_service_health","status":"degraded"}

```

Lines beginning with `:` are SSE comments. Clients ignore them, while reverse proxies treat them as active traffic and reset their idle-timeout timers.

### Typed Agent Event Protocol

Keep your event schema small, versioned, and predictable so frontend developers do not have to parse markdown heuristics:

| Event Name | When Emitted | Frontend UX Behavior |
| :--- | :--- | :--- |
| `plan` | After task decomposition | Render step checklist |
| `tool_start` | Immediately before invoking a tool | Show active tool badge + args preview |
| `tool_result` | Immediately after tool completes | Mark tool done with compact status |
| `approval_required` | Policy engine requires HITL sign-off | Show Approve / Deny dialog with `approval_id` |
| `token` | Each streamed text chunk | Append `delta` to assistant reply |
| `done` | Run finished cleanly | Close stream, display latency/cost summary |
| `error` | Unhandled failure mid-stream | Display structured error with `request_id` |

### Request IDs, Heartbeats & Reverse Proxies

Three response headers save hours of production debugging:

- **`X-Request-ID`:** Accept a caller-supplied `X-Request-ID` (or generate one) and echo it in both response headers and every SSE JSON payload so frontend errors map 1-to-1 to backend traces and audit logs.
- **`Cache-Control: no-cache`** and **`X-Accel-Buffering: no`:** By default, Nginx buffers up to 4KB–8KB of response body before flushing downstream, turning your streaming UX back into a 30-second stall. `X-Accel-Buffering: no` disables proxy buffering immediately.
- **Heartbeats every 15s:** Emit `: keep-alive\n\n` while waiting on slow tools or long reasoning steps so load balancers never drop an idle connection.

### Client Disconnects & Cancellation

Users close tabs and click "Stop generating" all the time. If your backend keeps calling expensive frontier models and mutating tools after the client disconnects, you burn budget for nobody:

- In **FastAPI/ASGI**, check `await request.is_disconnected()` before each model or tool step and catch `asyncio.CancelledError` cleanly.
- In **blocking socket servers**, catch `BrokenPipeError` / `ConnectionResetError` when flushing frames and abort the remaining steps.
- **Important nuance for side effects:** Never abort a mutating tool halfway through a database write—finish or roll back the atomic tool call, record its audit receipt, and stop *before* starting the next step.

---

## Code Walkthrough

### 1. `stdlib_sse_demo.py` (Zero Dependencies)

Open [`stdlib_sse_demo.py`](./stdlib_sse_demo.py). It implements a real HTTP/1.1 SSE server using Python's built-in `ThreadingHTTPServer` and a line-by-line SSE frame parser using `http.client`:

1. Binds to `127.0.0.1:0` (ephemeral free port) and verifies `GET /health`.
2. Sends `POST /v1/agent/stream` and parses all 10 typed SSE events (`plan`, `tool_start`, `tool_result`, `approval_required`, `token`, `done`) plus a `: keep-alive` heartbeat comment.
3. Opens a second stream, closes the client socket after 3 events, and asserts that the server detects the disconnect and halts cleanly.

Run it right now with zero third-party installs:

```bash
python3 29_Day_Deploying_Agents_as_APIs_and_Streaming_UX/stdlib_sse_demo.py
```

Sample output:

```text
=== 1. Health Check ===
  HTTP 200: {'status': 'ok', 'service': 'stdlib-sse-agent', 'timestamp': '2026-11-23T22:41:52.994681+00:00'}

=== 2. Full Agent SSE Stream (POST /v1/agent/stream) ===
  [id=1 event=plan] {'steps': ['Check checkout-api service health', 'Query recent 502 error logs', 'Prepare rollback pending operator approval']}
  [id=2 event=tool_start] {'tool': 'get_service_health', 'arguments': {'service': 'checkout-api'}}
  [heartbeat] : keep-alive 2026-11-23T22:41:53.145462+00:00
  [id=3 event=tool_result] {'tool': 'get_service_health', 'status': 'degraded', 'error_rate_pct': 14.8}
  [id=4 event=tool_start] {'tool': 'query_error_logs', 'arguments': {'service': 'checkout-api'}}
  [id=5 event=tool_result] {'tool': 'query_error_logs', 'top_error': 'DB pool exhausted after v2.14.0 deploy'}
  [id=6 event=approval_required] {'approval_id': 'apr_demo_901', 'tool': 'rollback_deployment', 'arguments': {'service': 'checkout-api', 'target_version': 'v2.13.9'}}
  [id=7 event=token] delta='checkout-api has a 14.8% 502 rate '
  [id=8 event=token] delta='caused by DB pool exhaustion in v2.14.0. '
  [id=9 event=token] delta='Rollback to v2.13.9 is queued for approval.'
  [id=10 event=done] {'elapsed_ms': 584.7, 'prompt_chars': 56}

  Reassembled streamed text: checkout-api has a 14.8% 502 rate caused by DB pool exhaustion in v2.14.0. Rollback to v2.13.9 is queued for approval.
  Received 10 typed events + 1 keep-alive heartbeat(s).

=== 3. Mid-Stream Client Disconnect & Server Cleanup ===
  [id=1 event=plan] {'steps': ['Check checkout-api service health', 'Query recent 502 error logs', 'Prepare rollback pending operator approval']}
  [id=2 event=tool_start] {'tool': 'get_service_health', 'arguments': {'service': 'checkout-api'}}
  [heartbeat] : keep-alive 2026-11-23T22:41:53.728367+00:00
  [id=3 event=tool_result] {'tool': 'get_service_health', 'status': 'degraded', 'error_rate_pct': 14.8}
  [client] Cancelling stream early after 3 events...
  Client stopped after 3 events; server cleanly recorded disconnect for: ['req-stream-cancel-02']

Server shut down cleanly.
```

### 2. `sse_agent_server.py` (FastAPI + Uvicorn)

Open [`sse_agent_server.py`](./sse_agent_server.py) for the production FastAPI pattern using `StreamingResponse(media_type="text/event-stream")`, Pydantic input validation (`AgentStreamRequest`), `X-Request-ID` propagation, `X-Accel-Buffering: no`, and `await request.is_disconnected()`.

Once `fastapi` and `uvicorn` are installed (`pip install -r requirements.txt`), start the server:

```bash
uvicorn 29_Day_Deploying_Agents_as_APIs_and_Streaming_UX.sse_agent_server:app --reload --port 8000
```

Then stream live events in another terminal with `curl -N` (`-N` disables curl's output buffer):

```bash
curl -N -X POST http://127.0.0.1:8000/v1/agent/stream \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: req-local-001" \
  -d '{"prompt": "Check checkout-api errors and prepare a rollback", "demo": true}'
```

---

## Common Pitfalls & Bugs

1. **Forgetting HTTP 200 cannot change after streaming starts:** Once you yield the first SSE frame, the HTTP status code is already `200 OK` on the wire. If an exception occurs on step 4, you cannot return HTTP 500—you must catch the error inside the generator and emit a structured `event: error` frame so the client knows the stream failed.
2. **Nginx / CDN buffering hiding your stream:** Everything streams locally on `localhost:8000`, then arrives in one giant lump behind Nginx or CloudFront. Always send `X-Accel-Buffering: no` and `Cache-Control: no-cache`, and avoid gzipping `text/event-stream` responses.
3. **Browser `EventSource` only supports `GET`:** The built-in browser `EventSource` API cannot send `POST` JSON bodies or custom `Authorization: Bearer ...` headers. In real web frontends, use `fetch()` with `response.body.getReader()` (or `@microsoft/fetch-event-source`) for authenticated `POST` streams.
4. **Blocking the asyncio event loop inside FastAPI:** Calling synchronous `requests.post()` or heavy CPU code inside `async def stream_agent_events` freezes *every* concurrent SSE connection on that worker. Use `httpx.AsyncClient` or offload blocking calls with `asyncio.to_thread()`.
5. **Leaking raw tool secrets into `tool_result` SSE events:** Everything streamed over SSE is visible in the browser DevTools Network tab. Scrub API keys, internal stack traces, and raw database credentials before emitting `tool_result` or `error` payloads.

---

## 💻 Exercises - Day 29

### Exercise: Level 1

1. Add a `metrics` SSE event emitted right before `done` in both `stdlib_sse_demo.py` and `sse_agent_server.py` carrying `{"input_tokens": ..., "output_tokens": ..., "estimated_cost_usd": ...}`.
2. Modify `stdlib_sse_demo.py` to test an invalid request (`{"prompt": ""}`) and assert the server returns HTTP `422` JSON *before* starting an SSE stream.
3. Add a `--slow` CLI flag to `stdlib_sse_demo.py` that simulates a 3-second tool call emitting a `: keep-alive` heartbeat every 0.5 seconds.

### Exercise: Level 2

1. Implement `Last-Event-ID` resumption in `sse_agent_server.py`: buffer recent SSE frames per `request_id` in an in-memory ring buffer (e.g., last 100 events) and allow `GET /v1/agent/stream/{request_id}` with header `Last-Event-ID: 4` to replay events `5..N` after a flaky mobile disconnect.
2. Add a `POST /v1/agent/approvals/{approval_id}` endpoint to `sse_agent_server.py` backed by an `asyncio.Event` so when `stream_agent_events` emits `approval_required`, the stream actually pauses waiting for the operator's Approve/Deny POST before continuing or aborting.
3. Add CORS middleware and a bearer-token dependency (`Authorization: Bearer <token>`) to `sse_agent_server.py`, rejecting unauthenticated requests with HTTP `401` while keeping `GET /health` public for Kubernetes readiness probes.

### Exercise: Level 3

1. Wire `sse_agent_server.py` to a live OpenAI-compatible `/chat/completions` call with `"stream": true` (using `httpx.AsyncClient`) so real LLM deltas are forwarded live as `event: token` frames, while tool calls emit `tool_start` and `tool_result`.
2. Build a tiny browser HTML/JS page (single `index.html` served at `GET /`) that uses `fetch()` + `ReadableStream` to render the plan checklist, live tool badges, streamed tokens, and a working "Stop" button (`AbortController`) that cleanly triggers server-side cancellation.
3. Load-test `sse_agent_server.py` with 25 concurrent streaming clients using `asyncio` + `httpx`, measuring memory footprint, time-to-first-event (TTFE), and verifying zero leaked background tasks when half the clients disconnect mid-stream.

[<< Day 28](../28_Day_Handling_Non_Determinism_and_Reliability/README.md) | [Home](../README.md) | [Day 30 >>](../30_Day_Capstone_Production_Autonomous_Agent/README.md)
