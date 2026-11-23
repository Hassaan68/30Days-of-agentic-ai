[<< Day 19](../19_Day_Supervisor_and_Orchestrator_Patterns/README.md) | [Home](../README.md) | [Day 21 >>](../21_Day_Input_and_Output_Guardrails/README.md)

# Day 20: Code Execution & Sandboxing

Giving an agent a Python interpreter turns a text generator into an analyst that can compute exact statistics, transform JSON, and verify math instead of guessing. Calling `exec()` directly inside your agent's host process, however, hands arbitrary code execution—and every secret in your process memory—straight to untrusted model output. Today you learn the concrete threat model for agent-generated code, build a defense-in-depth subprocess runner, and draw the exact boundary for when you must move to containers or microVMs.

## Table of Contents

- [Core Concept & Why It Matters](#core-concept--why-it-matters)
  - [Why `exec()` in the Host Process Is Never Safe](#why-exec-in-the-host-process-is-never-safe)
  - [Concrete Threat Model](#concrete-threat-model)
- [How It Works Under the Hood](#how-it-works-under-the-hood)
  - [Defense-in-Depth Layers in `sandbox_runner.py`](#defense-in-depth-layers-in-sandbox_runnerpy)
  - [When You MUST Graduate to Containers or MicroVMs](#when-you-must-graduate-to-containers-or-microvms)
- [Walkthrough of the Python Example](#walkthrough-of-the-python-example)
- [Common Pitfalls & Bugs Developers Run Into](#common-pitfalls--bugs-developers-run-into)
- [💻 Exercises - Day 20](#-exercises---day-20)

---

## Core Concept & Why It Matters

### Why `exec()` in the Host Process Is Never Safe

Even if you pass `exec(code, {"__builtins__": {}})` inside your agent's Python process, you still share:
- **The same process memory and secrets**: `os.environ` (holding `OPENAI_API_KEY`, database credentials, cloud tokens) and live Python objects live in the same address space.
- **The same Python object graph**: Python is famously hard to sandbox in-process. Attackers routinely traverse `().__class__.__bases__[0].__subclasses__()` to locate file loaders or `os.wrap_close` and regain full shell access.
- **The same CPU thread and event loop**: A simple `while True: pass` or `10 ** 10_000_000` freezes your entire API server with no clean way to kill a Python thread from the outside.

Prompt injection (which you will defend against on Day 21) makes this worse: an agent reading an untrusted document or ticket can be tricked into generating malicious Python and passing it to your code-execution tool.

### Concrete Threat Model

Design your sandbox around four real failure modes—both accidental LLM bugs and deliberate prompt-injection payloads:

| Threat | Example Payload | Impact Without Isolation |
| :--- | :--- | :--- |
| **Infinite loops / CPU exhaustion** | `while True: pass` or catastrophic regex backtracking | Hangs worker thread and starves other users |
| **Memory & output exhaustion** | `x = "A" * 10**9` or `for _ in range(10**6): print("X"*1000)` | Triggers host OOM killer or crashes parent agent when buffering `stdout` |
| **Filesystem & secret exfiltration** | Reading `~/.ssh`, `.env`, or `os.environ["OPENAI_API_KEY"]` | Leaks API keys, customer files, or host credentials |
| **Network abuse & SSRF** | Calling internal metadata endpoints (`169.254.169.254`) or exfiltrating data over HTTP | Lateral movement inside your VPC or unauthorized egress |

---

## How It Works Under the Hood

### Defense-in-Depth Layers in `sandbox_runner.py`

No single check is sufficient on its own. Our runner stacks six layers so a failure in one layer is caught by the next:

```text
  Agent-Generated Python Code
              |
              v
  [Layer 1: Static AST Policy]      Allowlisted imports only; blocks eval/exec/open
              |                     and dunder attribute access (.__subclasses__, etc.)
              v
  [Layer 2: Temp Working Dir]       Fresh tempfile.TemporaryDirectory(); deleted on exit
              |
              v
  [Layer 3: Stripped Environment]   `python -I` with minimal env (NO API keys, NO HOME)
              |
              v
  [Layer 4: POSIX rlimits]          RLIMIT_CPU, RLIMIT_FSIZE=0, RLIMIT_NOFILE, RLIMIT_AS*
              |
              v
  [Layer 5: Wall-Clock Timeout]     Isolated process group + os.killpg(SIGKILL) on timeout
              |
              v
  [Layer 6: Bounded Pipe Reader]    Caps stdout/stderr bytes so output floods cannot OOM parent
```

*\*On Linux, `RLIMIT_AS` caps virtual memory cleanly. On macOS, `dyld` maps a multi-gigabyte shared cache into every process, so low `RLIMIT_AS` values break Python startup or return `EINVAL`; our runner applies `RLIMIT_AS` on Linux and relies on containers/cgroups for strict memory enforcement in production.*

### When You MUST Graduate to Containers or MicroVMs

Be completely honest about what a local subprocess sandbox does and does **not** give you:
- **Good enough for**: local development, curriculum experiments, single-user desktop scripts, or running trusted/semi-trusted math snippets on a non-sensitive machine.
- **NOT good enough for**: multi-tenant SaaS products, agents exposed to untrusted external inputs (web pages, emails, customer tickets), or any host holding production secrets or internal network access. A local subprocess still shares the host kernel, filesystem read permissions for world-readable paths, and host network stack.

When executing untrusted code in production, graduate to OS- or hypervisor-level isolation:
1. **Hardened Docker containers** (good baseline for internal tools):
   ```bash
   docker run --rm --network=none --read-only --cap-drop=ALL \
     --memory=256m --cpus=0.5 --pids-limit=64 --user=65534:65534 \
     python:3.12-slim python -I -c "print(2 + 2)"
   ```
2. **gVisor (`runsc`) or Kata Containers**: intercepts syscalls with a user-space kernel or lightweight VM, shrinking the Linux kernel attack surface.
3. **Firecracker microVMs / Managed Sandboxes (E2B, Modal, Fly Machines)**: dedicated guest kernel per sandbox with sub-second boot times—the standard choice for public multi-tenant coding agents.

---

## Walkthrough of the Python Example

[sandbox_runner.py](./sandbox_runner.py) implements all six defense-in-depth layers using only the Python standard library, plus a built-in 6-case security test suite.

Run the built-in test suite:

```bash
python3 20_Day_Code_Execution_and_Sandboxing/sandbox_runner.py
```

Sample output:

```text
=== Day 20: Code Execution & Sandboxing — Built-in Security Test Suite ===

[PASS] 1. Safe statistical computation (78ms)
       stdout: {"mean_ms": 160, "median_ms": 130}

[PASS] 2. Infinite loop timeout enforcement (810ms)
       stderr: Execution killed after exceeding timeout of 0.8s.

[PASS] 3. Blocked dangerous import (`import os`) (0ms)
       stderr: Blocked by AST policy: | - Line 1: import of 'os' is not in allowlist (...)

[PASS] 4. Blocked Python dunder sandbox escape (0ms)
       stderr: Blocked by AST policy: | - Line 1: dunder attribute access '.__subclasses__' is blocked

[PASS] 5. Stripped env vars (verified at OS subprocess layer with AST bypassed) (40ms)
       stdout: SECRET_VISIBLE=False

[PASS] 6. Output flood truncation (40ms)
       stdout: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
       note:   stdout truncated safely (captured 541 chars)

Summary: 6/6 sandbox security checks passed.
```

Look closely at Test 5: it injects a fake `DEMO_SECRET_API_KEY` into the parent process and intentionally sets `enforce_ast_policy=False` to verify that **even if AST filtering were bypassed**, the subprocess layer itself never exposes host environment variables.

You can also run ad-hoc snippets from the command line:

```bash
python3 20_Day_Code_Execution_and_Sandboxing/sandbox_runner.py \
  --code "import math; print(round(math.pi, 4))"
```

---

## Common Pitfalls & Bugs Developers Run Into

1. **Relying on an import blocklist instead of an allowlist.** Blocking `os` and `subprocess` while forgetting `ctypes`, `importlib`, `shutil`, `posix`, `pty`, `webbrowser`, or `socket` leaves dozens of escape paths wide open. Always default-deny and allowlist only the minimal safe modules your task needs.
2. **Using `proc.kill()` without process groups.** If sandboxed code spawns a child process and you only kill the parent PID on timeout, the grandchild can survive as an orphan burning CPU in the background. Use `start_new_session=True` and `os.killpg(proc.pid, signal.SIGKILL)`.
3. **Using `subprocess.run(..., capture_output=True)` on untrusted code.** `capture_output=True` buffers **unlimited** bytes from `stdout` in parent RAM until the child exits. A 2-line print loop can exhaust your host API server's memory before the timeout fires. Always stream pipes through a bounded byte cap (`_read_bounded`).
4. **Assuming AST checks catch every dynamic escape.** Clever Python expressions (like abusing exception tracebacks, `format()` attribute traversal, or C-extension side effects in allowed libraries) can defeat pure AST filters. Treat AST checks as a fast first filter—never as your sole security boundary.
5. **Forgetting to strip `PYTHONPATH` and user site-packages.** Passing `env={}` without `-I` still lets Python load packages from `~/.local/lib/...` or add the script directory to `sys.path`. Launching with `python -I` disables both.

---

## 💻 Exercises - Day 20

### Exercise: Level 1

1. Run `sandbox_runner.py --code` with three snippets: (a) a valid `decimal` currency calculation, (b) `import urllib.request` (should be blocked), and (c) `open('/etc/passwd').read()` (should be blocked). Inspect the exact `ExecutionResult` output for each.
2. Add a 7th test case to `run_built_in_tests()` that bypasses AST (`enforce_ast_policy=False`) and tries to write a file (`open("pwn.txt", "w").write("x" * 100)`), verifying that `RLIMIT_FSIZE=0` blocks file creation at the OS level on Unix.
3. Add a `max_code_bytes` limit (e.g., 16 KB) to `SandboxConfig` and reject oversized code strings before calling `ast.parse()` (preventing parser CPU/memory spikes on giant inputs).

### Exercise: Level 2

1. Wrap `run_in_sandbox()` as an OpenAI-compatible agent tool (`run_python_code`) with JSON Schema `{"code": str}` and wire it into a short agent loop that formats `ExecutionResult` cleanly back to the model so the model can fix syntax errors or blocked imports on retry.
2. Add support for structured JSON input data: accept an optional `input_data: dict[str, Any]` parameter in `run_in_sandbox()`, write it to a read-only `input.json` inside `tmpdir` (or pass via `stdin`), and test a snippet that reads JSON from `sys.stdin` (allowing only `sys.stdin.read()` safely or pre-injecting a `DATA` constant) to compute summary stats.
3. Add a `max_ast_nodes` check inside `ASTPolicyChecker` (e.g., max 500 AST nodes) and reject deeply nested expressions like `"(" * 1000 + "1" + ")" * 1000` before execution.

### Exercise: Level 3

1. Implement a `DockerSandboxRunner` class with the exact same `run(code, config) -> ExecutionResult` interface that executes code via `docker run --rm --network=none --read-only --cap-drop=ALL --memory=256m --cpus=0.5 --pids-limit=64 --user=65534:65534 python:3.12-slim python -I -`, passing code over `stdin` and preserving bounded stdout/stderr and hard timeouts.
2. Add a `SandboxPolicyProfile` selector (`strict_math` vs. `data_analysis`) that swaps allowed module sets, timeouts, and memory limits per tool tier—and refuses to run `data_analysis` unless the container/microVM backend is active.
3. Write a red-team test suite of 10 adversarial Python payloads (including `importlib.import_module`, `"{0.__class__.__mro__}".format(())`, `getattr(__builtins__, "open")`, fork bombs, and sleep/busy-wait loops) and verify every single one fails safely inside your sandbox with zero leaked secrets and zero lingering processes.

---

[<< Day 19](../19_Day_Supervisor_and_Orchestrator_Patterns/README.md) | [Home](../README.md) | [Day 21 >>](../21_Day_Input_and_Output_Guardrails/README.md)
