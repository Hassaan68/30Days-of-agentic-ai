"""Defense-in-depth Python subprocess sandbox runner for agent-generated code.

IMPORTANT SECURITY BOUNDARY NOTE:
A Python AST check + local subprocess sandbox is a useful guardrail for local
development, learning, and low-risk data transformations—it is NOT a hard
security boundary against a determined attacker. For untrusted or multi-tenant
code execution in production, run inside an isolated container or microVM
(Docker with --network=none --read-only --cap-drop=ALL, gVisor, Firecracker,
or managed sandboxes like E2B / Modal).

Defense-in-depth layers implemented here:
  1. Static AST policy check (module allowlist, blocked builtins, blocked dunders)
  2. Fresh temporary working directory (deleted immediately after execution)
  3. Isolated Python interpreter (`python -I`) with stripped environment variables
  4. Unix OS resource limits (`RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NOFILE`, and
     best-effort `RLIMIT_AS` on Linux)
  5. Hard wall-clock timeout with process-group termination (`os.killpg`)
  6. Bounded stdout/stderr capture to prevent parent-process memory floods

Usage:
  # Run the built-in 6-case security and isolation test suite
  python3 20_Day_Code_Execution_and_Sandboxing/sandbox_runner.py

  # Run an ad-hoc Python snippet inside the sandbox
  python3 20_Day_Code_Execution_and_Sandboxing/sandbox_runner.py --code "import math; print(round(math.pi, 4))"
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore[assignment]


DEFAULT_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "collections",
        "datetime",
        "decimal",
        "fractions",
        "itertools",
        "json",
        "math",
        "re",
        "statistics",
    }
)

BLOCKED_CALL_NAMES: frozenset[str] = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration limits for isolated subprocess execution."""

    timeout_seconds: float = 2.0
    cpu_time_limit_seconds: int = 2
    max_file_bytes: int = 0  # Disallow creating/writing files via OS limit
    max_open_files: int = 32
    max_memory_mb: int = 256
    max_output_bytes: int = 4096
    allowed_imports: frozenset[str] = DEFAULT_ALLOWED_IMPORTS
    enforce_ast_policy: bool = True


@dataclass
class ExecutionResult:
    """Structured result returned to the agent loop after code execution."""

    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    blocked_by_policy: bool = False
    policy_violations: list[str] = field(default_factory=list)
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class ASTPolicyChecker(ast.NodeVisitor):
    """Static AST analyzer enforcing an import allowlist and blocking dangerous calls."""

    def __init__(self, allowed_imports: frozenset[str]) -> None:
        self.allowed_imports = allowed_imports
        self.violations: list[str] = []

    def check(self, code: str) -> list[str]:
        self.violations = []
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return [f"SyntaxError on line {exc.lineno}: {exc.msg}"]
        self.visit(tree)
        return self.violations

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_module = alias.name.split(".")[0]
            if root_module not in self.allowed_imports:
                self.violations.append(
                    f"Line {node.lineno}: import of '{alias.name}' is not in allowlist "
                    f"({sorted(self.allowed_imports)})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level > 0 or not node.module:
            self.violations.append(f"Line {node.lineno}: relative imports are blocked")
        else:
            root_module = node.module.split(".")[0]
            if root_module not in self.allowed_imports:
                self.violations.append(
                    f"Line {node.lineno}: 'from {node.module} import ...' is not in allowlist"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALL_NAMES:
            self.violations.append(
                f"Line {node.lineno}: call to '{node.func.id}()' is blocked by policy"
            )
        elif isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_CALL_NAMES:
            self.violations.append(
                f"Line {node.lineno}: call to '.{node.func.attr}()' is blocked by policy"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.violations.append(
                f"Line {node.lineno}: dunder attribute access '.{node.attr}' is blocked"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") and node.id.endswith("__"):
            self.violations.append(
                f"Line {node.lineno}: dunder identifier '{node.id}' is blocked"
            )
        self.generic_visit(node)


def _build_preexec_limiter(config: SandboxConfig):
    """Create a preexec_fn that applies POSIX rlimits inside the child process."""

    def apply_limits() -> None:
        if resource is None:
            return
        # 1. Hard CPU time cap (kills busy-wait loops at kernel level)
        try:
            cpu_limit = max(1, int(config.cpu_time_limit_seconds))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        except (ValueError, OSError):
            pass

        # 2. Prevent writing files on disk
        try:
            fsize = max(0, int(config.max_file_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        except (ValueError, OSError):
            pass

        # 3. Cap open file descriptors
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = min(config.max_open_files, hard if hard != resource.RLIM_INFINITY else config.max_open_files)
            if target < soft or soft == resource.RLIM_INFINITY:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, target))
        except (ValueError, OSError):
            pass

        # 4. Cap virtual address space on Linux.
        # On macOS, dyld maps a multi-gigabyte shared cache into every process,
        # so setting RLIMIT_AS to 256 MB breaks Python startup or returns EINVAL.
        if sys.platform.startswith("linux") and hasattr(resource, "RLIMIT_AS"):
            try:
                mem_bytes = config.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass

    return apply_limits


def _read_bounded(stream, max_bytes: int, bucket: dict[str, object]) -> None:
    """Drain a pipe while storing at most `max_bytes` to prevent parent OOM."""
    chunks: list[bytes] = []
    collected = 0
    truncated = False
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        if collected < max_bytes:
            remaining = max_bytes - collected
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                collected += remaining
                truncated = True
            else:
                chunks.append(chunk)
                collected += len(chunk)
        else:
            truncated = True
    bucket["data"] = b"".join(chunks)
    bucket["truncated"] = truncated


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the sandboxed process and any children it may have spawned."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass


def run_in_sandbox(
    code: str,
    config: SandboxConfig | None = None,
) -> ExecutionResult:
    """Validate and execute Python code inside an isolated subprocess."""
    cfg = config or SandboxConfig()
    started = time.perf_counter()

    # Layer 1: Static AST policy validation before spawning any process
    if cfg.enforce_ast_policy:
        violations = ASTPolicyChecker(cfg.allowed_imports).check(code)
        if violations:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return ExecutionResult(
                ok=False,
                exit_code=None,
                stdout="",
                stderr="Blocked by AST policy:\n" + "\n".join(f"- {v}" for v in violations),
                duration_ms=elapsed_ms,
                blocked_by_policy=True,
                policy_violations=violations,
            )

    # Layer 2 & 3: Fresh temp directory + stripped environment + `python -I`
    with tempfile.TemporaryDirectory(prefix="agent_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        # Minimal clean environment — NO OPENAI_API_KEY, AWS_*, HOME, or PATH secrets.
        clean_env = {
            "LC_ALL": "C",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        popen_kwargs: dict[str, object] = {
            "cwd": tmpdir,
            "env": clean_env,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "start_new_session": True,  # Isolate in its own process group
        }
        if os.name == "posix":
            popen_kwargs["preexec_fn"] = _build_preexec_limiter(cfg)

        # `-I` runs Python in isolated mode (ignores PYTHON* env vars and user site-packages).
        proc = subprocess.Popen(
            [sys.executable, "-I", str(script_path)],
            **popen_kwargs,  # type: ignore[arg-type]
        )

        out_bucket: dict[str, object] = {"data": b"", "truncated": False}
        err_bucket: dict[str, object] = {"data": b"", "truncated": False}
        t_out = threading.Thread(
            target=_read_bounded, args=(proc.stdout, cfg.max_output_bytes, out_bucket), daemon=True
        )
        t_err = threading.Thread(
            target=_read_bounded, args=(proc.stderr, cfg.max_output_bytes, err_bucket), daemon=True
        )
        t_out.start()
        t_err.start()

        timed_out = False
        try:
            proc.wait(timeout=cfg.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc)
            proc.wait(timeout=2)

        t_out.join(timeout=2)
        t_err.join(timeout=2)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        stdout_text = bytes(out_bucket["data"]).decode("utf-8", errors="replace")
        stderr_text = bytes(err_bucket["data"]).decode("utf-8", errors="replace")
        stdout_trunc = bool(out_bucket["truncated"])
        stderr_trunc = bool(err_bucket["truncated"])

        if stdout_trunc:
            stdout_text += f"\n... [truncated at {cfg.max_output_bytes} bytes]"
        if stderr_trunc:
            stderr_text += f"\n... [truncated at {cfg.max_output_bytes} bytes]"
        if timed_out:
            stderr_text = (
                f"Execution killed after exceeding timeout of {cfg.timeout_seconds}s."
                + (f"\n{stderr_text}" if stderr_text else "")
            )

        return ExecutionResult(
            ok=(not timed_out and proc.returncode == 0),
            exit_code=proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_ms=elapsed_ms,
            timed_out=timed_out,
            stdout_truncated=stdout_trunc,
            stderr_truncated=stderr_trunc,
        )


def run_built_in_tests() -> None:
    """Verify every defense layer: safe run, timeout, AST blocks, env isolation, truncation."""
    print("=== Day 20: Code Execution & Sandboxing — Built-in Security Test Suite ===\n")

    # Inject a fake secret into THIS parent process to verify child isolation.
    os.environ["DEMO_SECRET_API_KEY"] = "sk-demo-super-secret-value"

    tests: list[tuple[str, str, SandboxConfig, str]] = [
        (
            "1. Safe statistical computation",
            (
                "import json, statistics\n"
                "latencies = [110, 125, 130, 145, 290]\n"
                "print(json.dumps({'mean_ms': statistics.mean(latencies), 'median_ms': statistics.median(latencies)}))\n"
            ),
            SandboxConfig(timeout_seconds=2.0),
            "expect_ok",
        ),
        (
            "2. Infinite loop timeout enforcement",
            "while True:\n    pass\n",
            SandboxConfig(timeout_seconds=0.8, cpu_time_limit_seconds=2),
            "expect_timeout",
        ),
        (
            "3. Blocked dangerous import (`import os`)",
            "import os\nprint(os.listdir('.'))\n",
            SandboxConfig(timeout_seconds=2.0),
            "expect_policy_block",
        ),
        (
            "4. Blocked Python dunder sandbox escape",
            "print(().__class__.__bases__[0].__subclasses__()[:2])\n",
            SandboxConfig(timeout_seconds=2.0),
            "expect_policy_block",
        ),
        (
            "5. Stripped env vars (verified at OS subprocess layer with AST bypassed)",
            (
                "import os\n"
                "leaked = os.environ.get('DEMO_SECRET_API_KEY')\n"
                "print('SECRET_VISIBLE=' + str(leaked is not None))\n"
                "print('ENV_KEYS=' + str(sorted(os.environ.keys())))\n"
            ),
            # Bypass AST here ONLY to prove the subprocess layer strips parent secrets too!
            SandboxConfig(timeout_seconds=2.0, enforce_ast_policy=False),
            "expect_no_secret_leak",
        ),
        (
            "6. Output flood truncation",
            "for i in range(5000):\n    print('X' * 100)\n",
            SandboxConfig(timeout_seconds=2.0, max_output_bytes=512),
            "expect_truncated",
        ),
    ]

    passed = 0
    for name, code, cfg, expectation in tests:
        res = run_in_sandbox(code, cfg)
        ok = False
        if expectation == "expect_ok":
            ok = res.ok and '"mean_ms": 160' in res.stdout
        elif expectation == "expect_timeout":
            ok = not res.ok and res.timed_out
        elif expectation == "expect_policy_block":
            ok = not res.ok and res.blocked_by_policy and len(res.policy_violations) > 0
        elif expectation == "expect_no_secret_leak":
            ok = res.ok and "SECRET_VISIBLE=False" in res.stdout
        elif expectation == "expect_truncated":
            ok = res.ok and res.stdout_truncated and len(res.stdout) < 700

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"[{status}] {name} ({res.duration_ms}ms)")
        if res.stdout.strip():
            preview = res.stdout.strip().splitlines()[0][:120]
            print(f"       stdout: {preview}")
        if res.stderr.strip():
            lines = res.stderr.strip().splitlines()
            preview = " | ".join(lines[:2])[:140]
            print(f"       stderr: {preview}")
        if res.stdout_truncated:
            print(f"       note:   stdout truncated safely (captured {len(res.stdout)} chars)")
        print()

    print(f"Summary: {passed}/{len(tests)} sandbox security checks passed.")
    if passed != len(tests):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Python snippets in an isolated subprocess sandbox."
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="Execute a specific Python snippet inside the sandbox.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Wall-clock timeout in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the built-in 6-case sandbox test suite (default when --code is omitted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.code is not None:
        result = run_in_sandbox(
            args.code, SandboxConfig(timeout_seconds=args.timeout)
        )
        print(f"ok={result.ok} exit_code={result.exit_code} duration_ms={result.duration_ms}")
        if result.stdout:
            print("--- stdout ---")
            print(result.stdout)
        if result.stderr:
            print("--- stderr ---")
            print(result.stderr)
        if not result.ok:
            raise SystemExit(1)
        return

    run_built_in_tests()


if __name__ == "__main__":
    main()
