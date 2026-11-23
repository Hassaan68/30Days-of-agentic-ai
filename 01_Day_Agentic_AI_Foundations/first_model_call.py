"""Day 1: Verify your environment and make your first OpenAI-compatible model call.

Uses only the Python standard library (`urllib.request`) so you can run it
immediately against OpenAI, Ollama, Groq, OpenRouter, or in offline `--demo` mode.

Examples:
    python3 01_Day_Agentic_AI_Foundations/first_model_call.py --demo
    python3 01_Day_Agentic_AI_Foundations/first_model_call.py "What makes an AI system agentic?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROMPT = (
    "In two plain sentences, explain the difference between a plain LLM call "
    "and an agentic AI system."
)


@dataclass(frozen=True)
class ModelConfig:
    """Connection settings for any OpenAI-compatible /chat/completions endpoint."""

    api_key: str
    base_url: str
    model: str
    demo_mode: bool


@dataclass(frozen=True)
class ModelResponse:
    """Normalized subset of an OpenAI-compatible chat completion response."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    source: str


def is_local_endpoint(base_url: str) -> bool:
    """Return True when the URL points to a local runner such as Ollama or vLLM."""
    lowered = base_url.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered


def load_config(force_demo: bool = False) -> ModelConfig:
    """Read environment variables and decide whether to use live or demo mode."""
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    env_demo = os.getenv("AGENT_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}

    missing_key = (not api_key) or api_key == "your-api-key-here"
    demo_mode = force_demo or env_demo or (missing_key and not is_local_endpoint(base_url))

    if is_local_endpoint(base_url) and missing_key:
        # Local OpenAI-compatible servers like Ollama accept any non-empty bearer token.
        api_key = "ollama"

    return ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        demo_mode=demo_mode,
    )


def print_environment_check(config: ModelConfig) -> None:
    """Display Python version and active model configuration without leaking secrets."""
    key_status = "configured" if config.api_key and not config.demo_mode else "not set (demo fallback)"
    if is_local_endpoint(config.base_url) and not config.demo_mode:
        key_status = "local endpoint (no paid key needed)"

    print("=== Environment Check ===")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Base URL       : {config.base_url}")
    print(f"Model          : {config.model}")
    print(f"API key status : {key_status}")
    print(f"Execution mode : {'DEMO (offline mock)' if config.demo_mode else 'LIVE API'}")
    print()


def call_mock_model(prompt: str, model: str) -> ModelResponse:
    """Return a deterministic response so learners can inspect the contract offline."""
    start = time.perf_counter()
    reply = (
        "A plain LLM call takes a prompt and returns one generated response with no "
        "ability to act on external state. An agentic AI system places that model inside "
        "a loop where it can call tools, inspect observations, and decide what to do next "
        "until a goal is met or a safety limit is reached."
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    prompt_tokens = max(1, len(prompt.split()) * 4 // 3)
    completion_tokens = max(1, len(reply.split()) * 4 // 3)
    return ModelResponse(
        content=reply,
        model=f"{model} (mock)",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=elapsed_ms,
        source="demo",
    )


def call_live_model(prompt: str, config: ModelConfig, timeout_s: float = 30.0) -> ModelResponse:
    """Send a single POST request to an OpenAI-compatible /chat/completions endpoint."""
    url = f"{config.base_url}/chat/completions"
    payload = {
        "model": config.model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise, practical AI engineering instructor.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = json.loads(raw_body)
    choice = data["choices"][0]
    content = choice["message"].get("content", "").strip()
    usage = data.get("usage") or {}

    return ModelResponse(
        content=content,
        model=data.get("model", config.model),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        latency_ms=elapsed_ms,
        source="live",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check your setup and run a single chat completion call."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Prompt text to send to the model.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with a deterministic offline mock instead of calling a live API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(force_demo=args.demo)
    print_environment_check(config)

    print(f"Prompt: {args.prompt}\n")

    try:
        if config.demo_mode:
            result = call_mock_model(args.prompt, config.model)
        else:
            result = call_live_model(args.prompt, config)
    except RuntimeError as exc:
        print(f"Error calling model: {exc}", file=sys.stderr)
        print("Tip: Re-run with `--demo` to test the script offline.", file=sys.stderr)
        return 1

    print("=== Model Response ===")
    print(result.content)
    print()
    print("=== Call Metadata ===")
    print(f"Source            : {result.source}")
    print(f"Model             : {result.model}")
    print(f"Prompt tokens     : {result.prompt_tokens}")
    print(f"Completion tokens : {result.completion_tokens}")
    print(f"Latency           : {result.latency_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
