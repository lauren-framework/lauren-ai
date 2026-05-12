"""Inspect the raw token stream a real LLM emits for a given prompt.

Bypasses the agentic loop, tool execution, and SSE framing — calls the
transport's ``complete(..., stream=True)`` directly and prints each
``chunk.delta`` with ``repr()`` so newlines are visible.  Useful when
debugging dense / unformatted model output: you can see exactly what the
model emitted before any client-side rendering.

Usage::

    # Interactive — pass system + user as flags
    uv run python inspect_stream.py \\
        --system "You are concise." \\
        --prompt "Summarise photosynthesis in three bullets."

    # Pipe a longer system prompt from a file
    uv run python inspect_stream.py \\
        --system-file ./agent_system.md \\
        --prompt "List three risks."

    # Compare two prompts back-to-back
    uv run python inspect_stream.py --prompt "A" --prompt "B"

Provider configuration is read from environment variables:

* ``LLM_PROVIDER``  — ``anthropic`` (default), ``openai``, or
  ``ollama``.  ``openai`` works for OpenRouter / any OpenAI-compatible
  endpoint when ``LLM_BASE_URL`` is also set.
* ``LLM_MODEL``     — model identifier.  Required.
* ``LLM_API_KEY``   — credential.  Falls back to provider-specific
  variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
  ``OPENROUTER_API_KEY``).
* ``LLM_BASE_URL``  — optional override (set this for OpenRouter:
  ``https://openrouter.ai/api/v1``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from lauren_ai import LLMConfig
from lauren_ai._transport import Message

TYPE_CHECKING = False
if TYPE_CHECKING:
    from lauren_ai._transport._anthropic import AnthropicTransport
    from lauren_ai._transport._ollama import OllamaTransport
    from lauren_ai._transport._openai import OpenAITransport


def _build_config() -> LLMConfig:
    """Build an :class:`LLMConfig` from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL environment variable is required.")

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get(
        {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "ollama": "OLLAMA_API_KEY",
        }.get(provider, "ANTHROPIC_API_KEY"),
        "",
    )
    if not api_key and provider != "ollama":
        # OpenRouter is OpenAI-compatible; allow OPENROUTER_API_KEY too.
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

    base_url = os.environ.get("LLM_BASE_URL")
    kwargs: dict = {"provider": provider, "model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return LLMConfig(**kwargs)


def _build_transport(cfg: LLMConfig) -> AnthropicTransport | OpenAITransport | OllamaTransport:
    """Pick the transport class matching the configured provider."""
    if cfg.provider == "anthropic":
        from lauren_ai._transport._anthropic import AnthropicTransport

        return AnthropicTransport(cfg)
    if cfg.provider == "openai":
        from lauren_ai._transport._openai import OpenAITransport

        return OpenAITransport(cfg)
    if cfg.provider == "ollama":
        from lauren_ai._transport._ollama import OllamaTransport

        return OllamaTransport(cfg)
    raise SystemExit(f"Unknown provider: {cfg.provider!r}")


async def trace(
    cfg: LLMConfig,
    *,
    system: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Stream a single completion and print every chunk with ``repr()``.

    Returns the full accumulated text so callers can do further assertions.
    """
    transport = _build_transport(cfg)
    print(f"\n{'=' * 70}")
    print(f"  model={cfg.model}  provider={cfg.provider}")
    print(f"{'=' * 70}")
    print(f"PROMPT: {user_prompt!r}\n")

    print("--- STREAMED CHUNK DELTAS (repr'd, \\n preserved) ---")
    full = ""
    stream = await transport.complete(
        [Message.user(user_prompt)],
        model=cfg.model,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    async for chunk in stream:
        if chunk.delta:
            full += chunk.delta
            print(repr(chunk.delta))

    newline_count = full.count("\n")
    print("\n--- FULL ACCUMULATED TEXT (raw, \\n preserved) ---")
    print(full)
    print(f"\n--- NEWLINE COUNT: {newline_count}  TOTAL LEN: {len(full)} ---")
    return full


def _read_system(args: argparse.Namespace) -> str:
    if args.system_file:
        return Path(args.system_file).read_text(encoding="utf-8")
    if args.system:
        return args.system
    return "You are a helpful assistant."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--system",
        help="System prompt as a literal string. Mutually exclusive with --system-file.",
    )
    parser.add_argument(
        "--system-file",
        help="Path to a file containing the system prompt.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="User prompt.  Repeat the flag to run multiple prompts back-to-back.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Cap output tokens (default: 1024).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--env-file",
        help="Optional path to a dotenv file to load before reading environment variables.",
    )
    return parser.parse_args()


async def _amain(args: argparse.Namespace) -> None:
    if args.env_file:
        try:
            from dotenv import load_dotenv  # noqa: PLC0415

            load_dotenv(args.env_file)
        except ImportError:
            print(
                "warning: --env-file requested but python-dotenv is not installed; ignoring",
                file=sys.stderr,
            )

    cfg = _build_config()
    system = _read_system(args)
    for prompt in args.prompt:
        await trace(
            cfg,
            system=system,
            user_prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )


def main() -> None:
    asyncio.run(_amain(_parse_args()))


if __name__ == "__main__":
    main()
