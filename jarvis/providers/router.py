"""Multi-provider router — Anthropic, OpenRouter, OpenAI-compat, Ollama with fallbacks."""

from __future__ import annotations

import asyncio
from typing import Any

from jarvis.config import cfg


class ProviderRouter:
    """Routes inference requests across providers with automatic fallback.

    Priority order:
    1. Anthropic (primary — best quality)
    2. OpenRouter (100+ free/cheap models)
    3. OpenAI-compatible (any local or remote endpoint)
    4. Ollama (local LLM, fully offline)
    """

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        self._order = cfg.PROVIDER_ORDER  # e.g. ["anthropic", "openrouter", "ollama"]

    def _anthropic(self):
        if "anthropic" not in self._clients:
            import anthropic
            self._clients["anthropic"] = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)
        return self._clients["anthropic"]

    def _openrouter(self):
        if "openrouter" not in self._clients:
            try:
                from openai import AsyncOpenAI
                self._clients["openrouter"] = AsyncOpenAI(
                    api_key=cfg.OPENROUTER_API_KEY or "sk-or-free",
                    base_url="https://openrouter.ai/api/v1",
                )
            except ImportError:
                return None
        return self._clients["openrouter"]

    def _openai_compat(self):
        if "openai_compat" not in self._clients:
            if not cfg.OPENAI_COMPAT_BASE_URL:
                return None
            try:
                from openai import AsyncOpenAI
                self._clients["openai_compat"] = AsyncOpenAI(
                    api_key=cfg.OPENAI_COMPAT_API_KEY or "none",
                    base_url=cfg.OPENAI_COMPAT_BASE_URL,
                )
            except ImportError:
                return None
        return self._clients["openai_compat"]

    def _ollama(self):
        if "ollama" not in self._clients:
            try:
                import ollama as ollama_lib
                self._clients["ollama"] = ollama_lib.AsyncClient(host=cfg.OLLAMA_HOST)
            except ImportError:
                return None
        return self._clients["ollama"]

    async def create_message(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
    ) -> Any:
        """Create a message, trying providers in fallback order."""
        max_tokens = max_tokens or cfg.MAX_TOKENS
        providers = [provider] if provider else self._order

        last_exc = None
        for p in providers:
            try:
                return await self._call(p, messages, system, tools, max_tokens)
            except Exception as exc:
                print(f"[JARVIS Provider] '{p}' failed: {exc}. Trying next...")
                last_exc = exc

        raise RuntimeError(f"All providers failed. Last error: {last_exc}")

    async def _call(self, provider: str, messages, system, tools, max_tokens) -> Any:
        if provider == "anthropic":
            client = self._anthropic()
            kwargs = dict(
                model=cfg.CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            if tools:
                kwargs["tools"] = tools
            return await client.messages.create(**kwargs)

        elif provider == "openrouter":
            client = self._openrouter()
            if not client:
                raise RuntimeError("OpenRouter not configured")
            oai_msgs = _to_openai_messages(messages, system)
            return _wrap_openai(await client.chat.completions.create(
                model=cfg.OPENROUTER_MODEL or "meta-llama/llama-3-8b-instruct:free",
                messages=oai_msgs,
                max_tokens=max_tokens,
            ))

        elif provider == "openai_compat":
            client = self._openai_compat()
            if not client:
                raise RuntimeError("OpenAI-compat endpoint not configured")
            oai_msgs = _to_openai_messages(messages, system)
            return _wrap_openai(await client.chat.completions.create(
                model=cfg.OPENAI_COMPAT_MODEL or "gpt-3.5-turbo",
                messages=oai_msgs,
                max_tokens=max_tokens,
            ))

        elif provider == "ollama":
            client = self._ollama()
            if not client:
                raise RuntimeError("Ollama not available")
            oai_msgs = _to_openai_messages(messages, system)
            resp = await client.chat(model=cfg.OLLAMA_MODEL, messages=oai_msgs)
            return _wrap_ollama(resp)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def health_check(self) -> dict[str, bool]:
        results = {}
        for p in ("anthropic", "openrouter", "openai_compat", "ollama"):
            try:
                client = getattr(self, f"_{p}")()
                results[p] = client is not None
            except Exception:
                results[p] = False
        return results


# ── Adapters ─────────────────────────────────────────────────────────────

def _to_openai_messages(messages: list[dict], system: str) -> list[dict]:
    result = []
    if system:
        result.append({"role": "system", "content": system})
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # Anthropic tool results — flatten to text
            content = " ".join(
                b.get("content", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        result.append({"role": role, "content": str(content)})
    return result


class _WrappedResponse:
    """Minimal Anthropic-compatible response wrapper for OpenAI/Ollama responses."""
    def __init__(self, text: str) -> None:
        self.content = [_TextBlock(text)]
        self.stop_reason = "end_turn"


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


def _wrap_openai(resp) -> _WrappedResponse:
    text = resp.choices[0].message.content or ""
    return _WrappedResponse(text)


def _wrap_ollama(resp) -> _WrappedResponse:
    text = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else str(resp)
    return _WrappedResponse(text)
