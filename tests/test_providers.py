"""Tests for jarvis.providers.router — ProviderRouter, adapters, wrappers."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Inject fakes before heavy imports ────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "ollama", "chromadb", "sentence_transformers", "modal"):
        mod = sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock
    sys.modules["openai"].AsyncOpenAI = MagicMock


_inject_fakes()

from jarvis.providers.router import (  # noqa: E402
    ProviderRouter,
    _to_openai_messages,
    _wrap_openai,
    _wrap_ollama,
    _WrappedResponse,
)


# ── _to_openai_messages ───────────────────────────────────────────────────────

def test_to_openai_messages_no_system():
    msgs = [{"role": "user", "content": "hi"}]
    out = _to_openai_messages(msgs, system="")
    assert len(out) == 1
    assert out[0]["role"] == "user"


def test_to_openai_messages_with_system():
    out = _to_openai_messages([], system="You are JARVIS.")
    assert out[0] == {"role": "system", "content": "You are JARVIS."}


def test_to_openai_messages_flattens_list_content():
    msgs = [{"role": "user", "content": [{"content": "part1"}, {"content": "part2"}]}]
    out = _to_openai_messages(msgs, system="")
    assert "part1" in out[0]["content"]
    assert "part2" in out[0]["content"]


def test_to_openai_messages_preserves_order():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    out = _to_openai_messages(msgs, system="sys")
    assert out[0]["role"] == "system"
    assert [m["content"] for m in out[1:]] == ["a", "b", "c"]


# ── _WrappedResponse / _wrap_openai / _wrap_ollama ────────────────────────────

def test_wrapped_response_structure():
    r = _WrappedResponse("hello")
    assert r.content[0].text == "hello"
    assert r.content[0].type == "text"
    assert r.stop_reason == "end_turn"


def test_wrap_openai():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "the answer"
    wrapped = _wrap_openai(mock_resp)
    assert wrapped.content[0].text == "the answer"


def test_wrap_openai_none_content():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = None
    wrapped = _wrap_openai(mock_resp)
    assert wrapped.content[0].text == ""


def test_wrap_ollama_dict():
    wrapped = _wrap_ollama({"message": {"content": "ollama reply"}})
    assert wrapped.content[0].text == "ollama reply"


def test_wrap_ollama_missing_key():
    wrapped = _wrap_ollama({})
    assert wrapped.content[0].text == ""


# ── ProviderRouter ────────────────────────────────────────────────────────────

@pytest.fixture()
def router(monkeypatch):
    monkeypatch.setattr("jarvis.providers.router.cfg.PROVIDER_ORDER", ["anthropic"])
    monkeypatch.setattr("jarvis.providers.router.cfg.ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("jarvis.providers.router.cfg.MAX_TOKENS", 1024)
    monkeypatch.setattr("jarvis.providers.router.cfg.CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    return ProviderRouter()


@pytest.mark.asyncio
async def test_create_message_anthropic(router, monkeypatch):
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="Hello, Sir.")]
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)
    router._clients["anthropic"] = mock_client

    resp = await router.create_message([{"role": "user", "content": "hi"}])
    assert resp.content[0].text == "Hello, Sir."


@pytest.mark.asyncio
async def test_create_message_falls_back_on_failure(monkeypatch):
    monkeypatch.setattr("jarvis.providers.router.cfg.MAX_TOKENS", 256)
    monkeypatch.setattr("jarvis.providers.router.cfg.CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setattr("jarvis.providers.router.cfg.PROVIDER_ORDER", ["anthropic", "openrouter"])
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENROUTER_API_KEY", "key")
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENROUTER_MODEL", "llama")

    router = ProviderRouter()

    # anthropic fails
    bad_client = MagicMock()
    bad_client.messages.create = AsyncMock(side_effect=RuntimeError("down"))
    router._clients["anthropic"] = bad_client

    # openrouter succeeds
    good_completion = MagicMock()
    good_completion.choices[0].message.content = "fallback answer"
    good_or_client = MagicMock()
    good_or_client.chat.completions.create = AsyncMock(return_value=good_completion)
    router._clients["openrouter"] = good_or_client

    resp = await router.create_message([{"role": "user", "content": "hi"}])
    assert resp.content[0].text == "fallback answer"


@pytest.mark.asyncio
async def test_create_message_all_fail_raises(router):
    bad = MagicMock()
    bad.messages.create = AsyncMock(side_effect=RuntimeError("all bad"))
    router._clients["anthropic"] = bad

    with pytest.raises(RuntimeError, match="All providers failed"):
        await router.create_message([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_health_check_returns_dict(router):
    result = await router.health_check()
    assert isinstance(result, dict)
    assert "anthropic" in result


@pytest.mark.asyncio
async def test_create_message_with_explicit_provider(router):
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="ok")]
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)
    router._clients["anthropic"] = mock_client

    resp = await router.create_message(
        [{"role": "user", "content": "hi"}],
        provider="anthropic",
    )
    assert resp.content[0].text == "ok"


def test_unknown_provider_raises(router):
    import asyncio
    with pytest.raises(ValueError, match="Unknown provider"):
        asyncio.run(
            router._call("bogus_provider", [], "", None, 256)
        )


# ── Additional provider paths ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_openrouter_no_client_raises(monkeypatch):
    monkeypatch.setattr("jarvis.providers.router.cfg.MAX_TOKENS", 256)
    monkeypatch.setattr("jarvis.providers.router.cfg.PROVIDER_ORDER", ["openrouter"])
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENROUTER_API_KEY", "")
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENROUTER_MODEL", "")
    router = ProviderRouter()
    router._clients["openrouter"] = None  # simulates _openrouter() returning None
    with pytest.raises(RuntimeError, match="OpenRouter not configured"):
        await router._call("openrouter", [], "", None, 256)


@pytest.mark.asyncio
async def test_call_openrouter_success(monkeypatch):
    monkeypatch.setattr("jarvis.providers.router.cfg.MAX_TOKENS", 256)
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENROUTER_MODEL", "llama3")
    monkeypatch.setattr("jarvis.providers.router.cfg.PROVIDER_ORDER", ["openrouter"])
    router = ProviderRouter()

    good_completion = MagicMock()
    good_completion.choices[0].message.content = "openrouter reply"
    mock_or = MagicMock()
    mock_or.chat.completions.create = AsyncMock(return_value=good_completion)
    router._clients["openrouter"] = mock_or

    resp = await router._call("openrouter", [{"role": "user", "content": "hi"}], "sys", None, 256)
    assert resp.content[0].text == "openrouter reply"


@pytest.mark.asyncio
async def test_call_openai_compat_no_client_raises(monkeypatch):
    monkeypatch.setattr("jarvis.providers.router.cfg.MAX_TOKENS", 256)
    router = ProviderRouter()
    router._clients["openai_compat"] = None
    with pytest.raises(RuntimeError, match="OpenAI-compat endpoint not configured"):
        await router._call("openai_compat", [], "", None, 256)


@pytest.mark.asyncio
async def test_call_openai_compat_success(monkeypatch):
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENAI_COMPAT_MODEL", "gpt-neo")
    router = ProviderRouter()

    good_completion = MagicMock()
    good_completion.choices[0].message.content = "compat reply"
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=good_completion)
    router._clients["openai_compat"] = mock_client

    resp = await router._call("openai_compat", [{"role": "user", "content": "hey"}], "s", None, 256)
    assert resp.content[0].text == "compat reply"


@pytest.mark.asyncio
async def test_call_ollama_no_client_raises(monkeypatch):
    router = ProviderRouter()
    router._clients["ollama"] = None
    with pytest.raises(RuntimeError, match="Ollama not available"):
        await router._call("ollama", [], "", None, 256)


@pytest.mark.asyncio
async def test_call_ollama_success(monkeypatch):
    monkeypatch.setattr("jarvis.providers.router.cfg.OLLAMA_MODEL", "mistral")
    router = ProviderRouter()

    mock_ollama = MagicMock()
    mock_ollama.chat = AsyncMock(return_value={"message": {"content": "ollama says hi"}})
    router._clients["ollama"] = mock_ollama

    resp = await router._call("ollama", [{"role": "user", "content": "yo"}], "", None, 256)
    assert resp.content[0].text == "ollama says hi"


def test_openai_compat_returns_none_when_no_url(monkeypatch):
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENAI_COMPAT_BASE_URL", "")
    router = ProviderRouter()
    assert router._openai_compat() is None


def test_anthropic_client_reused():
    router = ProviderRouter()
    c1 = router._anthropic()
    c2 = router._anthropic()
    assert c1 is c2


@pytest.mark.asyncio
async def test_create_message_with_tools(router):
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="used tools")]
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)
    router._clients["anthropic"] = mock_client

    resp = await router.create_message(
        [{"role": "user", "content": "search web"}],
        tools=[{"name": "web_search", "description": "Search", "input_schema": {}}],
    )
    kwargs = mock_client.messages.create.call_args[1]
    assert "tools" in kwargs
    assert resp.content[0].text == "used tools"


# ── ImportError fallback paths (lines 38-39, 52-53, 61-62) ───────────────────

def test_openrouter_returns_none_on_import_error():
    """Lines 38-39: _openrouter() returns None when openai not installed."""
    from unittest.mock import patch
    router = ProviderRouter()
    with patch.dict(sys.modules, {"openai": None}):
        result = router._openrouter()
    assert result is None


def test_openai_compat_success_with_url(monkeypatch):
    """Lines 46-51: _openai_compat() creates client when URL is configured."""
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENAI_COMPAT_BASE_URL", "http://localhost:8080")
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENAI_COMPAT_API_KEY", "my-key")
    router = ProviderRouter()
    result = router._openai_compat()
    assert result is not None
    assert "openai_compat" in router._clients


def test_openai_compat_returns_none_on_import_error(monkeypatch):
    """Lines 52-53: _openai_compat() returns None when openai not installed."""
    from unittest.mock import patch
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENAI_COMPAT_BASE_URL", "http://localhost:8080")
    router = ProviderRouter()
    with patch.dict(sys.modules, {"openai": None}):
        result = router._openai_compat()
    assert result is None


def test_ollama_returns_none_on_import_error():
    """Lines 61-62: _ollama() returns None when ollama not installed."""
    from unittest.mock import patch
    router = ProviderRouter()
    with patch.dict(sys.modules, {"ollama": None}):
        result = router._ollama()
    assert result is None


@pytest.mark.asyncio
async def test_health_check_swallows_provider_exception(monkeypatch):
    """Lines 139-140: health_check returns False when a provider method raises."""
    router = ProviderRouter()

    def _raise():
        raise RuntimeError("provider auth failed")

    monkeypatch.setattr(router, "_anthropic", _raise)
    result = await router.health_check()
    assert result["anthropic"] is False
