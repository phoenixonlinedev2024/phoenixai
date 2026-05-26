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


# ── Client caching ────────────────────────────────────────────────────────────

def test_openrouter_client_reused(monkeypatch):
    """_openrouter() returns the same object on repeated calls."""
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENROUTER_API_KEY", "key")
    router = ProviderRouter()
    c1 = router._openrouter()
    c2 = router._openrouter()
    assert c1 is not None
    assert c1 is c2


def test_openai_compat_client_reused(monkeypatch):
    """_openai_compat() returns the same object on repeated calls."""
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENAI_COMPAT_BASE_URL", "http://localhost:8080")
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENAI_COMPAT_API_KEY", "key")
    router = ProviderRouter()
    c1 = router._openai_compat()
    c2 = router._openai_compat()
    assert c1 is not None
    assert c1 is c2


# ── Full 4-provider fallback chain ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_message_full_fallback_chain(monkeypatch):
    """anthropic→openrouter→openai_compat all fail; ollama succeeds."""
    monkeypatch.setattr("jarvis.providers.router.cfg.MAX_TOKENS", 128)
    monkeypatch.setattr("jarvis.providers.router.cfg.CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENROUTER_MODEL", "llama")
    monkeypatch.setattr("jarvis.providers.router.cfg.OPENAI_COMPAT_MODEL", "local-model")
    monkeypatch.setattr("jarvis.providers.router.cfg.OLLAMA_MODEL", "mistral")
    monkeypatch.setattr("jarvis.providers.router.cfg.PROVIDER_ORDER",
                        ["anthropic", "openrouter", "openai_compat", "ollama"])

    router = ProviderRouter()
    fail_client = MagicMock()
    fail_client.messages.create = AsyncMock(side_effect=RuntimeError("fail"))
    router._clients["anthropic"] = fail_client

    fail_or = MagicMock()
    fail_or.chat = MagicMock()
    fail_or.chat.completions = MagicMock()
    fail_or.chat.completions.create = AsyncMock(side_effect=RuntimeError("or fail"))
    router._clients["openrouter"] = fail_or

    fail_oa = MagicMock()
    fail_oa.chat = MagicMock()
    fail_oa.chat.completions = MagicMock()
    fail_oa.chat.completions.create = AsyncMock(side_effect=RuntimeError("oa fail"))
    router._clients["openai_compat"] = fail_oa

    ollama_resp = {"message": {"content": "ollama wins"}}
    good_ollama = MagicMock()
    good_ollama.chat = AsyncMock(return_value=ollama_resp)
    router._clients["ollama"] = good_ollama

    resp = await router.create_message([{"role": "user", "content": "hi"}])
    assert resp.content[0].text == "ollama wins"


# ── to_openai_messages with non-dict content block ────────────────────────────

def test_to_openai_messages_non_dict_content_block():
    """List content with a non-dict item falls back to str()."""
    msgs = [{"role": "user", "content": ["plain string item", {"content": "nested"}]}]
    result = _to_openai_messages(msgs, system="")
    assert result[0]["role"] == "user"
    assert "plain string item" in result[0]["content"]
    assert "nested" in result[0]["content"]


# ── _WrappedResponse structure ────────────────────────────────────────────────

def test_wrapped_response_content_block():
    """_WrappedResponse has content list with one text block."""
    resp = _WrappedResponse("hello world")
    assert len(resp.content) == 1
    assert resp.content[0].text == "hello world"
    assert resp.content[0].type == "text"
    assert resp.stop_reason == "end_turn"


def test_wrapped_response_empty_text():
    resp = _WrappedResponse("")
    assert resp.content[0].text == ""


# ── _wrap_openai ──────────────────────────────────────────────────────────────

def test_wrap_openai_extracts_message_content():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "openai response"
    wrapped = _wrap_openai(mock_resp)
    assert wrapped.content[0].text == "openai response"


def test_wrap_openai_none_content_becomes_empty():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = None
    wrapped = _wrap_openai(mock_resp)
    assert wrapped.content[0].text == ""


# ── _wrap_ollama ──────────────────────────────────────────────────────────────

def test_wrap_ollama_dict_response():
    resp = {"message": {"content": "ollama says hello"}}
    wrapped = _wrap_ollama(resp)
    assert wrapped.content[0].text == "ollama says hello"


def test_wrap_ollama_non_dict_response():
    """Non-dict response falls back to str()."""
    resp = "plain string response"
    wrapped = _wrap_ollama(resp)
    assert "plain string response" in wrapped.content[0].text


def test_wrap_ollama_empty_message():
    resp = {"message": {}}
    wrapped = _wrap_ollama(resp)
    assert wrapped.content[0].text == ""


def test_wrap_ollama_missing_message_key():
    resp = {}
    wrapped = _wrap_ollama(resp)
    assert wrapped.content[0].text == ""


# ── ProviderRouter.health_check() ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check_anthropic_ok(monkeypatch):
    """health_check anthropic succeeds when client returns normally."""
    router = ProviderRouter()

    def _ok():
        return MagicMock()

    monkeypatch.setattr(router, "_anthropic", _ok)
    result = await router.health_check()
    assert result["anthropic"] is True


# ── _to_openai_messages dict content with content key ─────────────────────────

def test_to_openai_messages_dict_content_with_content_key():
    """Dict block with 'content' key is extracted correctly."""
    msgs = [{"role": "user", "content": [{"content": "hello from block"}]}]
    result = _to_openai_messages(msgs, system="")
    assert result[0]["content"] == "hello from block"


def test_to_openai_messages_multiple_content_blocks():
    """Multiple dict blocks joined by space."""
    msgs = [{"role": "user", "content": [
        {"content": "first"},
        {"content": "second"},
    ]}]
    result = _to_openai_messages(msgs, system="")
    assert "first" in result[0]["content"]
    assert "second" in result[0]["content"]


def test_to_openai_messages_list_with_string_content():
    """String content passed through as-is."""
    msgs = [{"role": "user", "content": "simple string"}]
    result = _to_openai_messages(msgs, system="")
    assert result[0]["content"] == "simple string"


# ── create_message explicit provider failure raises RuntimeError ──────────────

@pytest.mark.asyncio
async def test_create_message_explicit_provider_failure_raises(monkeypatch):
    """When a single explicit provider fails, RuntimeError is raised."""
    router = ProviderRouter()
    monkeypatch.setattr(
        router, "_anthropic",
        lambda: (_ for _ in ()).throw(RuntimeError("anthropic down")),
    )

    async def _boom(*args, **kwargs):
        raise RuntimeError("anthropic down")

    client_mock = MagicMock()
    client_mock.messages.create = _boom
    monkeypatch.setattr(router, "_anthropic", lambda: client_mock)

    with pytest.raises(RuntimeError, match="All providers failed"):
        await router.create_message(
            messages=[{"role": "user", "content": "hi"}],
            provider="anthropic",
        )


# ── _call() with unknown provider raises ValueError ──────────────────────────

@pytest.mark.asyncio
async def test_call_unknown_provider_raises_value_error():
    """_call() with an unrecognised provider string raises ValueError."""
    router = ProviderRouter()
    with pytest.raises(ValueError, match="Unknown provider"):
        await router._call("bogus_provider", [], "", None, 1024)


# ── _to_openai_messages with no messages (empty list) ────────────────────────

def test_to_openai_messages_empty_messages_no_system():
    result = _to_openai_messages([], system="")
    assert result == []


def test_to_openai_messages_empty_messages_with_system():
    result = _to_openai_messages([], system="You are a bot.")
    assert len(result) == 1
    assert result[0]["role"] == "system"


# ── _to_openai_messages non-dict content block falls back to str() ─────────────

def test_to_openai_messages_non_dict_block_in_list():
    """Non-dict block in a list content falls back to str()."""
    msgs = [{"role": "user", "content": ["plain string block"]}]
    result = _to_openai_messages(msgs, system="")
    assert "plain string block" in result[0]["content"]


# ── _to_openai_messages dict block without 'content' key ─────────────────────

def test_to_openai_messages_dict_block_missing_content_key():
    """Dict block with no 'content' key contributes empty string."""
    msgs = [{"role": "user", "content": [{"other_key": "val"}]}]
    result = _to_openai_messages(msgs, system="")
    assert result[0]["content"] == ""


def test_to_openai_messages_mixed_dict_and_string_blocks():
    """Mixed dict and string blocks are all included in output."""
    msgs = [{"role": "user", "content": [
        {"content": "from dict"},
        "from string",
        {"content": "also dict"},
    ]}]
    result = _to_openai_messages(msgs, system="")
    assert "from dict" in result[0]["content"]
    assert "from string" in result[0]["content"]
    assert "also dict" in result[0]["content"]


# ── ProviderRouter create_message explicit max_tokens ────────────────────────

@pytest.mark.asyncio
async def test_create_message_explicit_max_tokens_overrides_cfg(router):
    """Passing max_tokens to create_message uses it instead of cfg.MAX_TOKENS."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="ok")]
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)
    router._clients["anthropic"] = mock_client

    await router.create_message(
        [{"role": "user", "content": "hi"}],
        max_tokens=42,
    )
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["max_tokens"] == 42


@pytest.mark.asyncio
async def test_create_message_system_prompt_passed_through(router):
    """System prompt passed to create_message is forwarded to the API call."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="ok")]
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)
    router._clients["anthropic"] = mock_client

    await router.create_message(
        [{"role": "user", "content": "hi"}],
        system="You are a test assistant.",
    )
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["system"] == "You are a test assistant."


# ── health_check all-pass path ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check_all_providers_true_when_clients_available(monkeypatch):
    """health_check returns True for each provider that returns a non-None client."""
    router = ProviderRouter()
    for provider in ("anthropic", "openrouter", "openai_compat", "ollama"):
        monkeypatch.setattr(router, f"_{provider}", lambda: MagicMock())
    result = await router.health_check()
    assert all(result.values())


# ── _WrappedResponse stop_reason is always end_turn ──────────────────────────

def test_wrapped_response_stop_reason_always_end_turn():
    for text in ("", "hello", "x" * 1000):
        r = _WrappedResponse(text)
        assert r.stop_reason == "end_turn"
