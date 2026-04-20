"""Tests for jarvis.security.SecurityMiddleware + edge cases."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_scope(path: str, headers: list | None = None, method: str = "GET") -> dict:
    return {
        "type": "http",
        "path": path,
        "method": method,
        "headers": headers or [],
    }


async def _collect_send(messages: list):
    async def send(msg):
        messages.append(msg)
    return send


@pytest.fixture()
def store(tmp_path):
    return KeyStore(path=tmp_path / "keys.json")


@pytest.fixture()
def limiter():
    return RateLimiter(limit=3, window=60)


@pytest.fixture()
def app_mock():
    return AsyncMock()


@pytest.fixture()
def middleware(app_mock, store, limiter):
    return SecurityMiddleware(app_mock, store, limiter)


# ── Security disabled passthrough ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_passthrough_when_security_disabled(middleware, app_mock, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", False)
    scope = _make_scope("/chat")
    messages = []
    await middleware(scope, None, lambda msg: messages.append(msg))
    app_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_passthrough_for_non_http(middleware, app_mock, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", True)
    scope = {"type": "websocket", "path": "/ws"}
    await middleware(scope, None, lambda m: None)
    app_mock.assert_awaited_once()


# ── Public paths ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/", "/health", "/docs", "/openapi.json", "/metrics", "/ws/abc"])
async def test_public_paths_bypass(middleware, app_mock, monkeypatch, path):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", True)
    scope = _make_scope(path)
    await middleware(scope, None, lambda m: None)
    app_mock.assert_awaited_once()


# ── Bootstrap mode: no keys provisioned ───────────────────────────────────────

@pytest.mark.asyncio
async def test_bootstrap_mode_allows_when_no_keys(middleware, app_mock, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", True)
    scope = _make_scope("/chat")  # no X-Api-Key header, no keys stored
    await middleware(scope, None, lambda m: None)
    app_mock.assert_awaited_once()


# ── Missing key rejected when keys exist ──────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_key_rejected(middleware, app_mock, store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", True)
    store.generate()  # provision at least one key
    scope = _make_scope("/chat")
    sent = []

    async def send(m):
        sent.append(m)

    await middleware(scope, None, send)
    app_mock.assert_not_awaited()
    assert sent[0]["status"] == 401
    body = json.loads(sent[1]["body"])
    assert "Missing" in body["detail"]


# ── Invalid key ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_key_rejected(middleware, app_mock, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", True)
    scope = _make_scope("/chat", headers=[(b"x-api-key", b"jvs_fake")])
    # Provision another key so we're not in bootstrap mode
    middleware.key_store.generate()

    sent = []

    async def send(m):
        sent.append(m)

    await middleware(scope, None, send)
    app_mock.assert_not_awaited()
    assert sent[0]["status"] == 403


# ── Valid user key on user route ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_user_key_allowed(middleware, app_mock, store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", True)
    raw = store.generate(role="user")
    scope = _make_scope("/chat", headers=[(b"x-api-key", raw.encode())])
    await middleware(scope, None, lambda m: None)
    app_mock.assert_awaited_once()


# ── User key blocked from admin route ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_user_key_blocked_from_admin_route(middleware, app_mock, store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", True)
    raw = store.generate(role="user")
    scope = _make_scope("/schedule", headers=[(b"x-api-key", raw.encode())])

    sent = []

    async def send(m):
        sent.append(m)

    await middleware(scope, None, send)
    app_mock.assert_not_awaited()
    assert sent[0]["status"] == 403
    assert b"Admin role" in sent[1]["body"]


# ── Admin key allowed on admin route ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_admin_key_on_admin_route(middleware, app_mock, store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", True)
    raw = store.generate(role="admin")
    scope = _make_scope("/self-improve/status", headers=[(b"x-api-key", raw.encode())])
    await middleware(scope, None, lambda m: None)
    app_mock.assert_awaited_once()


# ── Rate limit blocks after too many calls ────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limit_triggers(monkeypatch, store, app_mock):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SECURITY_ENABLED", True)
    limiter = RateLimiter(limit=2, window=60)
    mid = SecurityMiddleware(app_mock, store, limiter)
    raw = store.generate(role="user")
    scope = _make_scope("/chat", headers=[(b"x-api-key", raw.encode())])

    await mid(scope, None, lambda m: None)
    await mid(scope, None, lambda m: None)

    sent = []

    async def send(m):
        sent.append(m)

    await mid(scope, None, send)
    assert sent[0]["status"] == 429
    assert b"Rate limit" in sent[1]["body"]


# ── RateLimiter edge cases ────────────────────────────────────────────────────

def test_rate_limiter_reset_at_future():
    rl = RateLimiter(limit=10, window=30)
    import time
    now = time.time()
    # No calls yet — reset should be roughly now+window
    assert rl.reset_at("fresh") >= now + 29


def test_rate_limiter_reset_at_active_bucket():
    rl = RateLimiter(limit=10, window=30)
    import time
    before = time.time()
    rl.is_allowed("client")
    after = time.time()
    reset = rl.reset_at("client")
    # Should be window-seconds past the first request
    assert before + 30 <= reset <= after + 30


def test_rate_limiter_remaining_empty_client():
    rl = RateLimiter(limit=7, window=60)
    assert rl.remaining("never_seen") == 7


# ── KeyStore load corruption handling ─────────────────────────────────────────

def test_keystore_handles_corrupt_file(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text("{ not valid json")
    ks = KeyStore(path=path)
    # Should silently initialise empty
    assert ks.has_any() is False


def test_keystore_persistence_includes_calls(tmp_path):
    path = tmp_path / "keys.json"
    ks1 = KeyStore(path=path)
    raw = ks1.generate(role="user", name="persistent")
    ks1.validate(raw)
    ks1.validate(raw)

    ks2 = KeyStore(path=path)
    key = ks2.validate(raw)
    assert key.calls == 3  # 2 from ks1, +1 from the reload validate
