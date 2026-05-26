"""Tests for jarvis.security — API keys, rate limiting, RBAC middleware."""

import time
import pytest
from pathlib import Path


@pytest.fixture
def key_store(tmp_path):
    from jarvis.security import KeyStore
    return KeyStore(path=tmp_path / "keys.json")


@pytest.fixture
def rate_limiter():
    from jarvis.security import RateLimiter
    return RateLimiter(limit=5, window=60)


# ── KeyStore ─────────────────────────────────────────────────────────────────

def test_generate_and_validate(key_store):
    raw = key_store.generate(name="test", role="user")
    assert raw.startswith("jvs_")
    key = key_store.validate(raw)
    assert key is not None
    assert key.role == "user"


def test_validate_unknown_key_returns_none(key_store):
    assert key_store.validate("jvs_badkey") is None


def test_generate_admin_key(key_store):
    raw = key_store.generate(name="admin", role="admin")
    key = key_store.validate(raw)
    assert key.role == "admin"


def test_revoke_key(key_store):
    raw = key_store.generate(name="temp")
    assert key_store.validate(raw) is not None
    assert key_store.revoke(raw) is True
    assert key_store.validate(raw) is None


def test_revoke_nonexistent_returns_false(key_store):
    assert key_store.revoke("jvs_doesnotexist") is False


def test_list_keys_returns_masked(key_store):
    key_store.generate(name="a")
    key_store.generate(name="b")
    listing = key_store.list_keys()
    assert len(listing) == 2
    for item in listing:
        assert item["key_prefix"].endswith("...")
        assert len(item["key_prefix"]) < 20  # not the full key


def test_has_any_true_false(key_store):
    assert key_store.has_any() is False
    key_store.generate()
    assert key_store.has_any() is True


def test_calls_incremented(key_store):
    raw = key_store.generate()
    key_store.validate(raw)
    key_store.validate(raw)
    key = key_store.validate(raw)
    assert key.calls == 3


def test_persistence(tmp_path):
    from jarvis.security import KeyStore
    ks1 = KeyStore(path=tmp_path / "keys.json")
    raw = ks1.generate(name="persistent", role="admin")
    # Create a new instance pointing at the same file
    ks2 = KeyStore(path=tmp_path / "keys.json")
    key = ks2.validate(raw)
    assert key is not None
    assert key.role == "admin"
    assert key.name == "persistent"


# ── RateLimiter ──────────────────────────────────────────────────────────────

def test_allows_within_limit(rate_limiter):
    for _ in range(5):
        assert rate_limiter.is_allowed("client1") is True


def test_blocks_over_limit(rate_limiter):
    for _ in range(5):
        rate_limiter.is_allowed("client2")
    assert rate_limiter.is_allowed("client2") is False


def test_remaining_counts_down(rate_limiter):
    assert rate_limiter.remaining("fresh") == 5
    rate_limiter.is_allowed("fresh")
    assert rate_limiter.remaining("fresh") == 4


def test_different_clients_independent(rate_limiter):
    for _ in range(5):
        rate_limiter.is_allowed("heavy_user")
    assert rate_limiter.is_allowed("light_user") is True


# ── HMAC signing ─────────────────────────────────────────────────────────────

def test_sign_and_verify():
    from jarvis.security import sign_payload, verify_signature
    payload = '{"event": "test"}'
    secret = "my-webhook-secret"
    sig = sign_payload(payload, secret)
    assert verify_signature(payload, sig, secret) is True


def test_verify_bad_signature():
    from jarvis.security import verify_signature
    assert verify_signature("data", "badsig", "secret") is False


# ── ApiKey.to_dict ────────────────────────────────────────────────────────────

def test_api_key_to_dict_masks_key():
    from jarvis.security import ApiKey
    k = ApiKey(key="jvs_abcdef1234567890", role="admin", name="test-key")
    d = k.to_dict()
    assert d["key_prefix"].endswith("...")
    assert "jvs_abcde" not in d["key_prefix"] or d["key_prefix"].endswith("...")
    assert d["role"] == "admin"
    assert d["name"] == "test-key"
    assert "created_at" in d
    assert d["calls"] == 0


def test_api_key_to_dict_includes_last_used():
    from jarvis.security import ApiKey
    k = ApiKey(key="jvs_x" * 4, role="user", last_used="2024-01-01T00:00:00Z")
    d = k.to_dict()
    assert d["last_used"] == "2024-01-01T00:00:00Z"


# ── KeyStore.generate defaults ────────────────────────────────────────────────

def test_generate_default_role_is_user(key_store):
    raw = key_store.generate(name="default_role")
    key = key_store.validate(raw)
    assert key.role == "user"


def test_generate_key_starts_with_prefix(key_store):
    raw = key_store.generate()
    assert raw.startswith("jvs_")


def test_key_last_used_updated_on_validate(key_store):
    raw = key_store.generate()
    key = key_store.validate(raw)
    assert key.last_used is not None


# ── RateLimiter edge cases ────────────────────────────────────────────────────

def test_rate_limiter_window_expires():
    from jarvis.security import RateLimiter
    rl = RateLimiter(limit=2, window=0)  # window=0 means all timestamps expire immediately
    rl.is_allowed("x")
    rl.is_allowed("x")
    # All old entries expired — should be allowed again
    assert rl.is_allowed("x") is True


def test_rate_limiter_remaining_never_negative():
    """remaining() returns 0 when over the limit, not a negative number."""
    from jarvis.security import RateLimiter
    rl = RateLimiter(limit=3, window=60)
    for _ in range(5):
        rl.is_allowed("u")
    assert rl.remaining("u") == 0


def test_rate_limiter_reset_at_returns_future():
    """reset_at() is always in the future (or now) when bucket has entries."""
    import time
    from jarvis.security import RateLimiter
    rl = RateLimiter(limit=10, window=60)
    rl.is_allowed("u")
    reset = rl.reset_at("u")
    assert reset > time.time() - 1   # at least now - 1s


def test_rate_limiter_reset_at_empty_bucket():
    """reset_at() for an unknown client returns a future time."""
    import time
    from jarvis.security import RateLimiter
    rl = RateLimiter(limit=10, window=60)
    reset = rl.reset_at("unknown_client")
    assert reset >= time.time()


def test_keystore_generate_multiple_keys_are_unique(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "k.json")
    keys = {ks.generate(name=f"app{i}") for i in range(10)}
    assert len(keys) == 10


def test_keystore_validate_increments_calls_each_time(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "k.json")
    raw = ks.generate(name="counter")
    for _ in range(5):
        ks.validate(raw)
    api_key = ks.validate(raw)
    assert api_key.calls == 6


def test_keystore_list_keys_includes_all(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "k.json")
    ks.generate(name="first")
    ks.generate(name="second", role="admin")
    keys = ks.list_keys()
    names = {k["name"] for k in keys}
    assert names == {"first", "second"}


# ── KeyStore: corrupt JSON file is silently ignored ──────────────────────────

def test_keystore_load_corrupt_json_starts_empty(tmp_path):
    bad_path = tmp_path / "corrupt.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    from jarvis.security import KeyStore
    ks = KeyStore(path=bad_path)
    assert ks.list_keys() == []


# ── KeyStore: missing file starts with empty store ───────────────────────────

def test_keystore_missing_file_starts_empty(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "nonexistent.json")
    assert ks.list_keys() == []


# ── sign_payload produces 64-char hex (SHA-256) ──────────────────────────────

def test_sign_payload_is_sha256_hex():
    from jarvis.security import sign_payload
    sig = sign_payload("hello", "secret")
    assert len(sig) == 64
    assert all(c in "0123456789abcdef" for c in sig)


# ── verify_signature rejects wrong secret ────────────────────────────────────

def test_verify_signature_wrong_secret_returns_false():
    from jarvis.security import sign_payload, verify_signature
    sig = sign_payload("data", "correct-secret")
    assert verify_signature("data", sig, "wrong-secret") is False


# ── RateLimiter: bucket reset when window expires ────────────────────────────

def test_rate_limiter_allows_after_window_expires():
    from jarvis.security import RateLimiter
    import time
    rl = RateLimiter(limit=2, window=1)
    assert rl.is_allowed("x") is True
    assert rl.is_allowed("x") is True
    assert rl.is_allowed("x") is False  # at limit
    time.sleep(1.1)
    assert rl.is_allowed("x") is True  # window expired, allowed again


# ── ApiKey.to_dict masks the key ─────────────────────────────────────────────

def test_api_key_to_dict_key_prefix_format():
    from jarvis.security import ApiKey
    key = ApiKey(key="jvs_abcdef1234567890", role="user", name="test")
    d = key.to_dict()
    assert d["key_prefix"].endswith("...")
    assert d["key_prefix"].startswith("jvs_abc")
    assert "key" not in d or "key_prefix" in d


# ── KeyStore: generate key always starts with jvs_ and is unique ─────────────

def test_keystore_all_generated_keys_start_with_jvs(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "k.json")
    keys = [ks.generate(name=f"k{i}") for i in range(5)]
    assert all(k.startswith("jvs_") for k in keys)
    assert len(set(keys)) == 5  # all unique


# ── SecurityMiddleware ────────────────────────────────────────────────────────

def _make_scope(path: str, headers: dict | None = None, scope_type: str = "http") -> dict:
    raw_headers = [[k.encode(), v.encode()] for k, v in (headers or {}).items()]
    return {"type": scope_type, "path": path, "headers": raw_headers}


async def _collect_responses(middleware, scope) -> list[dict]:
    sent = []

    async def receive():
        return {}

    async def send(msg):
        sent.append(msg)

    await middleware(scope, receive, send)
    return sent


def _make_middleware(tmp_path, *, security_enabled: bool = True):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    rl = RateLimiter(limit=5, window=60)
    inner = AsyncMock()

    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = security_enabled
        mw = SecurityMiddleware(inner, ks, rl)
        mw._mock_cfg = mock_cfg
    return mw, ks, rl, inner


async def test_middleware_passes_through_when_security_disabled(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    rl = RateLimiter(limit=5, window=60)
    inner = AsyncMock()
    scope = _make_scope("/chat")
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = False
        mw = SecurityMiddleware(inner, ks, rl)
        await mw(scope, AsyncMock(), AsyncMock())
    inner.assert_awaited_once()


async def test_middleware_passes_through_non_http_scope(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    rl = RateLimiter(limit=5, window=60)
    inner = AsyncMock()
    scope = _make_scope("/chat", scope_type="websocket")
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = True
        mw = SecurityMiddleware(inner, ks, rl)
        await mw(scope, AsyncMock(), AsyncMock())
    inner.assert_awaited_once()


async def test_middleware_public_paths_pass_through(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    ks.generate(name="exists")  # ensure has_any() is True
    rl = RateLimiter(limit=5, window=60)
    inner = AsyncMock()
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = True
        mw = SecurityMiddleware(inner, ks, rl)
        for path in ("/", "/health", "/docs", "/openapi.json", "/metrics", "/ws/stream"):
            inner.reset_mock()
            scope = _make_scope(path)
            await mw(scope, AsyncMock(), AsyncMock())
            assert inner.await_count == 1, f"Expected pass-through for {path}"


async def test_middleware_missing_key_no_keys_provisioned_bootstrap(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")  # empty — no keys
    rl = RateLimiter(limit=5, window=60)
    inner = AsyncMock()
    scope = _make_scope("/chat")
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = True
        mw = SecurityMiddleware(inner, ks, rl)
        await mw(scope, AsyncMock(), AsyncMock())
    inner.assert_awaited_once()


async def test_middleware_missing_key_returns_401(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    ks.generate(name="provisioned")
    rl = RateLimiter(limit=5, window=60)
    inner = AsyncMock()
    scope = _make_scope("/chat")  # no X-Api-Key header
    responses = []
    async def capture_send(msg):
        responses.append(msg)
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = True
        mw = SecurityMiddleware(inner, ks, rl)
        await mw(scope, AsyncMock(), capture_send)
    inner.assert_not_awaited()
    start = next(r for r in responses if r["type"] == "http.response.start")
    assert start["status"] == 401


async def test_middleware_invalid_key_returns_403(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    ks.generate(name="provisioned")
    rl = RateLimiter(limit=5, window=60)
    inner = AsyncMock()
    scope = _make_scope("/chat", headers={"x-api-key": "jvs_badkey"})
    responses = []
    async def capture_send(msg):
        responses.append(msg)
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = True
        mw = SecurityMiddleware(inner, ks, rl)
        await mw(scope, AsyncMock(), capture_send)
    inner.assert_not_awaited()
    start = next(r for r in responses if r["type"] == "http.response.start")
    assert start["status"] == 403


async def test_middleware_non_admin_on_admin_route_returns_403(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    raw = ks.generate(name="user", role="user")
    rl = RateLimiter(limit=5, window=60)
    inner = AsyncMock()
    scope = _make_scope("/schedule", headers={"x-api-key": raw})
    responses = []
    async def capture_send(msg):
        responses.append(msg)
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = True
        mw = SecurityMiddleware(inner, ks, rl)
        await mw(scope, AsyncMock(), capture_send)
    inner.assert_not_awaited()
    start = next(r for r in responses if r["type"] == "http.response.start")
    assert start["status"] == 403


async def test_middleware_admin_on_admin_route_passes_through(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    raw = ks.generate(name="admin", role="admin")
    rl = RateLimiter(limit=100, window=60)
    inner = AsyncMock()
    scope = _make_scope("/schedule", headers={"x-api-key": raw})
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = True
        mw = SecurityMiddleware(inner, ks, rl)
        await mw(scope, AsyncMock(), AsyncMock())
    inner.assert_awaited_once()


async def test_middleware_rate_limit_exceeded_returns_429(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    raw = ks.generate(name="user", role="user")
    rl = RateLimiter(limit=2, window=60)
    inner = AsyncMock()
    # Exhaust the rate limiter
    rl.is_allowed(raw)
    rl.is_allowed(raw)
    scope = _make_scope("/chat", headers={"x-api-key": raw})
    responses = []
    async def capture_send(msg):
        responses.append(msg)
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = True
        mw = SecurityMiddleware(inner, ks, rl)
        await mw(scope, AsyncMock(), capture_send)
    inner.assert_not_awaited()
    start = next(r for r in responses if r["type"] == "http.response.start")
    assert start["status"] == 429


async def test_middleware_valid_key_passes_through(tmp_path):
    from jarvis.security import KeyStore, RateLimiter, SecurityMiddleware
    from unittest.mock import AsyncMock, patch
    ks = KeyStore(path=tmp_path / "keys.json")
    raw = ks.generate(name="user", role="user")
    rl = RateLimiter(limit=100, window=60)
    inner = AsyncMock()
    scope = _make_scope("/chat", headers={"x-api-key": raw})
    with patch("jarvis.security.cfg") as mock_cfg:
        mock_cfg.SECURITY_ENABLED = True
        mw = SecurityMiddleware(inner, ks, rl)
        await mw(scope, AsyncMock(), AsyncMock())
    inner.assert_awaited_once()


# ── RateLimiter.remaining() edge cases ────────────────────────────────────────

def test_rate_limiter_remaining_full_for_new_client():
    """remaining() returns full limit for a client that hasn't made any requests."""
    from jarvis.security import RateLimiter
    rl = RateLimiter(limit=50, window=60)
    assert rl.remaining("brand_new_client") == 50


def test_rate_limiter_remaining_decrements_with_each_request():
    """remaining() decrements by 1 for each allowed request."""
    from jarvis.security import RateLimiter
    rl = RateLimiter(limit=5, window=60)
    rl.is_allowed("client_x")
    rl.is_allowed("client_x")
    assert rl.remaining("client_x") == 3


# ── KeyStore.has_any() transitions ───────────────────────────────────────────

def test_key_store_has_any_false_when_empty(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "empty.json")
    assert ks.has_any() is False


def test_key_store_has_any_true_after_generate(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "ks.json")
    ks.generate(name="first_key")
    assert ks.has_any() is True


def test_key_store_has_any_false_after_all_revoked(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "ks2.json")
    raw = ks.generate(name="only_key")
    assert ks.has_any() is True
    ks.revoke(raw)
    assert ks.has_any() is False


# ── ApiKey.to_dict() includes calls count ────────────────────────────────────

def test_api_key_to_dict_calls_increments(tmp_path):
    """to_dict() reflects calls count after validate() increments it."""
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "ks3.json")
    raw = ks.generate(name="call_test")
    ks.validate(raw)
    ks.validate(raw)
    key = ks.validate(raw)
    assert key.calls == 3


# ── sign_payload + verify_signature round-trip ───────────────────────────────

def test_sign_and_verify_round_trip():
    from jarvis.security import sign_payload, verify_signature
    sig = sign_payload("test_payload", "my_secret_key")
    assert verify_signature("test_payload", sig, "my_secret_key") is True
    assert verify_signature("other_payload", sig, "my_secret_key") is False


# ── sign_payload is deterministic ────────────────────────────────────────────

def test_sign_payload_is_deterministic():
    from jarvis.security import sign_payload
    sig1 = sign_payload("hello", "secret")
    sig2 = sign_payload("hello", "secret")
    assert sig1 == sig2


def test_sign_payload_different_secret_gives_different_sig():
    from jarvis.security import sign_payload
    sig1 = sign_payload("payload", "secret_a")
    sig2 = sign_payload("payload", "secret_b")
    assert sig1 != sig2


# ── ApiKey.to_dict key_prefix truncation ─────────────────────────────────────

def test_api_key_to_dict_key_prefix_is_first_8_chars_plus_dots(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "k.json")
    raw = ks.generate(name="prefix_test")
    key = ks.validate(raw)
    d = key.to_dict()
    assert d["key_prefix"] == raw[:8] + "..."


def test_api_key_to_dict_does_not_expose_full_key(tmp_path):
    from jarvis.security import KeyStore
    ks = KeyStore(path=tmp_path / "k2.json")
    raw = ks.generate(name="leak_test")
    key = ks.validate(raw)
    d = key.to_dict()
    assert raw not in str(d)


# ── KeyStore persistence — reload preserves attributes ───────────────────────

def test_keystore_reload_preserves_role_and_name(tmp_path):
    from jarvis.security import KeyStore
    path = tmp_path / "reload.json"
    ks1 = KeyStore(path=path)
    raw = ks1.generate(name="myapp", role="admin")
    ks1.validate(raw)

    ks2 = KeyStore(path=path)
    key = ks2.validate(raw)
    assert key is not None
    assert key.role == "admin"
    assert key.name == "myapp"
    assert key.calls == 2


def test_keystore_reload_preserves_calls_count(tmp_path):
    from jarvis.security import KeyStore
    path = tmp_path / "calls.json"
    ks1 = KeyStore(path=path)
    raw = ks1.generate(name="counter")
    for _ in range(4):
        ks1.validate(raw)

    ks2 = KeyStore(path=path)
    key = ks2.validate(raw)
    assert key.calls == 5


# ── RateLimiter limit=1 edge case ────────────────────────────────────────────

def test_rate_limiter_limit_one_second_call_blocked():
    from jarvis.security import RateLimiter
    rl = RateLimiter(limit=1, window=60)
    assert rl.is_allowed("u") is True
    assert rl.is_allowed("u") is False


def test_rate_limiter_different_identifiers_independent():
    from jarvis.security import RateLimiter
    rl = RateLimiter(limit=1, window=60)
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("bob") is True
    assert rl.is_allowed("alice") is False
    assert rl.is_allowed("bob") is False
