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
