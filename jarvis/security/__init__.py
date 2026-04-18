"""Security — API key management, rate limiting, RBAC, webhook signing."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jarvis.config import cfg


@dataclass
class ApiKey:
    key: str
    role: str = "user"      # "user" | "admin"
    name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    calls: int = 0
    last_used: str | None = None

    def to_dict(self) -> dict:
        return {
            "key_prefix": self.key[:8] + "...",
            "role": self.role,
            "name": self.name,
            "created_at": self.created_at,
            "calls": self.calls,
            "last_used": self.last_used,
        }


class RateLimiter:
    """Sliding-window rate limiter (in-memory, per identifier)."""

    def __init__(self, limit: int = 120, window: int = 60) -> None:
        self.limit = limit
        self.window = window
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        self._buckets[identifier] = [t for t in self._buckets[identifier] if now - t < self.window]
        if len(self._buckets[identifier]) >= self.limit:
            return False
        self._buckets[identifier].append(now)
        return True

    def remaining(self, identifier: str) -> int:
        now = time.time()
        active = [t for t in self._buckets.get(identifier, []) if now - t < self.window]
        return max(0, self.limit - len(active))

    def reset_at(self, identifier: str) -> float:
        bucket = self._buckets.get(identifier, [])
        return (min(bucket) + self.window) if bucket else (time.time() + self.window)


class KeyStore:
    """Persistent API key store backed by a JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (cfg.DATA_DIR / "api_keys.json")
        self._keys: dict[str, ApiKey] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    self._keys[k] = ApiKey(
                        key=k,
                        role=v.get("role", "user"),
                        name=v.get("name", ""),
                        created_at=v.get("created_at", ""),
                        calls=v.get("calls", 0),
                        last_used=v.get("last_used"),
                    )
            except Exception:
                pass

    def _save(self) -> None:
        data = {}
        for k, v in self._keys.items():
            data[k] = {
                "role": v.role,
                "name": v.name,
                "created_at": v.created_at,
                "calls": v.calls,
                "last_used": v.last_used,
            }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def generate(self, name: str = "", role: str = "user") -> str:
        raw = "jvs_" + secrets.token_urlsafe(32)
        self._keys[raw] = ApiKey(key=raw, role=role, name=name)
        self._save()
        return raw

    def validate(self, raw_key: str) -> ApiKey | None:
        key = self._keys.get(raw_key)
        if key:
            key.calls += 1
            key.last_used = datetime.now(timezone.utc).isoformat()
            self._save()
        return key

    def revoke(self, raw_key: str) -> bool:
        if raw_key in self._keys:
            del self._keys[raw_key]
            self._save()
            return True
        return False

    def list_keys(self) -> list[dict]:
        return [k.to_dict() for k in self._keys.values()]

    def has_any(self) -> bool:
        return bool(self._keys)


# Routes that require admin role
_ADMIN_ROUTES = {"/schedule", "/monitor", "/trajectories/export", "/security", "/self-improve"}


class SecurityMiddleware:
    """ASGI middleware: API key auth + rate limiting. Disabled when SECURITY_ENABLED=false."""

    def __init__(self, app, key_store: KeyStore, rate_limiter: RateLimiter) -> None:
        self.app = app
        self.key_store = key_store
        self.rate_limiter = rate_limiter

    async def __call__(self, scope, receive, send) -> None:
        if not cfg.SECURITY_ENABLED or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # Always pass through public paths
        if path in ("/", "/health", "/docs", "/openapi.json", "/metrics") or path.startswith("/ws"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw_key = headers.get(b"x-api-key", b"").decode()

        if not raw_key:
            if not self.key_store.has_any():
                # Bootstrap mode: no keys provisioned yet
                await self.app(scope, receive, send)
                return
            await self._respond(send, 401, "Missing X-Api-Key header")
            return

        key = self.key_store.validate(raw_key)
        if not key:
            await self._respond(send, 403, "Invalid API key")
            return

        if any(path.startswith(r) for r in _ADMIN_ROUTES) and key.role != "admin":
            await self._respond(send, 403, "Admin role required")
            return

        if not self.rate_limiter.is_allowed(raw_key):
            await self._respond(send, 429, "Rate limit exceeded")
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _respond(send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [[b"content-type", b"application/json"]]})
        await send({"type": "http.response.body", "body": body})


def sign_payload(payload: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def verify_signature(payload: str, signature: str, secret: str) -> bool:
    return hmac.compare_digest(sign_payload(payload, secret), signature)


# Global instances
key_store = KeyStore()
rate_limiter = RateLimiter(limit=cfg.RATE_LIMIT, window=60)
