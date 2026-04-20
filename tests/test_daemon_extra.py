"""Additional daemon endpoint tests — export, skills, monitor, providers,
self-improve, key revocation, semantic recall, sandbox, root."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    fakes = {
        "anthropic": MagicMock(),
        "chromadb": MagicMock(),
        "sentence_transformers": MagicMock(),
        "openai": MagicMock(),
        "modal": MagicMock(),
        "apscheduler": MagicMock(),
        "apscheduler.schedulers": MagicMock(),
        "apscheduler.schedulers.asyncio": MagicMock(),
        "apscheduler.triggers": MagicMock(),
        "apscheduler.triggers.cron": MagicMock(),
    }
    fakes["anthropic"].AsyncAnthropic = MagicMock
    for name, mod in fakes.items():
        sys.modules.setdefault(name, mod)


_inject_fakes()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("daemon_extra")

    with patch("jarvis.plugins.loader.PluginLoader") as MockLoader, \
         patch("jarvis.config.cfg.TRAJECTORY_COLLECTION", False):
        MockLoader.return_value.load_all.return_value = 0
        MockLoader.return_value.start_hot_reload.return_value = None
        import os; os.environ["ANTHROPIC_API_KEY"] = "test-placeholder"

        from jarvis.core import Jarvis
        jarvis = Jarvis()
        jarvis.chat = AsyncMock(return_value="ok")
        jarvis.end_session = AsyncMock(return_value="Session ended.")
        jarvis.semantic.recall = MagicMock(return_value=[
            {"text": "remembered", "similarity": 0.9, "metadata": {"type": "fact"}},
        ])
        jarvis.semantic.store_conversation_snippet = MagicMock()
        jarvis.semantic.recall_relevant = MagicMock(return_value="")
        jarvis.learner.build_context_prompt = MagicMock(return_value="")

        # Provider router
        jarvis.provider_router = MagicMock()
        jarvis.provider_router.health_check = AsyncMock(return_value={
            "anthropic": "ok",
            "ollama": "fail",
        })

        from jarvis.daemon import create_app
        app = create_app(jarvis)
        inner_app = getattr(app, "app", app)
        with TestClient(inner_app) as c:
            yield c


# ── Root (no web UI by default) ──────────────────────────────────────────────

def test_root_returns_something(client):
    resp = client.get("/")
    assert resp.status_code == 200


# ── End-session ──────────────────────────────────────────────────────────────

def test_end_session_returns_summary(client):
    resp = client.post("/session/end")
    assert resp.status_code == 200
    assert "summary" in resp.json()


# ── Profile ──────────────────────────────────────────────────────────────────

def test_set_profile(client):
    resp = client.post("/profile", json={"profile": "terse"})
    assert resp.status_code == 200
    assert resp.json()["profile"] == "terse"


# ── Semantic recall ──────────────────────────────────────────────────────────

def test_memory_semantic_recall(client):
    resp = client.get("/memory/semantic", params={"q": "test", "n": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 1


# ── Monitor ──────────────────────────────────────────────────────────────────

def test_add_monitor(client):
    resp = client.post("/monitor", json={
        "name": "docs-watch",
        "target_type": "file",
        "target": "/tmp/docs.txt",
        "action": "summarise",
    })
    assert resp.status_code == 200
    assert "Monitoring" in resp.json()["message"]


def test_list_monitors(client):
    resp = client.get("/monitor")
    assert resp.status_code == 200
    assert "targets" in resp.json()


# ── Export endpoints ─────────────────────────────────────────────────────────

def test_export_markdown(client):
    resp = client.post("/export/markdown")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_export_pdf(client):
    resp = client.post("/export/pdf")
    assert resp.status_code == 200
    assert "message" in resp.json()


# ── Skills ───────────────────────────────────────────────────────────────────

def test_list_skills(client):
    resp = client.get("/skills")
    assert resp.status_code == 200
    assert "skills" in resp.json()


def test_deactivate_skill(client):
    resp = client.post("/skills/deactivate")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_activate_unknown_skill_returns_false(client):
    resp = client.post("/skills/activate", json={"name": "nonexistent"})
    assert resp.status_code == 200
    data = resp.json()
    assert "activated" in data


# ── Trajectories ─────────────────────────────────────────────────────────────

def test_trajectory_stats(client):
    resp = client.get("/trajectories/stats")
    assert resp.status_code == 200


# ── Providers ────────────────────────────────────────────────────────────────

def test_providers_health(client):
    resp = client.get("/providers/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "anthropic" in data


# ── Sandbox ──────────────────────────────────────────────────────────────────

def test_sandbox_backends(client):
    resp = client.get("/sandbox/backends")
    assert resp.status_code == 200
    data = resp.json()
    assert "backends" in data
    assert "preferred" in data


# ── Self-improvement ─────────────────────────────────────────────────────────

def test_self_improve_capabilities(client):
    resp = client.get("/self-improve/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_tools" in data
    assert "open_gaps" in data


def test_self_improve_benchmarks_history(client):
    resp = client.get("/self-improve/benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert "history" in data
    assert "trend" in data


# ── Security key revocation ──────────────────────────────────────────────────

def test_revoke_key_not_found(client):
    resp = client.delete("/security/keys/zz_doesnotexist")
    assert resp.status_code == 200
    assert resp.json()["revoked"] is False


def test_create_and_revoke_key_roundtrip(client):
    # Create a key
    created = client.post("/security/keys", json={"name": "revoke-me", "role": "user"})
    assert created.status_code == 200
    raw = created.json()["key"]
    prefix = raw[:8]

    # Revoke using the prefix
    resp = client.delete(f"/security/keys/{prefix}")
    assert resp.status_code == 200
    assert resp.json()["revoked"] is True


# ── ACP publish validation ───────────────────────────────────────────────────

def test_acp_publish_requires_topic(client):
    resp = client.post("/acp/publish", json={"payload": "nothing"})
    assert resp.status_code == 400
    assert "topic" in resp.json()["detail"].lower()
