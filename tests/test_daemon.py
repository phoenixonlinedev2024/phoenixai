"""Tests for jarvis.daemon — FastAPI REST endpoints."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Inject fake heavy deps before daemon is imported
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Build the FastAPI app with a fully mocked Jarvis
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("daemon_data")

    with patch("jarvis.plugins.loader.PluginLoader") as MockLoader, \
         patch("jarvis.config.cfg.TRAJECTORY_COLLECTION", False):
        MockLoader.return_value.load_all.return_value = 0
        MockLoader.return_value.start_hot_reload.return_value = None
        import os; os.environ["ANTHROPIC_API_KEY"] = "test-placeholder"

        from jarvis.core import Jarvis
        jarvis = Jarvis()

        # Patch chat to return canned response without calling the API
        jarvis.chat = AsyncMock(return_value="Hello, Sir. All systems nominal.")
        jarvis.semantic.store_conversation_snippet = MagicMock()
        jarvis.semantic.recall_relevant = MagicMock(return_value="")
        jarvis.learner.build_context_prompt = MagicMock(return_value="")

        from jarvis.daemon import create_app
        app = create_app(jarvis)
        # Strip SecurityMiddleware to get the raw FastAPI app for testing
        inner_app = getattr(app, "app", app)
        with TestClient(inner_app, raise_server_exceptions=True) as client:
            yield client


# ── Health / Status ───────────────────────────────────────────────────────

def test_health(app_client):
    resp = app_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "timestamp" in data


def test_status(app_client):
    resp = app_client.get("/status")
    assert resp.status_code == 200
    assert "status" in resp.json()


# ── Chat ──────────────────────────────────────────────────────────────────

def test_chat_returns_response(app_client):
    resp = app_client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "session_id" in data
    assert data["response"] == "Hello, Sir. All systems nominal."


def test_chat_empty_message(app_client):
    resp = app_client.post("/chat", json={"message": ""})
    assert resp.status_code == 200


def test_chat_with_profile(app_client):
    resp = app_client.post("/chat", json={"message": "hi", "profile": "terse"})
    assert resp.status_code == 200


# ── Session ───────────────────────────────────────────────────────────────

def test_new_session(app_client):
    resp = app_client.post("/session/new")
    assert resp.status_code == 200
    assert "session_id" in resp.json()


# ── Memory ────────────────────────────────────────────────────────────────

def test_memory_facts(app_client):
    resp = app_client.get("/memory/facts")
    assert resp.status_code == 200
    assert "facts" in resp.json()


def test_memory_lessons(app_client):
    resp = app_client.get("/memory/lessons")
    assert resp.status_code == 200
    assert "lessons" in resp.json()


def test_memory_gaps(app_client):
    resp = app_client.get("/memory/gaps")
    assert resp.status_code == 200
    assert "gaps" in resp.json()


# ── Tools ─────────────────────────────────────────────────────────────────

def test_tools_list(app_client):
    resp = app_client.get("/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    assert len(data["tools"]) > 0


def test_tools_have_required_fields(app_client):
    data = app_client.get("/tools").json()
    for tool in data["tools"]:
        assert "name" in tool
        assert "description" in tool
        assert "category" in tool


# ── Schedule ──────────────────────────────────────────────────────────────

def test_schedule_list(app_client):
    resp = app_client.get("/schedule")
    assert resp.status_code == 200
    assert "tasks" in resp.json()


def test_schedule_add(app_client):
    resp = app_client.post("/schedule", json={
        "name": "test_job",
        "cron": "0 8 * * *",
        "prompt": "Good morning summary.",
    })
    assert resp.status_code == 200
    assert "message" in resp.json()


# ── Metrics ───────────────────────────────────────────────────────────────

def test_metrics_json(app_client):
    resp = app_client.get("/metrics/json")
    assert resp.status_code == 200
    data = resp.json()
    assert "counters" in data
    assert "histograms" in data
    assert "uptime_seconds" in data


def test_metrics_prometheus(app_client):
    resp = app_client.get("/metrics")
    assert resp.status_code == 200
    assert "jarvis_" in resp.text


# ── Security ─────────────────────────────────────────────────────────────

def test_list_api_keys_empty(app_client):
    resp = app_client.get("/security/keys")
    assert resp.status_code == 200
    assert "keys" in resp.json()


def test_create_api_key(app_client):
    resp = app_client.post("/security/keys", json={"name": "ci-test", "role": "user"})
    assert resp.status_code == 200
    data = resp.json()
    assert "key" in data
    assert data["key"].startswith("jvs_")


def test_list_api_keys_after_create(app_client):
    app_client.post("/security/keys", json={"name": "listed-key", "role": "user"})
    resp = app_client.get("/security/keys")
    keys = resp.json()["keys"]
    assert any(k["name"] == "listed-key" for k in keys)


# ── ACP Bus ───────────────────────────────────────────────────────────────

def test_acp_publish(app_client):
    resp = app_client.post("/acp/publish", json={"topic": "test.event", "payload": "hello"})
    assert resp.status_code == 200
    assert resp.json()["published"] is True


def test_acp_history(app_client):
    app_client.post("/acp/publish", json={"topic": "history.test", "payload": "data"})
    resp = app_client.get("/acp/history")
    assert resp.status_code == 200
    assert "messages" in resp.json()


def test_acp_stats(app_client):
    resp = app_client.get("/acp/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "topics" in data
    assert "history_size" in data


# ── NL Scheduler ─────────────────────────────────────────────────────────

def test_schedule_nl_parse(app_client):
    resp = app_client.get("/schedule/nl/parse", params={"phrase": "every morning"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["cron"] == "0 8 * * *"


def test_schedule_nl_add(app_client):
    resp = app_client.post("/schedule/nl", json={
        "name": "morning_brief",
        "schedule": "every morning",
        "prompt": "Daily briefing.",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["cron"] == "0 8 * * *"
    assert data["name"] == "morning_brief"
