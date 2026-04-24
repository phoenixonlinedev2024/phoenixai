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


# ── Root endpoint without index.html (line 104) ───────────────────────────────

def test_root_without_index_html(app_client, tmp_path):
    """Line 104: GET / returns JSON message when web_ui/index.html is missing."""
    with patch("jarvis.daemon.WEB_UI_DIR", tmp_path):  # empty dir, no index.html
        resp = app_client.get("/")
    assert resp.status_code == 200
    assert "JARVIS API online" in resp.json()["message"]


# ── Chat exception returns HTTP 500 (lines 128-130) ──────────────────────────

def test_chat_exception_returns_500():
    """Lines 128-130: chat endpoint raises HTTPException 500 when jarvis.chat raises."""
    from jarvis.daemon import create_app

    jarvis_mock = MagicMock()
    jarvis_mock.chat = AsyncMock(side_effect=RuntimeError("AI unavailable"))
    jarvis_mock.set_profile = MagicMock()
    jarvis_mock.status = MagicMock(return_value="")
    jarvis_mock._session_id = "err-session"

    app = create_app(jarvis_mock)
    inner_app = getattr(app, "app", app)
    with TestClient(inner_app, raise_server_exceptions=False) as client:
        resp = client.post("/chat", json={"message": "trigger error"})

    assert resp.status_code == 500
    assert "AI unavailable" in resp.json()["detail"]


# ── WebSocket streaming (lines 340-358) ──────────────────────────────────────

def test_websocket_streams_tokens():
    """Lines 340-351: WebSocket endpoint streams tokens then sends done event."""
    from jarvis.daemon import create_app

    async def mock_stream(msg):
        yield "Hello "
        yield "World"

    jarvis_mock = MagicMock()
    jarvis_mock.stream_chat = mock_stream
    jarvis_mock.set_profile = MagicMock()
    jarvis_mock.status = MagicMock(return_value="")
    jarvis_mock._session_id = "ws-session"

    app = create_app(jarvis_mock)
    inner_app = getattr(app, "app", app)

    with TestClient(inner_app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"message": "hello", "profile": "default"})
            msgs = []
            for _ in range(10):
                msg = ws.receive_json()
                msgs.append(msg)
                if msg.get("type") == "done":
                    break

    types = [m["type"] for m in msgs]
    assert "token" in types
    assert "done" in types


def test_websocket_exception_sends_error():
    """Lines 354-358: WebSocket exception path sends error JSON."""
    from jarvis.daemon import create_app

    async def bad_stream(msg):
        raise RuntimeError("stream exploded")
        yield  # make it an async generator

    jarvis_mock = MagicMock()
    jarvis_mock.stream_chat = bad_stream
    jarvis_mock.set_profile = MagicMock()
    jarvis_mock.status = MagicMock(return_value="")

    app = create_app(jarvis_mock)
    inner_app = getattr(app, "app", app)

    with TestClient(inner_app, raise_server_exceptions=False) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"message": "crash me"})
            msgs = []
            for _ in range(5):
                try:
                    msg = ws.receive_json()
                    msgs.append(msg)
                    if msg.get("type") in ("error", "done"):
                        break
                except Exception:
                    break

    error_msgs = [m for m in msgs if m.get("type") == "error"]
    assert len(error_msgs) >= 1
    assert "stream exploded" in error_msgs[0]["content"]


# ── VoiceLoop.start() with VOICE_ENABLED=True (lines 443-446) ────────────────

def test_voice_loop_start_when_enabled(monkeypatch):
    """Lines 443-446: VoiceLoop.start() sets up TTS+STT when VOICE_ENABLED=True."""
    from jarvis.config import cfg
    from jarvis.daemon import VoiceLoop

    monkeypatch.setattr(cfg, "VOICE_ENABLED", True)
    monkeypatch.setattr(cfg, "STT_ENGINE", "pyttsx3")

    fake_tts = MagicMock()
    fake_stt = MagicMock()

    with patch("jarvis.voice.text_to_speech.TTSEngine", return_value=fake_tts), \
         patch("jarvis.voice.speech_to_text.STTEngine", return_value=fake_stt):
        vl = VoiceLoop(MagicMock())
        vl.start(MagicMock())

    fake_tts.speak.assert_called_once()
    fake_stt.start_listening.assert_called_once()
