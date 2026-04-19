"""Tests for jarvis.core — Jarvis agent: profiles, skills, sessions, memory."""

from __future__ import annotations

import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module-level: inject fake heavy dependencies before jarvis.core is imported.
# This mirrors how CI runs (no anthropic/chromadb/openai installed).
# ---------------------------------------------------------------------------

def _inject_fakes():
    fakes = {
        "anthropic": MagicMock(),
        "chromadb": MagicMock(),
        "sentence_transformers": MagicMock(),
        "openai": MagicMock(),
        "modal": MagicMock(),
    }
    fakes["anthropic"].AsyncAnthropic = MagicMock
    for name, mod in fakes.items():
        sys.modules.setdefault(name, mod)


_inject_fakes()


@pytest.fixture()
def jarvis_instance(tmp_data_dir, monkeypatch):
    """Jarvis with all network/IO patched out."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setattr("jarvis.config.cfg.TRAJECTORY_COLLECTION", False)

    with patch("jarvis.plugins.loader.PluginLoader") as MockLoader:
        MockLoader.return_value.load_all.return_value = 0
        MockLoader.return_value.start_hot_reload.return_value = None
        from jarvis.core import Jarvis
        return Jarvis()


# ── Personality profiles ──────────────────────────────────────────────────

def test_default_profile(jarvis_instance):
    assert jarvis_instance._profile == "default"


def test_set_valid_profile(jarvis_instance):
    jarvis_instance.set_profile("terse")
    assert jarvis_instance._profile == "terse"


def test_set_all_profiles(jarvis_instance):
    for profile in ("default", "professional", "casual", "terse", "verbose"):
        jarvis_instance.set_profile(profile)
        assert jarvis_instance._profile == profile


def test_set_invalid_profile_keeps_current(jarvis_instance):
    jarvis_instance.set_profile("terse")
    jarvis_instance.set_profile("nonexistent_profile")
    assert jarvis_instance._profile == "terse"


# ── Skill activation ──────────────────────────────────────────────────────

def test_activate_unknown_skill_returns_false(jarvis_instance):
    assert jarvis_instance.activate_skill("does_not_exist") is False


def test_activate_builtin_skill(jarvis_instance):
    # Built-in skills are loaded; pick the first one
    skills = jarvis_instance.skills.all()
    if not skills:
        pytest.skip("No built-in skills loaded")
    ok = jarvis_instance.activate_skill(skills[0].name)
    assert ok is True
    assert jarvis_instance._active_skill == skills[0].name


def test_deactivate_skill(jarvis_instance):
    skills = jarvis_instance.skills.all()
    if not skills:
        pytest.skip("No built-in skills loaded")
    jarvis_instance.activate_skill(skills[0].name)
    jarvis_instance.deactivate_skill()
    assert jarvis_instance._active_skill is None


# ── Session management ────────────────────────────────────────────────────

def test_session_id_is_uuid(jarvis_instance):
    sid = jarvis_instance._session_id
    parsed = uuid.UUID(sid)
    assert str(parsed) == sid


def test_new_session_changes_id(jarvis_instance):
    old_id = jarvis_instance._session_id
    jarvis_instance.new_session()
    assert jarvis_instance._session_id != old_id


def test_new_session_clears_transcript(jarvis_instance):
    jarvis_instance._session_transcript.append({"role": "user", "content": "hi"})
    jarvis_instance.new_session()
    assert jarvis_instance._session_transcript == []


# ── Status ────────────────────────────────────────────────────────────────

def test_status_returns_string(jarvis_instance):
    status = jarvis_instance.status()
    assert isinstance(status, str)
    assert len(status) > 0


def test_status_mentions_jarvis(jarvis_instance):
    status = jarvis_instance.status().upper()
    assert "JARVIS" in status or "TOOLS" in status or "FACTS" in status


# ── Memory integration ────────────────────────────────────────────────────

def test_memory_saves_message(jarvis_instance):
    sid = jarvis_instance._session_id
    jarvis_instance.memory.save_message(sid, "user", "hello from test")
    history = jarvis_instance.memory.get_history(sid, limit=5)
    assert any(m.get("content") == "hello from test" for m in history)


def test_memory_fact_round_trip(jarvis_instance):
    jarvis_instance.memory.store_fact("test_key", "test_value")
    facts = {f["key"]: f["value"] for f in jarvis_instance.memory.all_facts()}
    assert facts.get("test_key") == "test_value"


# ── Streaming (mocked client) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_chat_yields_tokens(jarvis_instance, monkeypatch):
    """stream_chat should yield all text tokens and save to memory."""
    # Create event classes whose type().__name__ matches what _stream_agent_loop checks
    RawDeltaEvent = type("RawContentBlockDeltaEvent", (), {})

    def make_text_event(text):
        evt = RawDeltaEvent()
        evt.delta = type("Delta", (), {"text": text, "partial_json": None})()
        return evt

    class FakeStreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for text in ("Hello", ", ", "Sir."):
                yield make_text_event(text)

        async def get_final_message(self):
            return MagicMock(content=[])

    jarvis_instance.client.messages.stream = MagicMock(return_value=FakeStreamCtx())
    jarvis_instance.semantic.store_conversation_snippet = MagicMock()
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")

    tokens = []
    async for token in jarvis_instance.stream_chat("say hello"):
        tokens.append(token)

    assert len(tokens) == 3
    assert "".join(tokens) == "Hello, Sir."


@pytest.mark.asyncio
async def test_chat_saves_to_memory(jarvis_instance, monkeypatch):
    """chat() should store both user and assistant messages in memory."""
    async def fake_loop(msg, voice):
        return "Certainly, Sir."

    monkeypatch.setattr(jarvis_instance, "_run_agent_loop", fake_loop)
    jarvis_instance.semantic.store_conversation_snippet = MagicMock()
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")

    await jarvis_instance.chat("test message")

    sid = jarvis_instance._session_id
    history = jarvis_instance.memory.get_history(sid, limit=10)
    contents = [m["content"] for m in history]
    assert "test message" in contents
    assert "Certainly, Sir." in contents
