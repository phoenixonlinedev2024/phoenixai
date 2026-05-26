"""Tests for jarvis.memory.learning — LearningEngine: reflect, build_context."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.memory.learning import LearningEngine


def _fake_client_returning(text: str) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    client.messages.create = AsyncMock(return_value=resp)
    return client


# ── build_context_prompt (pure) ──────────────────────────────────────────────

def test_build_context_prompt_empty_memory(memory_store):
    engine = LearningEngine(memory_store)
    assert engine.build_context_prompt() == ""


def test_build_context_prompt_with_facts(memory_store):
    memory_store.store_fact("language", "python")
    memory_store.store_fact("editor", "vim")
    engine = LearningEngine(memory_store)
    prompt = engine.build_context_prompt()
    assert "Known Facts" in prompt
    assert "language" in prompt
    assert "python" in prompt


def test_build_context_prompt_with_lessons(memory_store):
    memory_store.store_lesson("Always test edge cases.", context="t")
    engine = LearningEngine(memory_store)
    prompt = engine.build_context_prompt()
    assert "Lessons Learned" in prompt
    assert "edge cases" in prompt


def test_build_context_prompt_combines_facts_and_lessons(memory_store):
    memory_store.store_fact("k", "v")
    memory_store.store_lesson("l1")
    engine = LearningEngine(memory_store)
    prompt = engine.build_context_prompt()
    assert "Known Facts" in prompt
    assert "Lessons Learned" in prompt


def test_build_context_prompt_truncates_to_20_facts(memory_store):
    for i in range(25):
        memory_store.store_fact(f"k{i:02d}", f"v{i}")  # zero-pad for lexicographic order
    engine = LearningEngine(memory_store)
    prompt = engine.build_context_prompt()
    # First 20 (k00..k19) included; k20..k24 truncated
    assert "k00" in prompt
    assert "k19" in prompt
    assert "k20" not in prompt
    assert "k24" not in prompt


# ── reflect ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflect_disabled_returns_empty(memory_store, monkeypatch):
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", False)
    engine = LearningEngine(memory_store)
    out = await engine.reflect([{"role": "user", "content": "hi"}], client=MagicMock())
    assert out == {}


@pytest.mark.asyncio
async def test_reflect_persists_lessons_and_facts(memory_store, monkeypatch):
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", True)
    client = _fake_client_returning(
        '{"lessons":["L1","L2"],"facts":{"fav_color":"red"},"capability_gaps":["no OCR tool"]}'
    )
    engine = LearningEngine(memory_store)
    result = await engine.reflect(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        client=client,
    )

    assert result["lessons"] == ["L1", "L2"]
    lessons = memory_store.get_lessons(limit=10)
    assert "L1" in lessons
    assert "L2" in lessons
    facts = {f["key"]: f["value"] for f in memory_store.all_facts()}
    assert facts["fav_color"] == "red"


@pytest.mark.asyncio
async def test_reflect_strips_code_fence(memory_store, monkeypatch):
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", True)
    fenced = '```json\n{"lessons":["ok"],"facts":{},"capability_gaps":[]}\n```'
    client = _fake_client_returning(fenced)
    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "x"}], client=client)
    assert result["lessons"] == ["ok"]


@pytest.mark.asyncio
async def test_reflect_on_api_error_returns_empty(memory_store, monkeypatch):
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", True)
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))
    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "x"}], client=client)
    assert result == {}


@pytest.mark.asyncio
async def test_reflect_truncates_to_last_30_turns(memory_store, monkeypatch):
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", True)
    client = _fake_client_returning('{"lessons":[],"facts":{},"capability_gaps":[]}')
    engine = LearningEngine(memory_store)

    transcript = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
    await engine.reflect(transcript, client=client)

    # Verify the call happened with only the last 30 turns
    call_args = client.messages.create.call_args
    sent = call_args.kwargs["messages"][0]["content"]
    assert "msg99" in sent
    assert "msg70" in sent
    assert "msg0" not in sent


@pytest.mark.asyncio
async def test_reflect_code_fence_without_json_prefix(memory_store, monkeypatch):
    """Branch 52->54: code fence content not starting with 'json' skips raw[4:]."""
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", True)
    # Fence is "```\n{...}" — split gives "\n{...}", which doesn't start with "json"
    payload = '{"lessons":["branch test"],"facts":{},"capability_gaps":[]}'
    fenced = f"```\n{payload}\n```"
    client = _fake_client_returning(fenced)
    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "x"}], client=client)
    assert result.get("lessons") == ["branch test"]


# ── Additional LearningEngine tests ──────────────────────────────────────────

def test_learning_engine_memory_attribute(memory_store):
    engine = LearningEngine(memory_store)
    assert engine.memory is memory_store


@pytest.mark.asyncio
async def test_reflect_capability_gaps_returned_in_result(memory_store, monkeypatch):
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", True)
    client = _fake_client_returning(
        '{"lessons":[],"facts":{},"capability_gaps":["needs PDF tool"]}'
    )
    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "hi"}], client=client)
    assert "capability_gaps" in result
    assert result["capability_gaps"] == ["needs PDF tool"]


@pytest.mark.asyncio
async def test_reflect_malformed_json_returns_empty(memory_store, monkeypatch):
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", True)
    client = _fake_client_returning("not valid json at all")
    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "x"}], client=client)
    assert result == {}


@pytest.mark.asyncio
async def test_reflect_lesson_stored_with_session_reflection_context(memory_store, monkeypatch):
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", True)
    client = _fake_client_returning('{"lessons":["important lesson"],"facts":{},"capability_gaps":[]}')
    engine = LearningEngine(memory_store)
    await engine.reflect([{"role": "user", "content": "test"}], client=client)
    with memory_store._conn() as conn:
        row = conn.execute(
            "SELECT context FROM lessons WHERE lesson=?", ("important lesson",)
        ).fetchone()
    assert row is not None
    assert row["context"] == "session_reflection"


@pytest.mark.asyncio
async def test_reflect_fact_stored_with_source_reflection(memory_store, monkeypatch):
    monkeypatch.setattr("jarvis.memory.learning.cfg.LEARNING_ENABLED", True)
    client = _fake_client_returning('{"lessons":[],"facts":{"pref_theme":"dark"},"capability_gaps":[]}')
    engine = LearningEngine(memory_store)
    await engine.reflect([{"role": "user", "content": "test"}], client=client)
    with memory_store._conn() as conn:
        row = conn.execute(
            "SELECT source, confidence FROM facts WHERE key=?", ("pref_theme",)
        ).fetchone()
    assert row["source"] == "reflection"
    assert row["confidence"] == pytest.approx(0.8)


def test_build_context_prompt_only_facts_excludes_lessons_header(memory_store):
    memory_store.store_fact("editor", "emacs")
    engine = LearningEngine(memory_store)
    prompt = engine.build_context_prompt()
    assert "Known Facts" in prompt
    assert "Lessons Learned" not in prompt


def test_build_context_prompt_only_lessons_excludes_facts_header(memory_store):
    memory_store.store_lesson("always write tests", context="")
    engine = LearningEngine(memory_store)
    prompt = engine.build_context_prompt()
    assert "Lessons Learned" in prompt
    assert "Known Facts" not in prompt
