"""Tests for jarvis.memory.learning — post-session reflection and context building."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.memory.learning import LearningEngine


# ── reflect() ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflect_disabled_returns_empty(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", False)
    engine = LearningEngine(memory_store)
    client = MagicMock()
    out = await engine.reflect([{"role": "user", "content": "hi"}], client)
    assert out == {}


@pytest.mark.asyncio
async def test_reflect_parses_json_and_persists(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(text='{"lessons": ["always validate"], '
                                    '"facts": {"fav_lang": "Python"}, '
                                    '"capability_gaps": []}')]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    out = await engine.reflect(
        [{"role": "user", "content": "teach me"},
         {"role": "assistant", "content": "done"}],
        client,
    )
    assert "lessons" in out
    # Lesson was persisted
    assert any("validate" in lesson for lesson in memory_store.get_lessons())
    # Fact was persisted
    assert memory_store.recall_fact("fav_lang") == "Python"


@pytest.mark.asyncio
async def test_reflect_strips_code_fences(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(text='```json\n{"lessons": ["use fences"], '
                                    '"facts": {}, "capability_gaps": []}\n```')]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    out = await engine.reflect([{"role": "user", "content": "x"}], client)
    assert "lessons" in out


@pytest.mark.asyncio
async def test_reflect_handles_api_error(memory_store, monkeypatch, capsys):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))

    engine = LearningEngine(memory_store)
    out = await engine.reflect([{"role": "user", "content": "x"}], client)
    assert out == {}
    assert "Reflection error" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_reflect_handles_malformed_json(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(text="not json at all")]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    out = await engine.reflect([{"role": "user", "content": "x"}], client)
    assert out == {}


@pytest.mark.asyncio
async def test_reflect_truncates_transcript(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(text='{"lessons": [], "facts": {}, "capability_gaps": []}')]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    # 50 turns, each with long content
    transcript = [{"role": "user", "content": "x" * 2000} for _ in range(50)]
    await engine.reflect(transcript, client)

    kwargs = client.messages.create.call_args.kwargs
    user_content = kwargs["messages"][0]["content"]
    # Only last 30 turns, each clipped to 500 chars (+ role prefix/newline)
    assert user_content.count("USER:") == 30
    # No turn should contain a full 2000-char run
    assert "x" * 1000 not in user_content


# ── build_context_prompt() ───────────────────────────────────────────────────

def test_build_context_prompt_empty(memory_store):
    engine = LearningEngine(memory_store)
    assert engine.build_context_prompt() == ""


def test_build_context_prompt_includes_facts(memory_store):
    memory_store.store_fact("lang", "Python")
    memory_store.store_fact("editor", "nvim")
    engine = LearningEngine(memory_store)
    out = engine.build_context_prompt()
    assert "Known Facts" in out
    assert "Python" in out
    assert "nvim" in out


def test_build_context_prompt_includes_lessons(memory_store):
    memory_store.store_lesson("always validate inputs")
    engine = LearningEngine(memory_store)
    out = engine.build_context_prompt()
    assert "Lessons Learned" in out
    assert "validate" in out


def test_build_context_prompt_caps_facts_at_20(memory_store):
    for i in range(30):
        memory_store.store_fact(f"key_{i:02d}", f"value_{i}")
    engine = LearningEngine(memory_store)
    out = engine.build_context_prompt()
    # Count the fact bullets
    fact_lines = [line for line in out.splitlines() if line.startswith("  - key_")]
    assert len(fact_lines) == 20


# ── reflect() stores lessons and facts ───────────────────────────────────────

@pytest.mark.asyncio
async def test_reflect_stores_lessons_and_facts(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(text='{"lessons": ["validate inputs", "cache results"], "facts": {"language": "Python", "os": "Linux"}, "capability_gaps": []}')]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "hi"}], client)

    assert result["lessons"] == ["validate inputs", "cache results"]
    # Facts should be stored in memory
    assert memory_store.recall_fact("language") == "Python"
    assert memory_store.recall_fact("os") == "Linux"
    # Lessons should be stored
    lessons = memory_store.get_lessons(limit=10)
    assert any("validate inputs" in l for l in lessons)


@pytest.mark.asyncio
async def test_reflect_strips_code_fence(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    payload = '{"lessons": ["lesson from fence"], "facts": {}, "capability_gaps": []}'
    fenced = f"```json\n{payload}\n```"
    reply = MagicMock()
    reply.content = [MagicMock(text=fenced)]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "test"}], client)
    assert result["lessons"] == ["lesson from fence"]


@pytest.mark.asyncio
async def test_reflect_capability_gaps_ignored_no_crash(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(text='{"lessons": [], "facts": {}, "capability_gaps": ["need PDF parsing"]}')]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "test"}], client)
    assert result["capability_gaps"] == ["need PDF parsing"]


# ── build_context_prompt includes lessons cap ────────────────────────────────

def test_build_context_prompt_caps_lessons(memory_store):
    for i in range(20):
        memory_store.store_lesson(f"lesson number {i}")
    engine = LearningEngine(memory_store)
    out = engine.build_context_prompt()
    lesson_lines = [line for line in out.splitlines() if line.startswith("  - lesson")]
    assert len(lesson_lines) == 10  # get_lessons(limit=10)


def test_build_context_prompt_only_facts(memory_store):
    memory_store.store_fact("theme", "dark mode")
    engine = LearningEngine(memory_store)
    out = engine.build_context_prompt()
    assert "Known Facts" in out
    assert "Lessons Learned" not in out


def test_build_context_prompt_only_lessons(memory_store):
    memory_store.store_lesson("prefer short responses")
    engine = LearningEngine(memory_store)
    out = engine.build_context_prompt()
    assert "Lessons Learned" in out
    assert "Known Facts" not in out


# ── LearningEngine attribute / additional reflect branches ────────────────────

def test_learning_engine_memory_attribute(memory_store):
    engine = LearningEngine(memory_store)
    assert engine.memory is memory_store


@pytest.mark.asyncio
async def test_reflect_empty_transcript_still_calls_api(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(text='{"lessons": [], "facts": {}, "capability_gaps": []}')]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    result = await engine.reflect([], client)
    client.messages.create.assert_called_once()
    assert result == {"lessons": [], "facts": {}, "capability_gaps": []}


@pytest.mark.asyncio
async def test_reflect_multiple_facts_all_stored(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(
        text='{"lessons": [], "facts": {"k1": "v1", "k2": "v2", "k3": "v3"}, "capability_gaps": []}'
    )]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    await engine.reflect([{"role": "user", "content": "hi"}], client)

    assert memory_store.recall_fact("k1") == "v1"
    assert memory_store.recall_fact("k2") == "v2"
    assert memory_store.recall_fact("k3") == "v3"


@pytest.mark.asyncio
async def test_reflect_multiple_lessons_all_stored(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(
        text='{"lessons": ["lesson A", "lesson B", "lesson C"], "facts": {}, "capability_gaps": []}'
    )]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    await engine.reflect([{"role": "user", "content": "test"}], client)

    lessons = memory_store.get_lessons(limit=10)
    assert any("lesson A" in l for l in lessons)
    assert any("lesson B" in l for l in lessons)
    assert any("lesson C" in l for l in lessons)


def test_build_context_prompt_both_sections_present(memory_store):
    memory_store.store_fact("color", "blue")
    memory_store.store_lesson("be concise")
    engine = LearningEngine(memory_store)
    out = engine.build_context_prompt()
    assert "Known Facts" in out
    assert "Lessons Learned" in out
    assert "blue" in out
    assert "concise" in out


# ── build_context_prompt: separator between sections ─────────────────────────

def test_build_context_prompt_sections_are_separated(memory_store):
    memory_store.store_fact("key", "value")
    memory_store.store_lesson("a lesson")
    engine = LearningEngine(memory_store)
    out = engine.build_context_prompt()
    # Verify both sections present and output is non-trivial
    assert len(out) > 30


def test_build_context_prompt_fact_key_value_format(memory_store):
    memory_store.store_fact("user_name", "Alice")
    engine = LearningEngine(memory_store)
    out = engine.build_context_prompt()
    assert "user_name" in out
    assert "Alice" in out


@pytest.mark.asyncio
async def test_reflect_returns_result_dict_on_success(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    reply = MagicMock()
    reply.content = [MagicMock(text='{"lessons": ["x"], "facts": {}, "capability_gaps": []}')]
    client.messages.create = AsyncMock(return_value=reply)

    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "q"}], client)
    assert isinstance(result, dict)
    assert "lessons" in result


@pytest.mark.asyncio
async def test_reflect_api_error_returns_empty_dict(memory_store, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LEARNING_ENABLED", True)

    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=ConnectionError("network down"))

    engine = LearningEngine(memory_store)
    result = await engine.reflect([{"role": "user", "content": "hi"}], client)
    assert result == {}
