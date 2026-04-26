"""Tests for jarvis.planner — TaskPlanner: decompose, score, group, A/B judge."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Inject fake heavy deps before jarvis.planner is imported ─────────────────

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

from jarvis.planner import TaskPlanner  # noqa: E402


def _fake_client_returning(text: str) -> MagicMock:
    """Build a client whose messages.create returns content[0].text = text."""
    client = MagicMock()
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    client.messages.create = AsyncMock(return_value=resp)
    return client


def _fake_memory() -> MagicMock:
    mem = MagicMock()
    mem.get_ab_winner = MagicMock(return_value=None)
    mem.record_ab_result = MagicMock()
    return mem


# ── group_subtasks (pure logic) ──────────────────────────────────────────────

def test_group_subtasks_single_group():
    planner = TaskPlanner(client=MagicMock(), memory=MagicMock())
    groups = planner.group_subtasks([
        {"id": 1, "group": 1, "description": "a"},
        {"id": 2, "group": 1, "description": "b"},
    ])
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_group_subtasks_multiple_groups_sorted():
    planner = TaskPlanner(client=MagicMock(), memory=MagicMock())
    groups = planner.group_subtasks([
        {"id": 1, "group": 3, "description": "c"},
        {"id": 2, "group": 1, "description": "a"},
        {"id": 3, "group": 2, "description": "b"},
    ])
    assert [g[0]["description"] for g in groups] == ["a", "b", "c"]


def test_group_subtasks_missing_group_defaults_to_1():
    planner = TaskPlanner(client=MagicMock(), memory=MagicMock())
    groups = planner.group_subtasks([{"id": 1, "description": "a"}])
    assert groups[0][0]["description"] == "a"


def test_group_subtasks_empty_list():
    planner = TaskPlanner(client=MagicMock(), memory=MagicMock())
    assert planner.group_subtasks([]) == []


# ── decompose ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decompose_returns_subtasks():
    client = _fake_client_returning(
        '{"subtasks":[{"id":1,"group":1,"description":"x"}],"requires_decomposition":true}'
    )
    planner = TaskPlanner(client=client, memory=MagicMock())
    out = await planner.decompose("do x")
    assert out == [{"id": 1, "group": 1, "description": "x"}]


@pytest.mark.asyncio
async def test_decompose_simple_task_returns_single_subtask():
    client = _fake_client_returning(
        '{"subtasks":[{"id":1,"group":1,"description":"x"}],"requires_decomposition":false}'
    )
    planner = TaskPlanner(client=client, memory=MagicMock())
    out = await planner.decompose("simple task")
    assert len(out) == 1
    assert out[0]["description"] == "simple task"


@pytest.mark.asyncio
async def test_decompose_strips_code_fence():
    fenced = '```json\n{"subtasks":[{"id":1,"group":1,"description":"y"}],"requires_decomposition":true}\n```'
    client = _fake_client_returning(fenced)
    planner = TaskPlanner(client=client, memory=MagicMock())
    out = await planner.decompose("do y")
    assert out[0]["description"] == "y"


@pytest.mark.asyncio
async def test_decompose_on_api_error_falls_back_to_single_task():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
    planner = TaskPlanner(client=client, memory=MagicMock())
    out = await planner.decompose("failing task")
    assert out == [{"id": 1, "group": 1, "description": "failing task"}]


# ── score_confidence ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_score_confidence_returns_float():
    client = _fake_client_returning('{"score": 0.92, "reason": "good"}')
    planner = TaskPlanner(client=client, memory=MagicMock())
    score = await planner.score_confidence("t", "r")
    assert score == 0.92


@pytest.mark.asyncio
async def test_score_confidence_fallback_on_error():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("nope"))
    planner = TaskPlanner(client=client, memory=MagicMock())
    score = await planner.score_confidence("t", "r")
    assert score == 0.7


# ── ab_test ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ab_test_returns_winner():
    client = _fake_client_returning(
        '{"winner":"B","score_a":0.4,"score_b":0.8,"reason":"B is more complete"}'
    )
    mem = _fake_memory()
    planner = TaskPlanner(client=client, memory=mem)
    winner, result = await planner.ab_test("task", "ans A", "ans B")
    assert winner == "B"
    assert result["score_b"] == 0.8
    mem.record_ab_result.assert_called_once()


@pytest.mark.asyncio
async def test_ab_test_cached_result_short_circuits():
    mem = _fake_memory()
    mem.get_ab_winner = MagicMock(return_value="A")
    client = MagicMock()
    client.messages.create = AsyncMock()
    planner = TaskPlanner(client=client, memory=mem)
    winner, _ = await planner.ab_test("task", "a", "b")
    assert winner == "A"
    client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_ab_test_error_defaults_to_a():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("x"))
    planner = TaskPlanner(client=client, memory=_fake_memory())
    winner, result = await planner.ab_test("task", "a", "b")
    assert winner == "A"
    assert result == {}


@pytest.mark.asyncio
async def test_score_confidence_strips_code_fence():
    """Line 79: backtick-fenced response is unwrapped in score_confidence."""
    fenced = '```json\n{"score": 0.85, "reason": "good"}\n```'
    client = _fake_client_returning(fenced)
    planner = TaskPlanner(client=client, memory=MagicMock())
    score = await planner.score_confidence("task", "response")
    assert score == 0.85


@pytest.mark.asyncio
async def test_ab_test_strips_code_fence():
    """Line 102: backtick-fenced response is unwrapped in ab_test."""
    payload = '{"winner":"A","score_a":0.9,"score_b":0.5,"reason":"A is better"}'
    fenced = f"```json\n{payload}\n```"
    client = _fake_client_returning(fenced)
    mem = _fake_memory()
    planner = TaskPlanner(client=client, memory=mem)
    winner, result = await planner.ab_test("task", "ans A", "ans B")
    assert winner == "A"
    assert result["score_a"] == 0.9


@pytest.mark.asyncio
async def test_decompose_code_fence_without_json_prefix():
    """Branch 59->61: code fence content not starting with 'json' skips raw[4:]."""
    payload = '{"subtasks":[{"id":1,"group":1,"description":"branch task"}],"requires_decomposition":true}'
    fenced = f"```\n{payload}\n```"
    client = _fake_client_returning(fenced)
    planner = TaskPlanner(client=client, memory=MagicMock())
    out = await planner.decompose("do task")
    assert len(out) == 1
    assert out[0]["description"] == "branch task"
