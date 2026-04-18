"""Tests for jarvis.memory.store — SQLite-backed long-term memory."""

import pytest
from pathlib import Path


def test_store_and_retrieve_fact(memory_store):
    memory_store.store_fact("colour", "blue", source="test")
    facts = memory_store.all_facts()
    keys = [f["key"] for f in facts]
    assert "colour" in keys


def test_fact_overwrite(memory_store):
    memory_store.store_fact("lang", "python")
    memory_store.store_fact("lang", "rust")  # should overwrite
    facts = {f["key"]: f["value"] for f in memory_store.all_facts()}
    assert facts["lang"] == "rust"


def test_store_and_retrieve_lesson(memory_store):
    memory_store.store_lesson("Always validate inputs.", context="test")
    lessons = memory_store.get_lessons(limit=10)
    assert any("validate" in l.lower() for l in lessons)


def test_save_and_get_message(memory_store):
    sid = "session-001"
    memory_store.save_message(sid, "user", "hello")
    memory_store.save_message(sid, "assistant", "hi there")
    history = memory_store.get_history(sid, limit=10)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_history_respects_limit(memory_store):
    sid = "session-002"
    for i in range(20):
        memory_store.save_message(sid, "user", f"message {i}")
    history = memory_store.get_history(sid, limit=5)
    assert len(history) == 5


def test_scheduled_task_crud(memory_store):
    memory_store.add_scheduled_task("daily_report", "0 9 * * 1-5", "Generate daily report.")
    tasks = memory_store.get_scheduled_tasks()
    assert any(t["name"] == "daily_report" for t in tasks)


def test_capability_gap_log_and_resolve(memory_store):
    memory_store.log_gap("Need a web scraper tool", context="test")
    gaps = memory_store.get_open_gaps()
    assert len(gaps) >= 1
    gap_id = gaps[0]["id"]
    memory_store.resolve_gap(gap_id)
    open_gaps = memory_store.get_open_gaps()
    assert all(g["id"] != gap_id for g in open_gaps)


def test_skill_usage_recording(memory_store):
    memory_store.record_skill_use("web_search", "web_search", success=True)
    memory_store.record_skill_use("web_search", "web_search", success=False)
    # No assertion on return value — just verify no exception raised


def test_summary_returns_string(memory_store):
    summary = memory_store.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_monitor_target_crud(memory_store):
    memory_store.add_monitor_target("test_url", "url", "https://example.com", "alert me")
    targets = memory_store.get_monitor_targets()
    assert any(t["name"] == "test_url" for t in targets)
