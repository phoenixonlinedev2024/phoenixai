"""Tests for jarvis.memory.store — SQLite-backed long-term memory."""

import pytest  # noqa: F401
from pathlib import Path  # noqa: F401


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


# ── Additional coverage ──────────────────────────────────────────────────────

def test_recall_fact_roundtrip_string(memory_store):
    memory_store.store_fact("city", "Paris")
    assert memory_store.recall_fact("city") == "Paris"


def test_recall_fact_deserialises_json(memory_store):
    memory_store.store_fact("settings", {"theme": "dark", "compact": True})
    assert memory_store.recall_fact("settings") == {"theme": "dark", "compact": True}


def test_recall_fact_missing_returns_none(memory_store):
    assert memory_store.recall_fact("nope") is None


def test_search_facts_by_key_or_value(memory_store):
    memory_store.store_fact("favourite_colour", "azure")
    memory_store.store_fact("mood", "azure-adjacent")
    memory_store.store_fact("unrelated", "rose")
    hits = memory_store.search_facts("azure")
    keys = [h["key"] for h in hits]
    assert "favourite_colour" in keys
    assert "mood" in keys
    assert "unrelated" not in keys


def test_search_facts_respects_limit(memory_store):
    for i in range(30):
        memory_store.store_fact(f"k_{i}", "match-me")
    hits = memory_store.search_facts("match-me")
    assert len(hits) <= 20


def test_update_scheduled_task_run(memory_store):
    memory_store.add_scheduled_task("nightly", "0 0 * * *", "do stuff")
    memory_store.update_task_run("nightly", "2024-01-01T00:00:00", "2024-01-02T00:00:00")
    tasks = memory_store.get_scheduled_tasks()
    match = next(t for t in tasks if t["name"] == "nightly")
    assert match["last_run"] == "2024-01-01T00:00:00"
    assert match["next_run"] == "2024-01-02T00:00:00"


def test_ab_result_records_and_retrieves_winner(memory_store):
    memory_store.record_ab_result(
        task_hash="h1", variant_a="A", variant_b="B",
        winner="A", score_a=0.9, score_b=0.5,
    )
    assert memory_store.get_ab_winner("h1") == "A"


def test_ab_winner_latest_wins(memory_store):
    memory_store.record_ab_result("h2", "A", "B", "A", 0.9, 0.5)
    memory_store.record_ab_result("h2", "A", "B", "B", 0.4, 0.8)
    assert memory_store.get_ab_winner("h2") == "B"


def test_ab_winner_unknown_task_returns_none(memory_store):
    assert memory_store.get_ab_winner("never-run") is None


def test_update_monitor_hash(memory_store):
    memory_store.add_monitor_target("docs", "file", "/tmp/x.md", "summarise")
    memory_store.update_monitor_hash("docs", "abc123")
    targets = memory_store.get_monitor_targets()
    match = next(t for t in targets if t["name"] == "docs")
    assert match["last_hash"] == "abc123"


def test_summary_reflects_counts(memory_store):
    memory_store.store_fact("a", "1")
    memory_store.store_fact("b", "2")
    memory_store.store_lesson("learn this")
    s = memory_store.summary()
    assert "2 facts" in s
    assert "1 lessons" in s


def test_lessons_ordered_newest_first(memory_store):
    memory_store.store_lesson("first")
    memory_store.store_lesson("second")
    memory_store.store_lesson("third")
    out = memory_store.get_lessons(limit=10)
    assert out[0] == "third"
    assert out[-1] == "first"


def test_get_history_chronological_order(memory_store):
    sid = "chrono"
    memory_store.save_message(sid, "user", "one")
    memory_store.save_message(sid, "assistant", "two")
    memory_store.save_message(sid, "user", "three")
    hist = memory_store.get_history(sid, limit=10)
    assert [m["content"] for m in hist] == ["one", "two", "three"]


def test_skill_usage_increments_and_averages(memory_store):
    memory_store.record_skill_use("s1", "desc", success=True)
    memory_store.record_skill_use("s1", "desc", success=False)
    memory_store.record_skill_use("s1", "desc", success=True)
    with memory_store._conn() as conn:
        row = conn.execute(
            "SELECT usage_count, success_rate FROM skills WHERE name=?", ("s1",)
        ).fetchone()
    assert row["usage_count"] == 3
    assert 0 < row["success_rate"] <= 1


def test_corrupt_db_path_auto_creates_parent(tmp_path):
    from jarvis.memory.store import MemoryStore
    nested = tmp_path / "deep" / "down" / "m.db"
    ms = MemoryStore(db_path=nested)
    ms.store_fact("x", "y")
    assert nested.exists()
    assert ms.recall_fact("x") == "y"


# ── Gap list / resolve / re-list ──────────────────────────────────────────────

def test_gaps_list_and_resolve(memory_store):
    memory_store.log_gap("can't generate images", context="user asked for image")
    memory_store.log_gap("can't parse PDFs", context="PDF upload")
    gaps = memory_store.get_open_gaps()
    assert len(gaps) >= 2
    descriptions = [g["description"] for g in gaps]
    assert any("image" in d for d in descriptions)
    assert any("PDF" in d for d in descriptions)

    # Resolve one gap
    gap_id = next(g["id"] for g in gaps if "image" in g["description"])
    memory_store.resolve_gap(gap_id)

    remaining_gaps = memory_store.get_open_gaps()
    remaining_descriptions = [g["description"] for g in remaining_gaps]
    assert all("image" not in d for d in remaining_descriptions)


def test_search_facts_matches_by_value(memory_store):
    memory_store.store_fact("colour", "blue sky")
    memory_store.store_fact("size", "very large building")
    # Search by value substring
    results = memory_store.search_facts("blue")
    assert any(r["key"] == "colour" for r in results)
    assert all(r["key"] != "size" for r in results)


def test_search_facts_returns_multiple_matches(memory_store):
    memory_store.store_fact("alpha_topic", "test_subject_xyz")
    memory_store.store_fact("beta_topic", "test_subject_xyz")
    results = memory_store.search_facts("test_subject_xyz")
    keys = {r["key"] for r in results}
    assert "alpha_topic" in keys
    assert "beta_topic" in keys


def test_search_facts_empty_when_no_match(memory_store):
    results = memory_store.search_facts("zzz_not_found_anywhere_xyz")
    assert results == []


def test_history_limit_zero_returns_empty(memory_store):
    memory_store.save_message("sid", "user", "hi")
    h = memory_store.get_history("sid", limit=0)
    assert h == []


def test_store_and_recall_json_fact(memory_store):
    memory_store.store_fact("config", {"key": "value", "count": 42})
    val = memory_store.recall_fact("config")
    assert val == {"key": "value", "count": 42}


# ── Fact confidence and source fields ─────────────────────────────────────────

def test_store_fact_custom_confidence_and_source(memory_store):
    memory_store.store_fact("expertise", "Python", source="user", confidence=0.5)
    results = memory_store.search_facts("expertise")
    assert len(results) == 1
    assert abs(results[0]["confidence"] - 0.5) < 1e-6


def test_store_fact_overwrites_preserves_new_confidence(memory_store):
    memory_store.store_fact("skill", "Python", confidence=0.8)
    memory_store.store_fact("skill", "Python+", confidence=0.95)
    results = memory_store.search_facts("skill")
    assert abs(results[0]["confidence"] - 0.95) < 1e-6
    assert results[0]["value"] == "Python+"


def test_all_facts_sorted_by_key(memory_store):
    memory_store.store_fact("zzz_last", "c")
    memory_store.store_fact("aaa_first", "a")
    memory_store.store_fact("mmm_mid", "b")
    keys = [f["key"] for f in memory_store.all_facts()]
    idxs = [keys.index("aaa_first"), keys.index("mmm_mid"), keys.index("zzz_last")]
    assert idxs == sorted(idxs)


def test_search_facts_confidence_in_results(memory_store):
    memory_store.store_fact("lang", "Rust", confidence=0.75)
    results = memory_store.search_facts("lang")
    assert "confidence" in results[0]
    assert abs(results[0]["confidence"] - 0.75) < 1e-6


# ── Conversation history ordering ─────────────────────────────────────────────

def test_get_history_multiple_sessions_isolated(memory_store):
    memory_store.save_message("sess_a", "user", "hello from A")
    memory_store.save_message("sess_b", "user", "hello from B")
    hist_a = memory_store.get_history("sess_a")
    hist_b = memory_store.get_history("sess_b")
    assert all("A" in m["content"] for m in hist_a)
    assert all("B" in m["content"] for m in hist_b)


def test_get_history_respects_limit_from_newest(memory_store):
    for i in range(10):
        memory_store.save_message("sid_lim", "user", f"msg {i}")
    hist = memory_store.get_history("sid_lim", limit=3)
    assert len(hist) == 3
    # Should be last 3 messages in chronological order
    assert hist[-1]["content"] == "msg 9"


# ── Lessons ordering and limit ────────────────────────────────────────────────

def test_get_lessons_newest_first(memory_store):
    memory_store.store_lesson("first lesson", context="session1")
    memory_store.store_lesson("second lesson", context="session2")
    memory_store.store_lesson("third lesson", context="session3")
    lessons = memory_store.get_lessons(limit=10)
    assert "third lesson" in lessons[0]
    assert "first lesson" in lessons[-1]


def test_get_lessons_respects_limit(memory_store):
    for i in range(10):
        memory_store.store_lesson(f"lesson {i}")
    lessons = memory_store.get_lessons(limit=3)
    assert len(lessons) == 3


# ── Monitor target CRUD ───────────────────────────────────────────────────────

def test_monitor_target_enabled_by_default(memory_store):
    memory_store.add_monitor_target("My Check", "url", "https://example.com", "notify")
    targets = memory_store.get_monitor_targets()
    t = next(t for t in targets if t["name"] == "My Check")
    assert t["enabled"] == 1


def test_monitor_target_list_returns_all(memory_store):
    memory_store.add_monitor_target("T1", "file", "/path/a", "log")
    memory_store.add_monitor_target("T2", "url", "http://b.com", "alert")
    targets = memory_store.get_monitor_targets()
    names = {t["name"] for t in targets}
    assert {"T1", "T2"} <= names


# ── SQL injection safety ──────────────────────────────────────────────────────

def test_search_facts_sql_injection_safe(memory_store):
    memory_store.store_fact("safe_key", "safe_value")
    results = memory_store.search_facts("' OR '1'='1")
    assert results == []


def test_store_fact_with_single_quotes_in_value(memory_store):
    memory_store.store_fact("greeting", "it's a beautiful day")
    val = memory_store.recall_fact("greeting")
    assert val == "it's a beautiful day"
