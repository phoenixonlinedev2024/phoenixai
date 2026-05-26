"""Tests for jarvis.research — Trajectory, TrajectoryCollector, ShareGPT export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.research.trajectory import Trajectory, TrajectoryCollector, Turn
from jarvis.research.sharegpt import (
    compress_trajectory,
    export_atropos_format,
    export_dataset,
    to_sharegpt,
)


# ── Turn / Trajectory ─────────────────────────────────────────────────────────

def test_turn_defaults():
    t = Turn(role="user", content="hello")
    assert t.tool_calls == []
    assert t.tool_results == []
    assert t.reward is None


def test_trajectory_defaults():
    tr = Trajectory()
    assert tr.outcome == "unknown"
    assert tr.reward == 0.0
    assert tr.turns == []
    assert len(tr.id) == 32  # uuid hex


def test_trajectory_add_turn():
    tr = Trajectory()
    tr.add_turn("user", "hi")
    tr.add_turn("assistant", "hello sir")
    assert len(tr.turns) == 2
    assert tr.turns[0].content == "hi"


def test_trajectory_set_outcome_and_reward():
    tr = Trajectory()
    tr.add_turn("user", "do x")
    tr.add_turn("assistant", "done")
    tr.set_outcome("success", reward=0.9)
    assert tr.outcome == "success"
    assert tr.reward == 0.9


def test_trajectory_set_outcome_backfills_last_assistant():
    tr = Trajectory()
    tr.add_turn("user", "q")
    tr.add_turn("assistant", "a1")
    tr.add_turn("user", "q2")
    tr.add_turn("assistant", "a2")
    tr.set_outcome("success", reward=1.0)
    assert tr.turns[3].reward == 1.0
    assert tr.turns[1].reward is None  # earlier assistant not backfilled


def test_trajectory_to_dict_shape():
    tr = Trajectory(session_id="s1", task="test task")
    tr.add_turn("user", "msg")
    d = tr.to_dict()
    assert d["session_id"] == "s1"
    assert d["task"] == "test task"
    assert len(d["turns"]) == 1
    assert d["turns"][0]["role"] == "user"


# ── TrajectoryCollector ───────────────────────────────────────────────────────

@pytest.fixture()
def collector(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    return TrajectoryCollector()


def test_collector_start_creates_trajectory(collector):
    traj = collector.start("s1", task="build something")
    assert traj.session_id == "s1"
    assert traj.task == "build something"


def test_collector_get_returns_active(collector):
    collector.start("s1")
    assert collector.get("s1") is not None


def test_collector_get_unknown_returns_none(collector):
    assert collector.get("unknown") is None


def test_collector_record_turn(collector):
    collector.start("s1")
    collector.record_turn("s1", "user", "hello")
    traj = collector.get("s1")
    assert len(traj.turns) == 1


def test_collector_complete_saves_file(collector, tmp_path):
    collector.start("s1", task="t")
    collector.record_turn("s1", "user", "q")
    collector.record_turn("s1", "assistant", "a")
    msg = collector.complete("s1", outcome="success", reward=1.0)
    assert "saved" in msg
    files = list((tmp_path / "trajectories").glob("*.json"))
    assert len(files) == 1


def test_collector_complete_removes_from_active(collector):
    collector.start("s1")
    collector.complete("s1")
    assert collector.get("s1") is None


def test_collector_complete_unknown_session(collector):
    msg = collector.complete("ghost")
    assert "No active" in msg


def test_collector_load_all_returns_saved(collector):
    collector.start("s1", task="t")
    collector.record_turn("s1", "user", "hello")
    collector.complete("s1", reward=0.8)
    all_traj = collector.load_all()
    assert len(all_traj) == 1
    assert all_traj[0].reward == 0.8


def test_collector_stats(collector):
    collector.start("s1", task="t")
    collector.complete("s1", outcome="success", reward=1.0)
    collector.start("s2", task="t2")
    collector.complete("s2", outcome="failure", reward=0.0)
    stats = collector.stats()
    assert stats["total"] == 2
    assert stats["success"] == 1
    assert stats["failure"] == 1
    assert 0.0 <= stats["avg_reward"] <= 1.0


def test_collector_record_turn_unknown_session_is_noop(collector):
    collector.record_turn("nonexistent", "user", "content")
    # no active trajectory for "nonexistent" — should not raise


def test_collector_stats_avg_turns(collector):
    collector.start("s1")
    collector.record_turn("s1", "user", "a")
    collector.record_turn("s1", "assistant", "b")
    collector.complete("s1", outcome="success")
    stats = collector.stats()
    assert stats["avg_turns"] == 2.0


def test_collector_stats_empty(collector):
    stats = collector.stats()
    assert stats["total"] == 0
    assert stats["avg_reward"] == 0.0
    assert stats["avg_turns"] == 0.0


def test_load_all_skips_corrupt_file(collector, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    # Create a corrupt JSON file in the trajectories directory
    coll = TrajectoryCollector()
    traj_dir = coll._dir
    traj_dir.mkdir(parents=True, exist_ok=True)
    (traj_dir / "corrupt.json").write_text("{not valid json{{")
    loaded = coll.load_all()
    assert loaded == []  # corrupt file is silently skipped


# ── to_sharegpt ───────────────────────────────────────────────────────────────

def test_to_sharegpt_basic():
    tr = Trajectory(task="test")
    tr.add_turn("user", "question")
    tr.add_turn("assistant", "answer")
    out = to_sharegpt(tr)
    assert out["conversations"][0] == {"from": "human", "value": "question"}
    assert out["conversations"][1] == {"from": "gpt", "value": "answer"}


def test_to_sharegpt_with_tool_calls():
    tr = Trajectory()
    turn = Turn(
        role="assistant",
        content="searching…",
        tool_calls=[{"name": "web_search", "input": {"query": "test"}}],
        tool_results=[{"content": "result"}],
    )
    tr.turns = [turn]
    out = to_sharegpt(tr)
    gpt_val = out["conversations"][0]["value"]
    assert "web_search" in gpt_val
    assert "searching" in gpt_val


def test_to_sharegpt_includes_metadata():
    tr = Trajectory(task="my task", outcome="success", reward=0.9)
    out = to_sharegpt(tr)
    assert out["task"] == "my task"
    assert out["outcome"] == "success"
    assert out["reward"] == 0.9


# ── compress_trajectory ───────────────────────────────────────────────────────

def test_compress_short_trajectory_unchanged():
    tr = Trajectory()
    for i in range(5):
        tr.add_turn("user", f"m{i}")
    compressed = compress_trajectory(tr, max_turns=20)
    assert compressed is tr


def test_compress_long_trajectory_truncates():
    tr = Trajectory()
    for i in range(50):
        tr.add_turn("user" if i % 2 == 0 else "assistant", f"m{i}")
    compressed = compress_trajectory(tr, max_turns=10)
    assert len(compressed.turns) == 10
    assert compressed.metadata.get("compressed") is True
    assert compressed.metadata["original_turns"] == 50


def test_compress_keeps_first_and_last_turns():
    tr = Trajectory()
    for i in range(30):
        tr.add_turn("user", f"m{i}")
    compressed = compress_trajectory(tr, max_turns=5)
    assert compressed.turns[0].content == "m0"
    assert compressed.turns[-1].content == "m29"


# ── export_dataset ────────────────────────────────────────────────────────────

def test_export_dataset_creates_jsonl(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    coll = TrajectoryCollector()

    coll.start("s1", task="t")
    coll.record_turn("s1", "user", "q")
    coll.record_turn("s1", "assistant", "a")
    coll.complete("s1", outcome="success", reward=1.0)

    out_path = str(tmp_path / "dataset.jsonl")
    msg = export_dataset(coll, output_path=out_path, min_reward=0.5, compress=False)
    assert "Exported 1" in msg
    lines = Path(out_path).read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "conversations" in record


def test_export_dataset_filters_low_reward(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    coll = TrajectoryCollector()

    coll.start("s1", task="t")
    coll.complete("s1", outcome="success", reward=0.2)  # below threshold

    out_path = str(tmp_path / "dataset.jsonl")
    export_dataset(coll, output_path=out_path, min_reward=0.5)
    lines = [l for l in Path(out_path).read_text().strip().split("\n") if l]
    assert len(lines) == 0


def test_export_atropos_format(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    coll = TrajectoryCollector()

    coll.start("s1", task="t")
    coll.record_turn("s1", "user", "prompt")
    coll.record_turn("s1", "assistant", "response")
    coll.complete("s1", reward=0.7)

    out_path = str(tmp_path / "atropos.jsonl")
    msg = export_atropos_format(coll, output_path=out_path)
    assert "Exported 1" in msg
    lines = Path(out_path).read_text().strip().split("\n")
    record = json.loads(lines[0])
    assert "messages" in record
    assert "final_reward" in record


def test_to_sharegpt_skips_non_user_non_assistant_turn():
    """Branch 21->18: turn with role 'system' is skipped (neither user nor assistant)."""
    from jarvis.research.trajectory import Trajectory, Turn
    tr = Trajectory(task="test skipping")
    tr.turns.append(Turn(role="user", content="hello"))
    tr.turns.append(Turn(role="system", content="system context"))
    tr.turns.append(Turn(role="assistant", content="reply"))

    out = to_sharegpt(tr)
    convs = out["conversations"]
    assert len(convs) == 2
    assert convs[0]["from"] == "human"
    assert convs[1]["from"] == "gpt"


# ── Tool-call formatting in to_sharegpt ──────────────────────────────────────

def test_to_sharegpt_with_tool_calls_and_results():
    tr = Trajectory(task="use tool")
    tr.add_turn("user", "search for python")
    tr.add_turn("assistant", "I'll search.",
                tool_calls=[{"name": "web_search", "input": {"q": "python"}}],
                tool_results=[{"content": "10 results"}])
    out = to_sharegpt(tr)
    gpt_val = out["conversations"][1]["value"]
    assert "web_search" in gpt_val
    assert "python" in gpt_val
    assert "10 results" in gpt_val


def test_to_sharegpt_tool_calls_without_content():
    """Assistant turn with tool_calls but empty text."""
    tr = Trajectory(task="bare tool")
    tr.add_turn("user", "do stuff")
    tr.add_turn("assistant", "",
                tool_calls=[{"name": "run_shell", "input": {"cmd": "ls"}}],
                tool_results=[{"content": "file.txt"}])
    out = to_sharegpt(tr)
    gpt_val = out["conversations"][1]["value"]
    assert "run_shell" in gpt_val
    assert "file.txt" in gpt_val


# ── Trajectory.set_outcome backfills only last assistant turn ─────────────────

def test_set_outcome_backfills_last_assistant_only():
    tr = Trajectory()
    tr.add_turn("user", "first msg")
    tr.add_turn("assistant", "first reply")
    tr.add_turn("user", "second msg")
    tr.add_turn("assistant", "second reply")
    tr.set_outcome("success", reward=0.9)
    rewards = [t.reward for t in tr.turns if t.role == "assistant"]
    assert rewards[-1] == pytest.approx(0.9)
    assert rewards[0] is None  # only the last assistant turn gets backfilled


# ── TrajectoryCollector.stats() with mixed outcomes ───────────────────────────

def test_collector_stats_mixed_outcomes(collector):
    s1 = collector.start("s1", task="task a")
    s1.add_turn("user", "go")
    s1.add_turn("assistant", "done")
    collector.complete("s1", outcome="success", reward=1.0)

    s2 = collector.start("s2", task="task b")
    s2.add_turn("user", "go")
    collector.complete("s2", outcome="failure", reward=0.0)

    s3 = collector.start("s3", task="task c")
    s3.add_turn("user", "go")
    s3.add_turn("assistant", "partial")
    s3.add_turn("user", "more")
    s3.add_turn("assistant", "result")
    collector.complete("s3", outcome="success", reward=0.5)

    stats = collector.stats()
    assert stats["total"] == 3
    assert stats["success"] == 2
    assert stats["failure"] == 1
    assert abs(stats["avg_reward"] - (1.0 + 0.0 + 0.5) / 3) < 1e-6


# ── Compress trajectory boundary: exactly max_turns needs no compression ─────

def test_compress_trajectory_at_exact_max_turns_unchanged():
    tr = Trajectory(task="precise")
    for i in range(5):
        tr.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
    result = compress_trajectory(tr, max_turns=5)
    assert result is tr  # returned unchanged


def test_compress_trajectory_tool_calls_preserved_in_kept_turns():
    tr = Trajectory(task="tools")
    for i in range(25):
        tr.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
    # Give tool_calls to last assistant turn (will be kept)
    tr.turns[-1].tool_calls = [{"name": "web_search", "input": {}}]
    compressed = compress_trajectory(tr, max_turns=10)
    # Last turn kept, tool_calls preserved
    assert compressed.turns[-1].tool_calls[0]["name"] == "web_search"
    assert compressed.metadata.get("compressed") is True


# ── TrajectoryCollector edge cases ────────────────────────────────────────────

def test_collector_complete_no_active_session(collector):
    """complete() returns error message when session not in _active."""
    msg = collector.complete("nonexistent-session")
    assert "No active trajectory" in msg


def test_collector_get_returns_none_for_unknown(collector):
    assert collector.get("no-such-session") is None


def test_collector_record_turn_ignored_when_no_session(collector):
    """record_turn() is a no-op when the session hasn't been started."""
    collector.record_turn("ghost-session", "user", "hello")
    # No exception, no trajectory created
    assert collector.get("ghost-session") is None


def test_collector_load_all_skips_corrupt_json(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    coll = TrajectoryCollector()
    # Write corrupt JSON to the trajectories dir
    (tmp_path / "trajectories" / "bad.json").write_text("NOT JSON", encoding="utf-8")
    loaded = coll.load_all()
    assert loaded == []


def test_collector_stats_empty_dir(collector):
    stats = collector.stats()
    assert stats["total"] == 0
    assert stats["success"] == 0
    assert stats["failure"] == 0
    assert stats["avg_reward"] == 0.0
    assert stats["avg_turns"] == 0.0


def test_collector_start_returns_trajectory(collector):
    traj = collector.start("s99", task="my task")
    assert traj.session_id == "s99"
    assert traj.task == "my task"
    assert collector.get("s99") is traj


def test_collector_complete_saves_file(collector, tmp_path):
    collector.start("save-test", task="save it")
    collector.record_turn("save-test", "user", "hello")
    msg = collector.complete("save-test", outcome="success", reward=1.0)
    assert "saved" in msg.lower()
    files = list((tmp_path / "trajectories").glob("*.json"))
    assert len(files) == 1


def test_collector_load_all_restores_turns(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    coll = TrajectoryCollector()
    coll.start("restore-test", task="roundtrip")
    coll.record_turn("restore-test", "user", "ping")
    coll.record_turn("restore-test", "assistant", "pong")
    coll.complete("restore-test", reward=0.8)

    coll2 = TrajectoryCollector()
    all_traj = coll2.load_all()
    assert len(all_traj) == 1
    assert len(all_traj[0].turns) == 2
    assert all_traj[0].turns[0].content == "ping"


# ── export_dataset with compression ──────────────────────────────────────────

def test_export_dataset_with_compression(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    coll = TrajectoryCollector()
    coll.start("long-session", task="compress me")
    for i in range(30):
        coll.record_turn("long-session", "user" if i % 2 == 0 else "assistant", f"turn {i}")
    coll.complete("long-session", outcome="success", reward=1.0)

    out_path = str(tmp_path / "compressed.jsonl")
    msg = export_dataset(coll, output_path=out_path, min_reward=0.5, compress=True, max_turns=10)
    assert "Exported 1" in msg
    record = json.loads(Path(out_path).read_text().strip())
    # Compressed: 10 turns = 10 conversations
    assert len(record["conversations"]) <= 10


def test_export_atropos_rewards_in_messages(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    coll = TrajectoryCollector()
    coll.start("atropos-test")
    coll.record_turn("atropos-test", "user", "prompt")
    coll.record_turn("atropos-test", "assistant", "response")
    coll.complete("atropos-test", reward=0.7)

    out_path = str(tmp_path / "atropos2.jsonl")
    export_atropos_format(coll, output_path=out_path)
    record = json.loads(Path(out_path).read_text().strip())
    assert record["final_reward"] == pytest.approx(0.7)
    assert len(record["messages"]) == 2
    assert record["messages"][0]["role"] == "user"
    assert record["messages"][1]["reward"] == pytest.approx(0.7)


# ── Trajectory.to_dict includes tool calls and results ───────────────────────

def test_trajectory_to_dict_includes_tool_calls():
    tr = Trajectory(task="tool task")
    tr.add_turn("user", "use tools")
    tr.add_turn("assistant", "sure",
                tool_calls=[{"name": "search", "input": {"q": "test"}}],
                tool_results=[{"content": "result data"}])
    d = tr.to_dict()
    turn_d = d["turns"][1]
    assert turn_d["tool_calls"][0]["name"] == "search"
    assert turn_d["tool_results"][0]["content"] == "result data"


def test_trajectory_to_dict_timestamp_is_iso():
    tr = Trajectory()
    d = tr.to_dict()
    assert "T" in d["timestamp"]  # ISO 8601 format
    assert "Z" in d["timestamp"] or "+" in d["timestamp"]  # timezone


# ── to_sharegpt with multiple tool calls ─────────────────────────────────────

def test_to_sharegpt_multiple_tool_calls_in_one_turn():
    tr = Trajectory(task="multi tools")
    tr.add_turn("user", "do two things")
    tr.add_turn("assistant", "calling both",
                tool_calls=[
                    {"name": "tool_a", "input": {"x": 1}},
                    {"name": "tool_b", "input": {"y": 2}},
                ],
                tool_results=[
                    {"content": "result A"},
                    {"content": "result B"},
                ])
    out = to_sharegpt(tr)
    val = out["conversations"][1]["value"]
    assert "tool_a" in val
    assert "tool_b" in val
    assert "result A" in val
    assert "result B" in val


def test_to_sharegpt_user_only_trajectory():
    """Trajectory with only user turns produces only human entries."""
    tr = Trajectory(task="questions only")
    tr.add_turn("user", "question 1")
    tr.add_turn("user", "question 2")
    out = to_sharegpt(tr)
    assert all(c["from"] == "human" for c in out["conversations"])
    assert len(out["conversations"]) == 2


# ── export_dataset with empty collector produces empty file ──────────────────

def test_export_dataset_empty_collector(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    (tmp_path / "trajectories").mkdir(parents=True, exist_ok=True)
    from jarvis.research.trajectory import TrajectoryCollector
    from jarvis.research.sharegpt import export_dataset
    coll = TrajectoryCollector()
    out_path = str(tmp_path / "empty.jsonl")
    msg = export_dataset(coll, output_path=out_path)
    assert "0 trajectories" in msg or "Exported 0" in msg
    assert Path(out_path).exists()


# ── export_atropos_format with empty collector ────────────────────────────────

def test_export_atropos_empty_collector(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    (tmp_path / "trajectories").mkdir(parents=True, exist_ok=True)
    from jarvis.research.trajectory import TrajectoryCollector
    from jarvis.research.sharegpt import export_atropos_format
    coll = TrajectoryCollector()
    out_path = str(tmp_path / "empty_atropos.jsonl")
    msg = export_atropos_format(coll, output_path=out_path)
    assert "0 trajectories" in msg or "Exported 0" in msg
    assert Path(out_path).exists()


# ── compress_trajectory sets "compressed": True in metadata ──────────────────

def test_compress_trajectory_sets_compressed_metadata():
    tr = Trajectory(task="long task")
    for i in range(25):
        tr.add_turn("user" if i % 2 == 0 else "assistant", f"turn {i}")
    compressed = compress_trajectory(tr, max_turns=10)
    assert compressed.metadata.get("compressed") is True
    assert compressed.metadata.get("original_turns") == 25


# ── compress_trajectory preserves first + last turns ─────────────────────────

def test_compress_trajectory_preserves_first_turn_content():
    tr = Trajectory(task="task")
    tr.add_turn("user", "very first message")
    for i in range(25):
        tr.add_turn("assistant" if i % 2 == 0 else "user", f"middle turn {i}")
    tr.add_turn("assistant", "very last message")
    compressed = compress_trajectory(tr, max_turns=5)
    contents = [t.content for t in compressed.turns]
    assert "very first message" in contents


# ── export_dataset with compress=False ────────────────────────────────────────

def test_export_dataset_no_compression(tmp_path, monkeypatch):
    """compress=False exports all turns without running compress_trajectory."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    collector = TrajectoryCollector()
    session = "no-compress-session"
    collector.start(session, task="long task")
    for i in range(25):
        collector.record_turn(session, "user" if i % 2 == 0 else "assistant", f"msg {i}")
    collector.complete(session, outcome="success", reward=1.0)

    out_path = str(tmp_path / "no_compress.jsonl")
    result = export_dataset(collector, output_path=out_path, min_reward=0.0, compress=False)

    assert "Exported 1" in result
    lines = Path(out_path).read_text().strip().splitlines()
    record = json.loads(lines[0])
    # Without compression the full 25 turns should be present (mapped to conversations)
    assert len(record["conversations"]) == 25


# ── export_dataset reward filter ─────────────────────────────────────────────

def test_export_dataset_filters_by_min_reward(tmp_path, monkeypatch):
    """Trajectories below min_reward are excluded."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    collector = TrajectoryCollector()
    for session, reward in [("s1", 0.9), ("s2", 0.3)]:
        collector.start(session, task="t")
        collector.record_turn(session, "user", "go")
        collector.record_turn(session, "assistant", "done")
        collector.complete(session, outcome="success", reward=reward)

    out_path = str(tmp_path / "filtered.jsonl")
    result = export_dataset(collector, output_path=out_path, min_reward=0.5, compress=False)
    assert "Exported 1" in result  # only the 0.9 trajectory


# ── to_sharegpt with assistant turn having no content but tool calls ──────────

def test_to_sharegpt_empty_content_with_tool_call():
    """Assistant turn with empty content but a tool call uses tool_str alone."""
    tr = Trajectory(id="tc1")
    tr.add_turn("assistant", "", tool_calls=[{"name": "search", "input": {"q": "x"}}],
                tool_results=[{"content": "result"}])
    data = to_sharegpt(tr)
    gpt_turn = data["conversations"][0]
    assert gpt_turn["from"] == "gpt"
    assert "search" in gpt_turn["value"]


# ── Trajectory.set_outcome no assistant turn ──────────────────────────────────

def test_set_outcome_with_no_assistant_turn():
    """set_outcome is a no-op for reward backfill when no assistant turn exists."""
    tr = Trajectory()
    tr.add_turn("user", "hello")
    tr.set_outcome("failure", reward=0.0)
    assert tr.outcome == "failure"
    assert tr.reward == 0.0
    # User turn reward should remain None
    assert tr.turns[0].reward is None


# ── TrajectoryCollector.stats() all failures ──────────────────────────────────

def test_collector_stats_all_failures(collector):
    """stats() counts failures correctly and avg_reward is 0."""
    collector.start("f1", task="t")
    collector.record_turn("f1", "user", "go")
    collector.complete("f1", outcome="failure", reward=0.0)

    collector.start("f2", task="t")
    collector.record_turn("f2", "user", "go")
    collector.complete("f2", outcome="failure", reward=0.0)

    stats = collector.stats()
    assert stats["failure"] == 2
    assert stats["success"] == 0
    assert stats["avg_reward"] == pytest.approx(0.0)


# ── Turn field defaults ───────────────────────────────────────────────────────

def test_turn_tool_calls_default_empty_list():
    from jarvis.research.trajectory import Turn
    t = Turn(role="user", content="hello")
    assert t.tool_calls == []


def test_turn_tool_results_default_empty_list():
    from jarvis.research.trajectory import Turn
    t = Turn(role="user", content="hello")
    assert t.tool_results == []


def test_turn_reward_default_none():
    from jarvis.research.trajectory import Turn
    t = Turn(role="assistant", content="response")
    assert t.reward is None


# ── Trajectory field defaults ─────────────────────────────────────────────────

def test_trajectory_metadata_default_empty_dict():
    from jarvis.research.trajectory import Trajectory
    tr = Trajectory()
    assert tr.metadata == {}


def test_trajectory_outcome_default_unknown():
    from jarvis.research.trajectory import Trajectory
    tr = Trajectory()
    assert tr.outcome == "unknown"


def test_trajectory_reward_default_zero():
    from jarvis.research.trajectory import Trajectory
    tr = Trajectory()
    assert tr.reward == 0.0


def test_trajectory_id_is_hex_string():
    from jarvis.research.trajectory import Trajectory
    tr = Trajectory()
    assert len(tr.id) == 32
    int(tr.id, 16)


# ── TrajectoryCollector.complete saves and removes from active ────────────────

def test_complete_removes_session_from_active(collector):
    collector.start("sess_remove", task="test")
    collector.complete("sess_remove")
    assert collector.get("sess_remove") is None


def test_complete_returns_message_with_id_and_turns(collector):
    collector.start("sess_msg", task="test")
    collector.record_turn("sess_msg", "user", "question")
    collector.record_turn("sess_msg", "assistant", "answer")
    msg = collector.complete("sess_msg", outcome="success", reward=0.9)
    assert "1 turns" in msg or "2 turns" in msg
    assert "0.9" in msg


def test_complete_unknown_session_returns_message(collector):
    msg = collector.complete("never_started")
    assert "No active trajectory" in msg


# ── TrajectoryCollector.start stores task ─────────────────────────────────────

def test_start_stores_task_in_trajectory(collector):
    traj = collector.start("task_sess", task="build a web scraper")
    assert traj.task == "build a web scraper"


def test_start_stores_session_id(collector):
    traj = collector.start("sid123")
    assert traj.session_id == "sid123"
