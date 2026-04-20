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
