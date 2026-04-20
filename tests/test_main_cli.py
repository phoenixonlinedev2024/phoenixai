"""Tests for jarvis.main CLI entry points via Typer's CliRunner."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.main import app  # noqa: E402


runner = CliRunner()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_jarvis(chat_reply: str = "Sir, how may I assist?"):
    j = MagicMock()
    j.set_profile = MagicMock()
    j.status = MagicMock(return_value="JARVIS running")
    j.chat = AsyncMock(return_value=chat_reply)
    j.end_session = AsyncMock(return_value="Session closed.")
    j._session_id = "session-xyz"

    # Streaming chat: async iterator yielding tokens
    async def _stream(text):
        for token in ["Hello ", "World"]:
            yield token

    j.stream_chat = _stream

    # Memory
    j.memory.all_facts = MagicMock(return_value=[{"key": "lang", "value": "Python"}])
    j.memory.get_lessons = MagicMock(return_value=["learn A", "learn B"])
    j.memory.get_open_gaps = MagicMock(return_value=[{"id": 1, "description": "missing tool X"}])
    j.memory.summary = MagicMock(return_value="2 facts, 2 lessons, 1 gap")

    # Registry
    tool = MagicMock(name="read_file", category="files", dynamic=False,
                     description="Read a file.")
    tool.name = "read_file"
    j.registry.all = MagicMock(return_value=[tool])
    return j


# ── Banner is printed ─────────────────────────────────────────────────────────

def test_status_command_prints_jarvis_status():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "JARVIS running" in result.stdout


def test_tools_command_lists_registered_tools():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["tools"])

    assert result.exit_code == 0
    assert "read_file" in result.stdout


def test_memory_command_displays_facts_and_lessons():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["memory"])

    assert result.exit_code == 0
    assert "Python" in result.stdout
    assert "learn A" in result.stdout
    assert "missing tool X" in result.stdout


def test_memory_command_no_facts():
    jarvis = _mock_jarvis()
    jarvis.memory.all_facts = MagicMock(return_value=[])
    jarvis.memory.get_lessons = MagicMock(return_value=[])
    jarvis.memory.get_open_gaps = MagicMock(return_value=[])
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["memory"])
    assert result.exit_code == 0


# ── chat (one-shot) ───────────────────────────────────────────────────────────

def test_chat_one_shot_non_streaming():
    jarvis = _mock_jarvis(chat_reply="All systems nominal.")
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["chat", "--no-stream", "hello"])

    assert result.exit_code == 0
    assert "All systems nominal" in result.stdout
    jarvis.set_profile.assert_called_with("default")


def test_chat_one_shot_streaming():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["chat", "--stream", "hello"])

    assert result.exit_code == 0
    # Streamed tokens appear in output
    assert "Hello" in result.stdout
    assert "World" in result.stdout


def test_chat_with_profile_flag():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        runner.invoke(app, ["chat", "--no-stream", "--profile", "terse", "hi"])
    jarvis.set_profile.assert_called_with("terse")


# ── export ────────────────────────────────────────────────────────────────────

def test_export_markdown(tmp_path):
    jarvis = _mock_jarvis()
    out = tmp_path / "session.md"
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.export.export_markdown", return_value="Exported markdown to out"):
        result = runner.invoke(app, ["export", "--format", "markdown", "--output", str(out)])
    assert result.exit_code == 0
    assert "Exported" in result.stdout


def test_export_pdf(tmp_path):
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.export.export_pdf", return_value="PDF ok"):
        result = runner.invoke(app, ["export", "--format", "pdf"])
    assert result.exit_code == 0
    assert "PDF ok" in result.stdout


# ── benchmark ─────────────────────────────────────────────────────────────────

def test_benchmark_trend_flag():
    jarvis = _mock_jarvis()
    fake_runner = MagicMock()
    fake_runner.trend = MagicMock(return_value={"runs": 4, "improving": True})
    fake_runner.run_suite = AsyncMock()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.self_improve.BenchmarkRunner", return_value=fake_runner):
        result = runner.invoke(app, ["benchmark", "--trend"])
    assert result.exit_code == 0
    assert "improving" in result.stdout or "True" in result.stdout
    fake_runner.run_suite.assert_not_called()


def test_benchmark_runs_suite():
    jarvis = _mock_jarvis()
    fake_runner = MagicMock()
    fake_runner.run_suite = AsyncMock(return_value={
        "pass_rate": 0.8,
        "passed": 4,
        "total": 5,
        "avg_latency_s": 1.2,
        "results": [
            {"case_id": "math", "passed": True, "score": 1.0, "latency_s": 0.5, "error": None},
            {"case_id": "reasoning", "passed": False, "score": 0.3, "latency_s": 2.0, "error": "nope"},
        ],
    })
    fake_runner.trend = MagicMock()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.self_improve.BenchmarkRunner", return_value=fake_runner):
        result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 0
    assert "Pass rate" in result.stdout
    assert "80%" in result.stdout


# ── keys ──────────────────────────────────────────────────────────────────────

def test_keys_list_empty():
    fake_ks = MagicMock()
    fake_ks.list_keys = MagicMock(return_value=[])
    with patch("jarvis.security.key_store", fake_ks):
        result = runner.invoke(app, ["keys"])
    assert result.exit_code == 0
    assert "No API keys" in result.stdout


def test_keys_list_with_entries():
    fake_ks = MagicMock()
    fake_ks.list_keys = MagicMock(return_value=[
        {"key_prefix": "jvs_abc...", "role": "admin", "name": "root",
         "calls": 42, "last_used": "2024-01-01"},
    ])
    with patch("jarvis.security.key_store", fake_ks):
        result = runner.invoke(app, ["keys"])
    assert result.exit_code == 0
    assert "jvs_abc" in result.stdout
    assert "admin" in result.stdout


def test_keys_create():
    fake_ks = MagicMock()
    fake_ks.generate = MagicMock(return_value="jvs_generated_xyz")
    with patch("jarvis.security.key_store", fake_ks):
        result = runner.invoke(app, ["keys", "--create", "--name", "myapp", "--role", "user"])
    assert result.exit_code == 0
    assert "jvs_generated_xyz" in result.stdout
    fake_ks.generate.assert_called_once_with(name="myapp", role="user")


def test_keys_revoke_success():
    fake_ks = MagicMock()
    fake_ks._keys = {"jvs_abcdef_rest": object()}
    fake_ks.revoke = MagicMock(return_value=True)
    with patch("jarvis.security.key_store", fake_ks):
        result = runner.invoke(app, ["keys", "--revoke", "jvs_abcdef"])
    assert result.exit_code == 0
    assert "Revoked" in result.stdout


def test_keys_revoke_not_found():
    fake_ks = MagicMock()
    fake_ks._keys = {}
    fake_ks.revoke = MagicMock(return_value=False)
    with patch("jarvis.security.key_store", fake_ks):
        result = runner.invoke(app, ["keys", "--revoke", "nothing"])
    assert result.exit_code == 0
    assert "not found" in result.stdout.lower()


# ── daemon (no-op because run_daemon is mocked) ───────────────────────────────

def test_daemon_invokes_run_daemon():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.daemon.run_daemon", AsyncMock()):
        result = runner.invoke(app, ["daemon"])
    assert result.exit_code == 0


# ── install-piper (platform error path) ───────────────────────────────────────

def test_install_piper_unsupported_platform():
    with patch("platform.system", return_value="plan9"), \
         patch("platform.machine", return_value="nonexistent-arch"):
        result = runner.invoke(app, ["install-piper"])
    assert result.exit_code == 1
    assert "Unsupported" in result.stdout
