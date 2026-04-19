"""Tests for jarvis.export and jarvis.monitor."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal",
                 "aiohttp", "plyer"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.export import export_markdown  # noqa: E402
from jarvis.monitor import ProactiveMonitor  # noqa: E402


# ── export_markdown ───────────────────────────────────────────────────────────

def test_export_markdown_no_history(memory_store):
    result = export_markdown(memory_store, "empty-session")
    assert "No conversation" in result


def test_export_markdown_creates_file(memory_store, tmp_path):
    memory_store.save_message("s1", "user", "Hello JARVIS")
    memory_store.save_message("s1", "assistant", "Good day, Sir.")
    out = str(tmp_path / "session.md")
    result = export_markdown(memory_store, "s1", output_path=out)
    assert "Exported" in result
    assert "2" in result
    content = Path(out).read_text(encoding="utf-8")
    assert "Hello JARVIS" in content
    assert "Good day, Sir." in content


def test_export_markdown_format(memory_store, tmp_path):
    memory_store.save_message("s1", "user", "test question")
    memory_store.save_message("s1", "assistant", "test answer")
    out = str(tmp_path / "out.md")
    export_markdown(memory_store, "s1", output_path=out)
    content = Path(out).read_text()
    assert "**You**" in content
    assert "**JARVIS**" in content
    assert "s1" in content


def test_export_pdf_missing_reportlab(memory_store):
    """export_pdf should gracefully handle missing reportlab."""
    memory_store.save_message("s1", "user", "q")
    from jarvis.export import export_pdf
    with patch.dict(sys.modules, {"reportlab": None,
                                   "reportlab.lib": None,
                                   "reportlab.lib.pagesizes": None,
                                   "reportlab.lib.styles": None,
                                   "reportlab.lib.units": None,
                                   "reportlab.lib.colors": None,
                                   "reportlab.platypus": None}):
        result = export_pdf(memory_store, "s1")
    assert "reportlab" in result.lower()


# ── ProactiveMonitor ──────────────────────────────────────────────────────────

def _make_jarvis_mock():
    jarvis = MagicMock()
    jarvis.memory.get_monitor_targets = MagicMock(return_value=[])
    jarvis.memory.update_monitor_hash = MagicMock()
    jarvis.chat = AsyncMock(return_value="action taken")
    return jarvis


def test_monitor_start_stop():
    mon = ProactiveMonitor(_make_jarvis_mock())
    assert mon._running is False
    mon.stop()
    assert mon._running is False


@pytest.mark.asyncio
async def test_monitor_check_all_no_targets():
    mon = ProactiveMonitor(_make_jarvis_mock())
    await mon._check_all()  # should not raise


@pytest.mark.asyncio
async def test_monitor_check_file_target_no_change(tmp_path):
    content = "stable content"
    fpath = tmp_path / "watch.txt"
    fpath.write_text(content)
    import hashlib
    h = hashlib.sha256(content.encode()).hexdigest()

    jarvis = _make_jarvis_mock()
    jarvis.memory.get_monitor_targets = MagicMock(return_value=[{
        "name": "watcher",
        "target_type": "file",
        "target": str(fpath),
        "action": "notify me",
        "last_hash": h,
    }])

    mon = ProactiveMonitor(jarvis)
    await mon._check_all()
    jarvis.chat.assert_not_called()


@pytest.mark.asyncio
async def test_monitor_check_file_target_change_triggers_action(tmp_path):
    fpath = tmp_path / "watch.txt"
    fpath.write_text("changed content")

    jarvis = _make_jarvis_mock()
    jarvis.memory.get_monitor_targets = MagicMock(return_value=[{
        "name": "watcher",
        "target_type": "file",
        "target": str(fpath),
        "action": "summarise changes",
        "last_hash": "old_hash_value",
    }])

    mon = ProactiveMonitor(jarvis)
    await mon._check_all()
    jarvis.chat.assert_called_once()
    jarvis.memory.update_monitor_hash.assert_called_once()


@pytest.mark.asyncio
async def test_monitor_check_missing_file_skips():
    jarvis = _make_jarvis_mock()
    jarvis.memory.get_monitor_targets = MagicMock(return_value=[{
        "name": "missing",
        "target_type": "file",
        "target": "/nonexistent/path/file.txt",
        "action": "alert",
        "last_hash": None,
    }])

    mon = ProactiveMonitor(jarvis)
    await mon._check_all()  # should not raise
    jarvis.chat.assert_not_called()


def test_monitor_hash_is_deterministic():
    h1 = ProactiveMonitor._hash("hello")
    h2 = ProactiveMonitor._hash("hello")
    assert h1 == h2


def test_monitor_hash_differs_for_different_content():
    assert ProactiveMonitor._hash("a") != ProactiveMonitor._hash("b")
