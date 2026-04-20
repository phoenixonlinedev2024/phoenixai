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


# ── _fetch_url ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_url_returns_text(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
    fake_resp.__aexit__ = AsyncMock(return_value=False)
    fake_resp.text = AsyncMock(return_value="page content")

    fake_session = MagicMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)
    fake_session.get = MagicMock(return_value=fake_resp)

    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(return_value=fake_session)
    fake_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())

    jarvis = _make_jarvis_mock()
    mon = ProactiveMonitor(jarvis)

    with patch.dict(sys.modules, {"aiohttp": fake_aiohttp}):
        result = await mon._fetch_url("https://example.com")
    assert result == "page content"


# ── _trigger_action ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trigger_action_calls_chat():
    jarvis = _make_jarvis_mock()
    jarvis.chat = AsyncMock(return_value="action taken")
    mon = ProactiveMonitor(jarvis)
    await mon._trigger_action("my_monitor", "alert me", "https://example.com")
    jarvis.chat.assert_called_once()
    call_args = jarvis.chat.call_args[0][0]
    assert "my_monitor" in call_args
    assert "alert me" in call_args


# ── _check_target URL type ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_target_url_no_change(monkeypatch):
    jarvis = _make_jarvis_mock()
    content_hash = ProactiveMonitor._hash("page content")
    jarvis.memory.get_monitor_targets = MagicMock(return_value=[{
        "name": "web_mon",
        "target_type": "url",
        "target": "https://example.com",
        "action": "report",
        "last_hash": content_hash,
    }])

    mon = ProactiveMonitor(jarvis)
    monkeypatch.setattr(mon, "_fetch_url", AsyncMock(return_value="page content"))
    await mon._check_all()
    # No change → no action triggered
    jarvis.chat.assert_not_called()


@pytest.mark.asyncio
async def test_check_target_url_change_triggers_action(monkeypatch):
    jarvis = _make_jarvis_mock()
    jarvis.memory.get_monitor_targets = MagicMock(return_value=[{
        "name": "web_mon",
        "target_type": "url",
        "target": "https://example.com",
        "action": "summarise changes",
        "last_hash": ProactiveMonitor._hash("old content"),
    }])

    mon = ProactiveMonitor(jarvis)
    monkeypatch.setattr(mon, "_fetch_url", AsyncMock(return_value="new content"))
    await mon._check_all()
    jarvis.chat.assert_called_once()


@pytest.mark.asyncio
async def test_check_target_unknown_type_skips():
    jarvis = _make_jarvis_mock()
    jarvis.memory.get_monitor_targets = MagicMock(return_value=[{
        "name": "weird",
        "target_type": "database",
        "target": "postgres://localhost",
        "action": "alert",
        "last_hash": None,
    }])
    mon = ProactiveMonitor(jarvis)
    await mon._check_all()
    jarvis.chat.assert_not_called()


@pytest.mark.asyncio
async def test_check_target_exception_does_not_crash(monkeypatch):
    jarvis = _make_jarvis_mock()
    jarvis.memory.get_monitor_targets = MagicMock(return_value=[{
        "name": "flaky",
        "target_type": "url",
        "target": "https://broken.example",
        "action": "alert",
        "last_hash": None,
    }])
    mon = ProactiveMonitor(jarvis)
    monkeypatch.setattr(mon, "_fetch_url", AsyncMock(side_effect=ConnectionError("refused")))
    await mon._check_all()  # must not raise


# ── _notify ───────────────────────────────────────────────────────────────────

def test_notify_falls_back_to_print_on_error(capsys):
    from jarvis.monitor import _notify
    # plyer is mocked as MagicMock — notification.notify may fail or succeed
    # Either way, no exception should propagate
    _notify("Test Title", "Test message")
    # Just assert it doesn't raise


def test_notify_prints_when_plyer_raises(monkeypatch, capsys):
    from jarvis.monitor import _notify
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "plyer":
            raise ImportError("no plyer")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _notify("Alert", "Something happened")
    out = capsys.readouterr().out
    assert "Alert" in out
    assert "Something happened" in out
