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


def _fake_reportlab():
    """Build a minimal fake reportlab module hierarchy."""
    fake_doc = MagicMock()
    fake_styles = MagicMock()
    fake_styles.__getitem__ = MagicMock(return_value=MagicMock())

    fake_pagesizes = MagicMock()
    fake_pagesizes.A4 = (595, 842)

    fake_styles_mod = MagicMock()
    fake_styles_mod.getSampleStyleSheet = MagicMock(return_value=fake_styles)
    fake_styles_mod.ParagraphStyle = MagicMock(return_value=MagicMock())

    fake_units = MagicMock()
    fake_units.cm = 28.35

    fake_colors = MagicMock()

    fake_platypus = MagicMock()
    fake_platypus.SimpleDocTemplate = MagicMock(return_value=fake_doc)
    fake_platypus.Paragraph = MagicMock(return_value=MagicMock())
    fake_platypus.Spacer = MagicMock(return_value=MagicMock())
    fake_platypus.HRFlowable = MagicMock(return_value=MagicMock())

    return {
        "reportlab": MagicMock(),
        "reportlab.lib": MagicMock(),
        "reportlab.lib.pagesizes": fake_pagesizes,
        "reportlab.lib.styles": fake_styles_mod,
        "reportlab.lib.units": fake_units,
        "reportlab.lib.colors": fake_colors,
        "reportlab.platypus": fake_platypus,
    }, fake_doc


def test_export_pdf_no_history(memory_store):
    from jarvis.export import export_pdf
    mods, _ = _fake_reportlab()
    with patch.dict(sys.modules, mods):
        result = export_pdf(memory_store, "empty-session")
    assert "No conversation" in result


def test_export_pdf_success(memory_store, tmp_path):
    """export_pdf should build the PDF when reportlab is available (mocked)."""
    memory_store.save_message("s1", "user", "Hello")
    memory_store.save_message("s1", "assistant", "Greetings, Sir.")

    out_path = str(tmp_path / "out.pdf")
    mods, fake_doc = _fake_reportlab()

    from jarvis.export import export_pdf
    with patch.dict(sys.modules, mods):
        result = export_pdf(memory_store, "s1", output_path=out_path)

    assert "Exported" in result
    assert "2" in result
    fake_doc.build.assert_called_once()


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


@pytest.mark.asyncio
async def test_monitor_start_runs_loop(capsys):
    """Lines 22-26: start() sets _running, prints, then loops until stop() is called."""
    import asyncio as _asyncio
    mon = ProactiveMonitor(_make_jarvis_mock())

    async def _stop_after_one_tick():
        await _asyncio.sleep(0.05)
        mon.stop()

    await _asyncio.gather(
        mon.start(interval_seconds=0),
        _stop_after_one_tick(),
        return_exceptions=True,
    )
    out = capsys.readouterr().out
    assert "Started" in out
    assert mon._running is False


# ── Export markdown: content verification ─────────────────────────────────────

def test_export_markdown_contains_user_and_jarvis_sections(memory_store, tmp_path):
    """Verify the markdown output has **You** and **JARVIS** section headers."""
    from jarvis.export import export_markdown
    session_id = "format_test"
    memory_store.save_message(session_id, "user", "What is the weather?")
    memory_store.save_message(session_id, "assistant", "I'll check the forecast.")
    out_path = str(tmp_path / "session.md")
    result = export_markdown(memory_store, session_id, output_path=out_path)
    assert "Exported 2 messages" in result
    content = (tmp_path / "session.md").read_text()
    assert "**You**" in content
    assert "**JARVIS**" in content
    assert "What is the weather?" in content
    assert "I'll check the forecast." in content


def test_export_markdown_returns_count(memory_store, tmp_path):
    """The return message includes the message count."""
    from jarvis.export import export_markdown
    session_id = "count_test"
    for i in range(5):
        memory_store.save_message(session_id, "user", f"msg {i}")
    out_path = str(tmp_path / "out.md")
    result = export_markdown(memory_store, session_id, output_path=out_path)
    assert "5 messages" in result


# ── export_markdown: file content verification ─────────────────────────────────

def test_export_markdown_file_has_session_header(memory_store, tmp_path):
    """The exported file contains session ID in the header."""
    session_id = "my-session-123"
    memory_store.save_message(session_id, "user", "hello")
    out_path = str(tmp_path / "header_test.md")
    export_markdown(memory_store, session_id, output_path=out_path)
    content = (tmp_path / "header_test.md").read_text()
    assert session_id in content
    assert "JARVIS Session" in content


def test_export_markdown_separator_between_messages(memory_store, tmp_path):
    """Each message is separated by ---."""
    session_id = "sep-test"
    memory_store.save_message(session_id, "user", "question one")
    memory_store.save_message(session_id, "assistant", "answer one")
    out_path = str(tmp_path / "sep.md")
    export_markdown(memory_store, session_id, output_path=out_path)
    content = (tmp_path / "sep.md").read_text()
    assert content.count("---") >= 2


def test_export_markdown_uses_auto_filename(memory_store, tmp_path, monkeypatch):
    """When output_path is None, auto-generates a filename."""
    import os
    session_id = "auto-file"
    memory_store.save_message(session_id, "user", "auto test")
    monkeypatch.chdir(tmp_path)
    result = export_markdown(memory_store, session_id, output_path=None)
    assert "jarvis_session_" in result
    assert ".md" in result


# ── ProactiveMonitor edge cases ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_monitor_check_all_no_targets():
    """_check_all() with no monitor targets does nothing."""
    jarvis = MagicMock()
    jarvis.memory.get_monitor_targets = MagicMock(return_value=[])
    mon = ProactiveMonitor(jarvis)
    await mon._check_all()  # should not raise


@pytest.mark.asyncio
async def test_monitor_check_target_unknown_type_skips():
    """Target with type other than 'url' or 'file' returns early."""
    jarvis = MagicMock()
    mon = ProactiveMonitor(jarvis)
    target = {
        "name": "unknown",
        "target_type": "database",  # not url or file
        "target": "some_target",
        "action": "notify",
        "last_hash": None,
    }
    await mon._check_target(target)  # should not raise


@pytest.mark.asyncio
async def test_monitor_check_file_target_nonexistent(tmp_path):
    """File target that doesn't exist returns early without error."""
    jarvis = MagicMock()
    mon = ProactiveMonitor(jarvis)
    target = {
        "name": "missing-file",
        "target_type": "file",
        "target": str(tmp_path / "nonexistent.txt"),
        "action": "notify",
        "last_hash": None,
    }
    await mon._check_target(target)  # should not raise


@pytest.mark.asyncio
async def test_monitor_check_file_no_hash_change(tmp_path):
    """File target with matching last_hash — no action triggered."""
    file_path = tmp_path / "watched.txt"
    file_path.write_text("same content", encoding="utf-8")
    import hashlib
    content_hash = hashlib.sha256("same content".encode()).hexdigest()

    jarvis = MagicMock()
    jarvis.memory.update_monitor_hash = MagicMock()
    mon = ProactiveMonitor(jarvis)
    target = {
        "name": "stable-file",
        "target_type": "file",
        "target": str(file_path),
        "action": "notify",
        "last_hash": content_hash,  # same hash → no change
    }
    await mon._check_target(target)
    # No chat() triggered since content hasn't changed
    jarvis.chat.assert_not_called()


@pytest.mark.asyncio
async def test_monitor_hash_static_method():
    """_hash() produces consistent SHA-256 hex digest."""
    h1 = ProactiveMonitor._hash("hello world")
    h2 = ProactiveMonitor._hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest length
    assert h1 != ProactiveMonitor._hash("different content")


# ── ProactiveMonitor initial state ────────────────────────────────────────────

def test_monitor_running_false_initially():
    jarvis = MagicMock()
    mon = ProactiveMonitor(jarvis)
    assert mon._running is False


def test_monitor_jarvis_attribute():
    jarvis = MagicMock()
    mon = ProactiveMonitor(jarvis)
    assert mon.jarvis is jarvis


def test_monitor_stop_sets_running_false():
    jarvis = MagicMock()
    mon = ProactiveMonitor(jarvis)
    mon._running = True
    mon.stop()
    assert mon._running is False


# ── export_markdown: session ID in output ────────────────────────────────────

def test_export_markdown_includes_session_id(memory_store, tmp_path):
    session_id = "session-xyz-789"
    memory_store.save_message(session_id, "user", "hello")
    memory_store.save_message(session_id, "assistant", "hi there")
    out_path = str(tmp_path / "out.md")
    result = export_markdown(memory_store, session_id, output_path=out_path)
    content = Path(out_path).read_text()
    assert session_id in content


def test_export_markdown_jarvis_role_formatting(memory_store, tmp_path):
    session_id = "fmt-test"
    memory_store.save_message(session_id, "assistant", "I am JARVIS.")
    out_path = str(tmp_path / "fmt.md")
    export_markdown(memory_store, session_id, output_path=out_path)
    content = Path(out_path).read_text()
    assert "JARVIS" in content


# ── _hash: hex format ────────────────────────────────────────────────────────

def test_monitor_hash_is_hex_string():
    h = ProactiveMonitor._hash("any text")
    assert all(c in "0123456789abcdef" for c in h)
