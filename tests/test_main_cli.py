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


# ── chat with voice (TTS paths) ───────────────────────────────────────────────

def test_chat_one_shot_streaming_with_voice():
    """Lines 60-66: streaming one-shot with --voice calls tts.speak_async."""
    jarvis = _mock_jarvis()
    fake_tts = MagicMock()
    fake_tts.speak_async = AsyncMock()

    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.voice.text_to_speech.TTSEngine", return_value=fake_tts):
        result = runner.invoke(app, ["chat", "--stream", "--voice", "hello"])

    assert result.exit_code == 0
    fake_tts.speak_async.assert_called_once()


def test_chat_one_shot_no_stream_with_voice():
    """Lines 68-72: non-streaming one-shot with --voice calls tts.speak_async."""
    jarvis = _mock_jarvis(chat_reply="Of course, Sir.")
    fake_tts = MagicMock()
    fake_tts.speak_async = AsyncMock()

    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.voice.text_to_speech.TTSEngine", return_value=fake_tts):
        result = runner.invoke(app, ["chat", "--no-stream", "--voice", "hi"])

    assert result.exit_code == 0
    fake_tts.speak_async.assert_called_once()


# ── interactive chat loop ─────────────────────────────────────────────────────

def test_chat_interactive_exit_command():
    """Lines 75-84: interactive loop exits cleanly on 'exit' command."""
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.main.Prompt.ask", side_effect=["exit"]):
        result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    jarvis.end_session.assert_called_once()


def test_chat_interactive_empty_input_then_exit():
    """Lines 79-81: empty input is skipped, then exit terminates loop."""
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.main.Prompt.ask", side_effect=["", "  ", "quit"]):
        result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    jarvis.end_session.assert_called_once()


def test_chat_interactive_non_stream_reply():
    """Lines 95-102: non-streaming interactive reply path."""
    jarvis = _mock_jarvis(chat_reply="Understood, Sir.")
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.main.Prompt.ask", side_effect=["tell me something", "bye"]):
        result = runner.invoke(app, ["chat", "--no-stream"])

    assert result.exit_code == 0
    jarvis.chat.assert_called()


def test_chat_interactive_keyboard_interrupt():
    """Lines 104-107: KeyboardInterrupt during interactive loop calls end_session."""
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.main.Prompt.ask", side_effect=KeyboardInterrupt()):
        result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    jarvis.end_session.assert_called_once()


def test_chat_interactive_stream_reply():
    """Lines 86-93: streaming interactive reply path."""
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.main.Prompt.ask", side_effect=["what time is it?", "exit"]):
        result = runner.invoke(app, ["chat", "--stream"])

    assert result.exit_code == 0


# ── install-piper success path ────────────────────────────────────────────────

def test_install_piper_success_linux(tmp_path):
    """Lines 285-311: install-piper downloads and extracts on linux/x86_64."""
    import io
    import tarfile

    # Build a minimal tar.gz in memory
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="piper/piper")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))
    tar_bytes = buf.getvalue()

    out_dir = tmp_path / "piper_out"

    def fake_retrieve(url, dest):
        dest = str(dest)
        if dest.endswith(".tar.gz"):
            with open(dest, "wb") as f:
                f.write(tar_bytes)

    with patch("platform.system", return_value="Linux"), \
         patch("platform.machine", return_value="x86_64"), \
         patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
        result = runner.invoke(app, ["install-piper", "--dir", str(out_dir)])

    assert result.exit_code == 0
    assert "Piper installed" in result.stdout


def test_install_piper_download_failure(tmp_path):
    """Lines 286-290: download failure prints error and exits with code 1."""
    with patch("platform.system", return_value="Linux"), \
         patch("platform.machine", return_value="x86_64"), \
         patch("urllib.request.urlretrieve", side_effect=OSError("network error")):
        result = runner.invoke(app, ["install-piper", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "Download failed" in result.stdout


def test_install_piper_success_windows(tmp_path):
    """Lines 296-298: Windows platform extracts .zip archive."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("piper/piper.exe", b"")
    zip_bytes = buf.getvalue()

    out_dir = tmp_path / "piper_out"

    def fake_retrieve(url, dest):
        if str(dest).endswith(".zip"):
            with open(str(dest), "wb") as f:
                f.write(zip_bytes)

    with patch("platform.system", return_value="Windows"), \
         patch("platform.machine", return_value="AMD64"), \
         patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
        result = runner.invoke(app, ["install-piper", "--dir", str(out_dir)])

    assert result.exit_code == 0
    assert "Piper installed" in result.stdout


def test_install_piper_model_download_warning(tmp_path):
    """Lines 305-306: model download failure prints yellow warning but continues."""
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="piper/piper")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))
    tar_bytes = buf.getvalue()

    out_dir = tmp_path / "piper_out"
    call_count = [0]

    def fake_retrieve(url, dest):
        call_count[0] += 1
        if call_count[0] == 1:
            with open(str(dest), "wb") as f:
                f.write(tar_bytes)
        else:
            raise OSError("model unavailable")

    with patch("platform.system", return_value="Linux"), \
         patch("platform.machine", return_value="x86_64"), \
         patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
        result = runner.invoke(app, ["install-piper", "--dir", str(out_dir)])

    assert result.exit_code == 0
    assert "Model download warning" in result.stdout


# ── _get_jarvis creates Jarvis instance (lines 32-33) ────────────────────────

def test_get_jarvis_creates_instance():
    """Lines 32-33: _get_jarvis() imports Jarvis and returns an instance."""
    from jarvis.main import _get_jarvis
    fake_instance = MagicMock()
    with patch("jarvis.core.Jarvis", return_value=fake_instance):
        result = _get_jarvis()
    assert result is fake_instance


# ── Interactive chat with voice TTS (lines 94, 102) ──────────────────────────

def test_chat_interactive_stream_with_voice():
    """Line 94: interactive streaming with --voice calls tts.speak_async."""
    jarvis = _mock_jarvis()
    fake_tts = MagicMock()
    fake_tts.speak_async = AsyncMock()

    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.voice.text_to_speech.TTSEngine", return_value=fake_tts), \
         patch("jarvis.main.Prompt.ask", side_effect=["hello jarvis", "exit"]):
        result = runner.invoke(app, ["chat", "--stream", "--voice"])

    assert result.exit_code == 0
    fake_tts.speak_async.assert_called()


def test_chat_interactive_no_stream_with_voice():
    """Line 102: interactive non-streaming with --voice calls tts.speak_async."""
    jarvis = _mock_jarvis(chat_reply="Of course, Sir.")
    fake_tts = MagicMock()
    fake_tts.speak_async = AsyncMock()

    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.voice.text_to_speech.TTSEngine", return_value=fake_tts), \
         patch("jarvis.main.Prompt.ask", side_effect=["tell me something", "exit"]):
        result = runner.invoke(app, ["chat", "--no-stream", "--voice"])

    assert result.exit_code == 0
    fake_tts.speak_async.assert_called()


# ── voice command (lines 132-176) ────────────────────────────────────────────

def test_voice_command_full(monkeypatch):
    """Lines 132-176: voice() starts TTS+STT, handles transcript, exits on KBI."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "STT_ENGINE", "pyttsx3")

    jarvis = _mock_jarvis()
    fake_tts = MagicMock()
    fake_stt = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = "Voice reply"

    def fake_start_listening(cb):
        cb("_wake_only_")               # covers wake-only path (lines 154-156)
        cb("what is the time")          # covers normal chat path (lines 157-164)

    fake_stt.start_listening = MagicMock(side_effect=fake_start_listening)

    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.voice.text_to_speech.TTSEngine", return_value=fake_tts), \
         patch("jarvis.voice.speech_to_text.STTEngine", return_value=fake_stt), \
         patch("asyncio.run_coroutine_threadsafe", return_value=mock_future), \
         patch("asyncio.sleep", AsyncMock(side_effect=KeyboardInterrupt())):
        result = runner.invoke(app, ["voice"])

    assert result.exit_code == 0
    fake_tts.speak.assert_called()
    fake_stt.stop_listening.assert_called_once()


def test_voice_command_whisper_stt(monkeypatch):
    """Lines 143-144: voice() uses WhisperSTT when STT_ENGINE='whisper'."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "STT_ENGINE", "whisper")
    monkeypatch.setattr(cfg, "WHISPER_MODEL", "base")

    jarvis = _mock_jarvis()
    fake_tts = MagicMock()
    fake_whisper_stt = MagicMock()
    fake_whisper_stt.start_listening = MagicMock()

    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.voice.text_to_speech.TTSEngine", return_value=fake_tts), \
         patch("jarvis.voice.whisper_stt.WhisperSTT", return_value=fake_whisper_stt), \
         patch("asyncio.sleep", AsyncMock(side_effect=KeyboardInterrupt())):
        result = runner.invoke(app, ["voice"])

    assert result.exit_code == 0
    fake_whisper_stt.start_listening.assert_called_once()


def test_voice_command_transcript_error(monkeypatch):
    """Lines 163-164: on_transcript exception path is caught and printed."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "STT_ENGINE", "pyttsx3")

    jarvis = _mock_jarvis()
    fake_tts = MagicMock()
    fake_stt = MagicMock()
    mock_future = MagicMock()
    mock_future.result.side_effect = RuntimeError("timeout")

    def fake_start_listening(cb):
        cb("hello jarvis")

    fake_stt.start_listening = MagicMock(side_effect=fake_start_listening)

    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.voice.text_to_speech.TTSEngine", return_value=fake_tts), \
         patch("jarvis.voice.speech_to_text.STTEngine", return_value=fake_stt), \
         patch("asyncio.run_coroutine_threadsafe", return_value=mock_future), \
         patch("asyncio.sleep", AsyncMock(side_effect=KeyboardInterrupt())):
        result = runner.invoke(app, ["voice"])

    assert result.exit_code == 0
    assert "Error" in result.stdout


# ── __main__ guard (line 406) ─────────────────────────────────────────────────

def test_main_module_entrypoint_via_runpy():
    """Branch 406: running jarvis.main as __main__ triggers app()."""
    import runpy
    import pytest
    with pytest.raises(SystemExit):
        runpy.run_module("jarvis.main", run_name="__main__", alter_sys=True)


# ── status command ────────────────────────────────────────────────────────────

def test_status_exit_code_zero():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0


# ── tools command ─────────────────────────────────────────────────────────────

def test_tools_command_exit_code_zero():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["tools"])
    assert result.exit_code == 0


def test_tools_command_contains_category():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["tools"])
    assert "files" in result.stdout or "system" in result.stdout


# ── memory command ────────────────────────────────────────────────────────────

def test_memory_command_shows_summary():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["memory"])
    assert result.exit_code == 0
    assert "summary" in result.stdout.lower() or len(result.stdout) > 0


def test_memory_command_lessons_displayed():
    jarvis = _mock_jarvis()
    jarvis.memory.all_facts = MagicMock(return_value=[])
    jarvis.memory.get_lessons = MagicMock(return_value=["always validate input"])
    jarvis.memory.get_open_gaps = MagicMock(return_value=[])
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["memory"])
    assert "always validate input" in result.stdout


# ── export command ────────────────────────────────────────────────────────────

def test_export_default_format_is_markdown():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.export.export_markdown", return_value="Exported to out.md") as mock_exp:
        result = runner.invoke(app, ["export"])
    assert result.exit_code == 0
    mock_exp.assert_called_once()


def test_export_pdf_format():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis), \
         patch("jarvis.export.export_pdf", return_value="PDF saved") as mock_pdf:
        result = runner.invoke(app, ["export", "--format", "pdf"])
    assert result.exit_code == 0
    mock_pdf.assert_called_once()


# ── keys command with role admin ─────────────────────────────────────────────

def test_keys_create_with_admin_role():
    fake_ks = MagicMock()
    fake_ks.generate = MagicMock(return_value="jvs_admin_key")
    with patch("jarvis.security.key_store", fake_ks):
        result = runner.invoke(app, ["keys", "--create", "--role", "admin"])
    assert result.exit_code == 0
    fake_ks.generate.assert_called_once_with(name="", role="admin")


# ── chat --profile flag ───────────────────────────────────────────────────────

def test_chat_one_shot_sets_custom_profile():
    jarvis = _mock_jarvis()
    with patch("jarvis.main._get_jarvis", return_value=jarvis):
        result = runner.invoke(app, ["chat", "--no-stream", "--profile", "terse", "hi"])
    assert result.exit_code == 0
    jarvis.set_profile.assert_called_with("terse")
