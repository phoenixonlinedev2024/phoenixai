"""Tests for JarvisScheduler and VoiceLoop (daemon.py internal components)."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    fakes = {
        "anthropic": MagicMock(),
        "chromadb": MagicMock(),
        "sentence_transformers": MagicMock(),
        "openai": MagicMock(),
        "modal": MagicMock(),
        "apscheduler": MagicMock(),
        "apscheduler.schedulers": MagicMock(),
        "apscheduler.schedulers.asyncio": MagicMock(),
        "apscheduler.triggers": MagicMock(),
        "apscheduler.triggers.cron": MagicMock(),
    }
    fakes["anthropic"].AsyncAnthropic = MagicMock
    for name, mod in fakes.items():
        sys.modules.setdefault(name, mod)


_inject_fakes()

from jarvis.daemon import JarvisScheduler, VoiceLoop  # noqa: E402


def _make_jarvis(scheduled_tasks=None):
    jarvis = MagicMock()
    jarvis.memory.get_scheduled_tasks.return_value = scheduled_tasks or []
    jarvis.memory.update_task_run = MagicMock()
    jarvis.learner.reflect = AsyncMock(return_value={"lessons": ["be concise"]})
    jarvis.chat = AsyncMock(return_value="task done")
    jarvis._session_transcript = []
    jarvis.client = MagicMock()
    return jarvis


# ── JarvisScheduler ───────────────────────────────────────────────────────────

def test_scheduler_stop_calls_shutdown():
    jarvis = _make_jarvis()
    sched = JarvisScheduler(jarvis)
    sched.stop()
    sched.scheduler.shutdown.assert_called_once_with(wait=False)


def test_scheduler_load_tasks_empty():
    jarvis = _make_jarvis(scheduled_tasks=[])
    sched = JarvisScheduler(jarvis)
    sched._load_tasks()
    sched.scheduler.add_job.assert_not_called()


def test_scheduler_load_tasks_adds_job(monkeypatch):
    jarvis = _make_jarvis(scheduled_tasks=[{"name": "daily", "cron": "0 9 * * *", "prompt": "reflect"}])
    sched = JarvisScheduler(jarvis)
    # Mock CronTrigger.from_crontab to avoid apscheduler parsing
    with patch("jarvis.daemon.CronTrigger") as mock_cron:
        mock_cron.from_crontab.return_value = MagicMock()
        sched._load_tasks()
    sched.scheduler.add_job.assert_called_once()


def test_scheduler_load_tasks_bad_cron_is_logged(capsys):
    jarvis = _make_jarvis(scheduled_tasks=[{"name": "bad", "cron": "not a cron", "prompt": "x"}])
    sched = JarvisScheduler(jarvis)
    with patch("jarvis.daemon.CronTrigger") as mock_cron:
        mock_cron.from_crontab.side_effect = ValueError("bad cron")
        sched._load_tasks()
    out = capsys.readouterr().out
    assert "Failed to load" in out


@pytest.mark.asyncio
async def test_scheduler_run_task_calls_chat():
    jarvis = _make_jarvis()
    sched = JarvisScheduler(jarvis)
    await sched._run_task("test-task", "do the thing")
    jarvis.chat.assert_called_once_with("do the thing")
    jarvis.memory.update_task_run.assert_called_once()


@pytest.mark.asyncio
async def test_scheduler_run_task_exception_is_caught(capsys):
    jarvis = _make_jarvis()
    jarvis.chat = AsyncMock(side_effect=RuntimeError("api down"))
    sched = JarvisScheduler(jarvis)
    await sched._run_task("failing-task", "prompt")
    out = capsys.readouterr().out
    assert "error" in out.lower()


@pytest.mark.asyncio
async def test_scheduler_run_reflection():
    jarvis = _make_jarvis()
    sched = JarvisScheduler(jarvis)
    await sched._run_reflection()
    jarvis.learner.reflect.assert_called_once()


@pytest.mark.asyncio
async def test_scheduler_run_reflection_prints_count(capsys):
    jarvis = _make_jarvis()
    jarvis.learner.reflect = AsyncMock(return_value={"lessons": ["a", "b", "c"]})
    sched = JarvisScheduler(jarvis)
    await sched._run_reflection()
    out = capsys.readouterr().out
    assert "3" in out


def test_scheduler_start(monkeypatch):
    jarvis = _make_jarvis()
    sched = JarvisScheduler(jarvis)
    monkeypatch.setattr(sched, "_load_tasks", MagicMock())
    sched.start()
    sched._load_tasks.assert_called_once()
    # add_job called at least once for the reflection job
    assert sched.scheduler.add_job.called
    sched.scheduler.start.assert_called_once()


# ── VoiceLoop ─────────────────────────────────────────────────────────────────

def test_voice_loop_start_noop_when_disabled(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", False)
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    vl.start(loop)
    assert vl._tts is None
    assert vl._stt is None


def test_voice_loop_stop_calls_stt_stop():
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    fake_stt = MagicMock()
    vl._stt = fake_stt
    vl.stop()
    fake_stt.stop_listening.assert_called_once()


def test_voice_loop_stop_no_stt_is_safe():
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    vl.stop()  # no _stt set — should not raise


def test_voice_loop_get_tts_creates_and_caches(monkeypatch):
    from jarvis.config import cfg
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    fake_tts = MagicMock()
    with patch("jarvis.voice.text_to_speech.TTSEngine", return_value=fake_tts):
        t1 = vl._get_tts()
        t2 = vl._get_tts()
    assert t1 is t2


def test_voice_loop_get_stt_default(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "STT_ENGINE", "google")
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    fake_stt = MagicMock()
    with patch("jarvis.voice.speech_to_text.STTEngine", return_value=fake_stt):
        s1 = vl._get_stt()
        s2 = vl._get_stt()
    assert s1 is s2


def test_voice_loop_get_stt_whisper(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "STT_ENGINE", "whisper")
    monkeypatch.setattr(cfg, "WHISPER_MODEL", "tiny")
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    fake_stt = MagicMock()
    with patch("jarvis.voice.whisper_stt.WhisperSTT", return_value=fake_stt):
        s = vl._get_stt()
    assert s is not None


def test_voice_loop_on_transcript_wake_only(monkeypatch):
    from jarvis.config import cfg
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    fake_tts = MagicMock()
    vl._tts = fake_tts
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    vl._on_transcript("_wake_only_", loop)
    fake_tts.speak.assert_called_once()


def test_voice_loop_on_transcript_regular(monkeypatch):
    from jarvis.config import cfg
    jarvis = _make_jarvis()
    jarvis.voice_chat = AsyncMock(return_value="Of course, Sir.")
    vl = VoiceLoop(jarvis)
    fake_tts = MagicMock()
    vl._tts = fake_tts
    loop = asyncio.new_event_loop()

    fake_future = MagicMock()
    fake_future.result.return_value = "Of course, Sir."
    with patch("asyncio.run_coroutine_threadsafe", return_value=fake_future):
        vl._on_transcript("open the pod bay doors", loop)

    fake_tts.speak.assert_called_with("Of course, Sir.")
    loop.close()


def test_voice_loop_on_transcript_exception(monkeypatch, capsys):
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    fake_tts = MagicMock()
    vl._tts = fake_tts
    loop = asyncio.new_event_loop()

    fake_future = MagicMock()
    fake_future.result.side_effect = RuntimeError("voice crash")
    with patch("asyncio.run_coroutine_threadsafe", return_value=fake_future):
        vl._on_transcript("something", loop)

    out = capsys.readouterr().out
    assert "Error" in out
    loop.close()
