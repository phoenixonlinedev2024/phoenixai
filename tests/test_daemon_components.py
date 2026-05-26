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


# ── Additional daemon component tests ────────────────────────────────────────

def test_voice_loop_jarvis_stored_as_attribute():
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    assert vl.jarvis is jarvis


def test_voice_loop_tts_is_none_initially():
    vl = VoiceLoop(_make_jarvis())
    assert vl._tts is None


def test_voice_loop_stt_is_none_initially():
    vl = VoiceLoop(_make_jarvis())
    assert vl._stt is None


def test_scheduler_jarvis_stored_as_attribute():
    jarvis = _make_jarvis()
    sched = JarvisScheduler(jarvis)
    assert sched.jarvis is jarvis


@pytest.mark.asyncio
async def test_scheduler_run_reflection_empty_result_is_ok(capsys):
    jarvis = _make_jarvis()
    jarvis.learner.reflect = AsyncMock(return_value={})
    sched = JarvisScheduler(jarvis)
    await sched._run_reflection()
    out = capsys.readouterr().out
    assert "0" in out


@pytest.mark.asyncio
async def test_scheduler_run_task_prints_result(capsys):
    jarvis = _make_jarvis()
    jarvis.chat = AsyncMock(return_value="All done with the task.")
    sched = JarvisScheduler(jarvis)
    await sched._run_task("my-task", "do something")
    out = capsys.readouterr().out
    assert "All done" in out


def test_scheduler_start_prints_running_message(capsys, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "REFLECTION_INTERVAL_HOURS", 6)
    jarvis = _make_jarvis()
    sched = JarvisScheduler(jarvis)
    monkeypatch.setattr(sched, "_load_tasks", MagicMock())
    sched.start()
    out = capsys.readouterr().out
    assert "Running" in out or "Reflection" in out


# ── JarvisScheduler additional ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scheduler_run_task_updates_memory(capsys):
    jarvis = _make_jarvis()
    sched = JarvisScheduler(jarvis)
    await sched._run_task("mem-task", "check things")
    jarvis.memory.update_task_run.assert_called_once()


def test_scheduler_load_tasks_prints_on_success(capsys, monkeypatch):
    task = {"name": "morning", "cron": "0 9 * * *", "prompt": "good morning"}
    jarvis = _make_jarvis(scheduled_tasks=[task])
    sched = JarvisScheduler(jarvis)
    with patch("jarvis.daemon.CronTrigger") as mock_cron:
        mock_cron.from_crontab.return_value = MagicMock()
        sched._load_tasks()
    out = capsys.readouterr().out
    assert "morning" in out


def test_scheduler_load_multiple_tasks_adds_multiple_jobs(monkeypatch):
    tasks = [
        {"name": "task1", "cron": "0 8 * * *", "prompt": "first"},
        {"name": "task2", "cron": "0 20 * * *", "prompt": "second"},
    ]
    jarvis = _make_jarvis(scheduled_tasks=tasks)
    sched = JarvisScheduler(jarvis)
    count_before = sched.scheduler.add_job.call_count
    with patch("jarvis.daemon.CronTrigger") as mock_cron:
        mock_cron.from_crontab.return_value = MagicMock()
        sched._load_tasks()
    assert sched.scheduler.add_job.call_count == count_before + 2


# ── VoiceLoop additional ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scheduler_run_reflection_uses_client():
    jarvis = _make_jarvis()
    sched = JarvisScheduler(jarvis)
    await sched._run_reflection()
    jarvis.learner.reflect.assert_called_once_with(
        jarvis._session_transcript, jarvis.client
    )


def test_voice_loop_stop_when_no_stt_does_not_raise():
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    vl._stt = None
    vl.stop()  # should not raise


# ── JarvisScheduler.stop calls shutdown ──────────────────────────────────────

def test_scheduler_stop_calls_shutdown():
    jarvis = _make_jarvis()
    sched = JarvisScheduler(jarvis)
    sched.stop()
    sched.scheduler.shutdown.assert_called_once_with(wait=False)


# ── JarvisScheduler._load_tasks skips tasks with bad cron ────────────────────

def test_scheduler_load_tasks_bad_cron_prints_error(monkeypatch, capsys):
    monkeypatch.setattr("jarvis.daemon.CronTrigger", MagicMock(
        **{"from_crontab.side_effect": ValueError("bad cron")}
    ))
    jarvis = _make_jarvis(scheduled_tasks=[
        {"name": "bad_task", "cron": "not a cron", "prompt": "do stuff"},
    ])
    sched = JarvisScheduler(jarvis)
    sched._load_tasks()
    out = capsys.readouterr().out
    assert "Failed" in out or "bad_task" in out


# ── JarvisScheduler._run_task exception path ─────────────────────────────────

@pytest.mark.asyncio
async def test_scheduler_run_task_exception_prints_error(capsys):
    jarvis = _make_jarvis()
    jarvis.chat = AsyncMock(side_effect=RuntimeError("chat down"))
    sched = JarvisScheduler(jarvis)
    await sched._run_task("failing_task", "please work")
    out = capsys.readouterr().out
    assert "failing_task" in out
    assert "error" in out.lower() or "Error" in out


# ── VoiceLoop attributes are correct type ────────────────────────────────────

def test_voice_loop_jarvis_is_stored_jarvis():
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    assert vl.jarvis is jarvis


def test_voice_loop_attributes_start_as_none():
    jarvis = _make_jarvis()
    vl = VoiceLoop(jarvis)
    assert vl._tts is None
    assert vl._stt is None


# ── JarvisScheduler._run_reflection with non-empty result ───────────────────

@pytest.mark.asyncio
async def test_scheduler_run_reflection_counts_lessons(capsys):
    jarvis = _make_jarvis()
    jarvis.learner.reflect = AsyncMock(return_value={
        "lessons": ["lesson_a", "lesson_b"],
        "facts": {},
    })
    sched = JarvisScheduler(jarvis)
    await sched._run_reflection()
    out = capsys.readouterr().out
    assert "2" in out
