"""Tests for jarvis.voice (TTS + STT engines)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.voice.speech_to_text import STTEngine  # noqa: E402
from jarvis.voice.text_to_speech import TTSEngine  # noqa: E402
from jarvis.voice.whisper_stt import WhisperSTT, _load_model, _MODEL_CACHE  # noqa: E402


# ── TTSEngine ─────────────────────────────────────────────────────────────────

def test_tts_init_uses_config_engine_by_default(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TTS_ENGINE", "pyttsx3")
    engine = TTSEngine()
    assert engine.engine_name == "pyttsx3"


def test_tts_init_with_explicit_engine():
    engine = TTSEngine(engine="elevenlabs")
    assert engine.engine_name == "elevenlabs"


def test_tts_speak_noop_when_voice_disabled(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", False)
    engine = TTSEngine(engine="pyttsx3")
    # Should not raise even without pyttsx3
    engine.speak("hello")


def test_tts_speak_noop_on_empty_text(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", True)
    engine = TTSEngine(engine="pyttsx3")
    engine.speak("")  # empty, ignored
    engine.speak("   ")  # whitespace, ignored


def test_tts_speak_pyttsx3_path(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", True)

    fake_engine = MagicMock()
    fake_engine.getProperty = MagicMock(return_value=[])
    fake_pyttsx3 = MagicMock()
    fake_pyttsx3.init = MagicMock(return_value=fake_engine)

    with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
        engine = TTSEngine(engine="pyttsx3")
        engine.speak("hello")

    fake_engine.say.assert_called_once_with("hello")
    fake_engine.runAndWait.assert_called_once()


def test_tts_pyttsx3_error_prints(monkeypatch, capsys):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", True)
    fake_pyttsx3 = MagicMock()
    fake_pyttsx3.init = MagicMock(side_effect=RuntimeError("no audio"))
    with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
        engine = TTSEngine(engine="pyttsx3")
        engine.speak("hello")
    captured = capsys.readouterr()
    assert "pyttsx3 error" in captured.out


def test_tts_elevenlabs_fallback_on_error(monkeypatch, capsys):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", True)
    monkeypatch.setattr(cfg, "ELEVENLABS_API_KEY", "k")

    fake_eleven = MagicMock()
    fake_eleven.ElevenLabs = MagicMock(side_effect=RuntimeError("api down"))
    fake_eleven.play = MagicMock()

    fake_pyttsx3 = MagicMock()
    fake_engine = MagicMock()
    fake_engine.getProperty = MagicMock(return_value=[])
    fake_pyttsx3.init = MagicMock(return_value=fake_engine)

    with patch.dict(sys.modules, {"elevenlabs": fake_eleven, "pyttsx3": fake_pyttsx3}):
        engine = TTSEngine(engine="elevenlabs")
        engine.speak("hello")

    out = capsys.readouterr().out
    assert "ElevenLabs error" in out
    # Falls back to pyttsx3
    fake_engine.say.assert_called_once()


def test_tts_piper_available_checks_shutil(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/piper" if name == "piper" else None)
    engine = TTSEngine(engine="piper")
    assert engine._piper_available() is True


def test_tts_piper_not_available(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "PIPER_BINARY", "/nonexistent/piper")
    engine = TTSEngine(engine="piper")
    assert engine._piper_available() is False


def test_tts_set_voice_by_name_found():
    voice = MagicMock(id="voice-id-1")
    voice.name = "English (UK) Male"
    fake_engine = MagicMock()
    fake_engine.getProperty = MagicMock(return_value=[voice])
    eng = TTSEngine(engine="pyttsx3")
    eng._pyttsx3_engine = fake_engine
    assert eng.set_voice_by_name("english") is True
    fake_engine.setProperty.assert_called_with("voice", "voice-id-1")


def test_tts_set_voice_by_name_not_found():
    voice = MagicMock(id="v1")
    voice.name = "French Female"
    fake_engine = MagicMock()
    fake_engine.getProperty = MagicMock(return_value=[voice])
    eng = TTSEngine(engine="pyttsx3")
    eng._pyttsx3_engine = fake_engine
    assert eng.set_voice_by_name("german") is False


def test_tts_list_voices():
    v1 = MagicMock(); v1.name = "A"
    v2 = MagicMock(); v2.name = "B"
    fake_engine = MagicMock()
    fake_engine.getProperty = MagicMock(return_value=[v1, v2])
    eng = TTSEngine(engine="pyttsx3")
    eng._pyttsx3_engine = fake_engine
    assert eng.list_voices() == ["A", "B"]


def test_tts_list_voices_error_returns_empty():
    fake_engine = MagicMock()
    fake_engine.getProperty = MagicMock(side_effect=RuntimeError("fail"))
    eng = TTSEngine(engine="pyttsx3")
    eng._pyttsx3_engine = fake_engine
    assert eng.list_voices() == []


@pytest.mark.asyncio
async def test_tts_speak_async_runs_in_executor(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", False)
    eng = TTSEngine(engine="pyttsx3")
    await eng.speak_async("hello")  # should not raise


# ── STTEngine ─────────────────────────────────────────────────────────────────

def test_stt_init_state():
    stt = STTEngine()
    assert stt._running is False
    assert stt._thread is None


def test_stt_start_listening_sets_running():
    stt = STTEngine()

    # Patch the internal _listen_loop so thread exits quickly
    def fake_loop(callback):
        pass
    stt._listen_loop = fake_loop

    stt.start_listening(lambda t: None)
    assert stt._running is True
    assert stt._thread is not None
    stt.stop_listening()


def test_stt_start_listening_idempotent():
    stt = STTEngine()
    stt._listen_loop = lambda cb: None
    stt.start_listening(lambda t: None)
    first_thread = stt._thread
    stt.start_listening(lambda t: None)
    # Second call should NOT create a new thread
    assert stt._thread is first_thread
    stt.stop_listening()


def test_stt_stop_listening_sets_flag():
    stt = STTEngine()
    stt._listen_loop = lambda cb: None
    stt.start_listening(lambda t: None)
    stt.stop_listening()
    assert stt._running is False


def test_stt_listen_loop_missing_speech_recognition(capsys):
    stt = STTEngine()
    stt._running = True
    with patch.dict(sys.modules, {"speech_recognition": None}):
        stt._listen_loop(lambda t: None)
    captured = capsys.readouterr()
    assert "not installed" in captured.out.lower()


@pytest.mark.asyncio
async def test_stt_listen_once_runs_in_executor(monkeypatch):
    stt = STTEngine()
    monkeypatch.setattr(stt, "_record_phrase", lambda timeout: "transcribed")
    result = await stt.listen_once(timeout=5)
    assert result == "transcribed"


def test_stt_record_phrase_returns_text(monkeypatch):
    stt = STTEngine()
    fake_audio = MagicMock()
    fake_sr = MagicMock()
    fake_recognizer = MagicMock()
    fake_recognizer.recognize_google.return_value = "hello jarvis"
    fake_recognizer.listen.return_value = fake_audio
    fake_sr.Recognizer.return_value = fake_recognizer

    fake_mic = MagicMock()
    fake_mic.__enter__ = MagicMock(return_value=fake_mic)
    fake_mic.__exit__ = MagicMock(return_value=False)
    fake_sr.Microphone.return_value = fake_mic

    with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
        result = stt._record_phrase(timeout=5)
    assert result == "hello jarvis"


def test_stt_record_phrase_returns_none_on_exception(monkeypatch, capsys):
    stt = STTEngine()
    fake_sr = MagicMock()
    fake_sr.Recognizer.return_value.listen.side_effect = RuntimeError("no mic")
    fake_mic = MagicMock()
    fake_mic.__enter__ = MagicMock(return_value=fake_mic)
    fake_mic.__exit__ = MagicMock(return_value=False)
    fake_sr.Microphone.return_value = fake_mic

    with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
        result = stt._record_phrase(timeout=5)
    assert result is None
    assert "STT error" in capsys.readouterr().out


# ── WhisperSTT / _load_model ──────────────────────────────────────────────────

def test_whisper_load_model_caches():
    _MODEL_CACHE.clear()
    fake_whisper = MagicMock()
    fake_whisper.load_model = MagicMock(return_value="MODEL_OBJ")

    with patch.dict(sys.modules, {"whisper": fake_whisper}):
        m1 = _load_model("base")
        m2 = _load_model("base")

    assert m1 == "MODEL_OBJ"
    assert m1 is m2
    # Only loaded once
    fake_whisper.load_model.assert_called_once_with("base")
    _MODEL_CACHE.clear()


def test_whisper_load_model_missing_whisper():
    _MODEL_CACHE.clear()
    with patch.dict(sys.modules, {"whisper": None}):
        with pytest.raises(RuntimeError) as exc:
            _load_model("tiny")
    assert "openai-whisper" in str(exc.value)


def test_whisper_stt_init_defaults():
    w = WhisperSTT()
    assert w.model_size == "base"
    assert w._running is False


def test_whisper_stt_init_custom_model():
    w = WhisperSTT(model_size="small")
    assert w.model_size == "small"


def test_whisper_start_listening_is_idempotent():
    w = WhisperSTT()
    w._listen_loop = lambda cb: None
    w.start_listening(lambda t: None)
    first_thread = w._thread
    w.start_listening(lambda t: None)
    assert w._thread is first_thread
    w.stop_listening()


def test_whisper_stop_listening_sets_flag():
    w = WhisperSTT()
    w._listen_loop = lambda cb: None
    w.start_listening(lambda t: None)
    w.stop_listening()
    assert w._running is False


# ── WhisperSTT._record_and_transcribe ────────────────────────────────────────

def test_record_and_transcribe_returns_text(monkeypatch):
    from jarvis.voice.whisper_stt import WhisperSTT

    # Fake speech_recognition + numpy
    fake_audio = MagicMock()
    fake_audio.get_raw_data = MagicMock(return_value=b"\x00" * 100)

    fake_sr = MagicMock()
    fake_sr.Recognizer.return_value.listen.return_value = fake_audio

    fake_np = MagicMock()
    fake_np.frombuffer = MagicMock(return_value=MagicMock())
    fake_np.int16 = int
    fake_arr = MagicMock()
    fake_arr.astype.return_value = MagicMock()
    fake_np.frombuffer.return_value = fake_arr

    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"text": " hello world "}

    import sys
    with patch.dict(sys.modules, {"speech_recognition": fake_sr, "numpy": fake_np}):
        with patch("jarvis.voice.whisper_stt._load_model", return_value=fake_model):
            w = WhisperSTT()
            result = w._record_and_transcribe(timeout=5)
    assert result == "hello world"


def test_record_and_transcribe_returns_none_on_exception(monkeypatch, capsys):
    from jarvis.voice.whisper_stt import WhisperSTT

    import sys
    fake_sr = MagicMock()
    fake_sr.Recognizer.return_value.listen.side_effect = RuntimeError("no mic")
    fake_np = MagicMock()

    with patch.dict(sys.modules, {"speech_recognition": fake_sr, "numpy": fake_np}):
        with patch("jarvis.voice.whisper_stt._load_model", return_value=MagicMock()):
            w = WhisperSTT()
            result = w._record_and_transcribe(timeout=5)
    assert result is None
    assert "Transcribe error" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_transcribe_once_runs_in_executor(monkeypatch):
    from jarvis.voice.whisper_stt import WhisperSTT
    w = WhisperSTT()
    monkeypatch.setattr(w, "_record_and_transcribe", lambda t: "transcribed text")
    result = await w.transcribe_once(timeout=5)
    assert result == "transcribed text"


# ── WhisperSTT._listen_loop ───────────────────────────────────────────────────

def test_whisper_listen_loop_missing_deps(capsys):
    from jarvis.voice.whisper_stt import WhisperSTT
    w = WhisperSTT()
    w._running = True
    with patch.dict(sys.modules, {"speech_recognition": None}):
        w._listen_loop(lambda t: None)
    out = capsys.readouterr().out
    assert "Missing dependency" in out or "not installed" in out.lower()


def test_whisper_listen_loop_with_wake_word(monkeypatch):
    from jarvis.voice.whisper_stt import WhisperSTT
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "WAKE_WORD", "jarvis")

    w = WhisperSTT()
    transcripts = []

    fake_audio = MagicMock()
    fake_audio.get_raw_data = MagicMock(return_value=b"\x00" * 200)

    fake_sr = MagicMock()
    recogniser = MagicMock()
    recogniser.listen.return_value = fake_audio
    fake_sr.Recognizer.return_value = recogniser

    fake_mic = MagicMock()
    fake_mic.__enter__ = MagicMock(return_value=fake_mic)
    fake_mic.__exit__ = MagicMock(return_value=False)
    fake_sr.Microphone.return_value = fake_mic

    fake_np = MagicMock()
    arr = MagicMock()
    arr.astype.return_value = MagicMock()
    fake_np.frombuffer.return_value = arr
    fake_np.int16 = int
    fake_np.float32 = float

    fake_model = MagicMock()
    # First call returns wake word + command, second call stops the loop
    call_count = [0]

    def fake_transcribe(audio, language="en", fp16=False):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"text": "jarvis open the pod bay doors"}
        w._running = False
        return {"text": ""}

    fake_model.transcribe.side_effect = fake_transcribe

    with patch.dict(sys.modules, {"speech_recognition": fake_sr, "numpy": fake_np}):
        with patch("jarvis.voice.whisper_stt._load_model", return_value=fake_model):
            w._running = True
            w._listen_loop(lambda t: transcripts.append(t))

    assert len(transcripts) == 1
    assert "open the pod bay doors" in transcripts[0]


# ── TTSEngine piper path ──────────────────────────────────────────────────────

def test_tts_speak_piper_path(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", True)
    monkeypatch.setattr(cfg, "PIPER_BINARY", "/usr/bin/piper")
    monkeypatch.setattr(cfg, "PIPER_MODEL", "/models/en.onnx")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/piper" if name == "piper" else None)

    engine = TTSEngine(engine="piper")
    fake_play_wav = MagicMock()
    fake_proc = MagicMock()
    fake_proc.returncode = 0

    with patch("subprocess.run", return_value=fake_proc):
        with patch.object(engine, "_play_wav", fake_play_wav):
            engine.speak("piper test")

    fake_play_wav.assert_called_once()


def test_tts_speak_piper_fallback_on_failure(monkeypatch, capsys):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", True)
    monkeypatch.setattr(cfg, "PIPER_BINARY", "/usr/bin/piper")
    monkeypatch.setattr(cfg, "PIPER_MODEL", "/models/en.onnx")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/piper" if name == "piper" else None)

    engine = TTSEngine(engine="piper")
    fake_pyttsx3 = MagicMock()
    fake_proc = MagicMock()
    fake_proc.returncode = 1
    fake_proc.stderr = "piper error"

    with patch("subprocess.run", return_value=fake_proc):
        with patch.object(engine, "_speak_pyttsx3", fake_pyttsx3):
            engine.speak("fallback test")

    fake_pyttsx3.assert_called_once_with("fallback test")
    out = capsys.readouterr().out
    assert "Piper error" in out


def test_tts_speak_piper_exception_fallback(monkeypatch, capsys):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "VOICE_ENABLED", True)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/piper" if name == "piper" else None)

    engine = TTSEngine(engine="piper")
    fake_pyttsx3 = MagicMock()

    with patch("subprocess.run", side_effect=RuntimeError("piper crashed")):
        with patch.object(engine, "_speak_pyttsx3", fake_pyttsx3):
            engine.speak("exception test")

    fake_pyttsx3.assert_called_once_with("exception test")
    out = capsys.readouterr().out
    assert "Piper exception" in out


def test_tts_get_pyttsx3_caches(monkeypatch):
    from jarvis.config import cfg
    fake_engine = MagicMock()
    fake_engine.getProperty = MagicMock(return_value=[])
    fake_pyttsx3 = MagicMock()
    fake_pyttsx3.init = MagicMock(return_value=fake_engine)
    engine = TTSEngine(engine="pyttsx3")
    with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
        e1 = engine._get_pyttsx3()
        e2 = engine._get_pyttsx3()
    assert e1 is e2
    fake_pyttsx3.init.assert_called_once()
