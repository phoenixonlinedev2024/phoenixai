"""Whisper STT — offline, free, far more accurate than Google STT."""

from __future__ import annotations

import asyncio
import threading
from typing import Callable

from jarvis.config import cfg

_MODEL_CACHE: dict = {}


def _load_model(size: str = "base"):
    if size not in _MODEL_CACHE:
        try:
            import whisper
            print(f"[JARVIS Whisper] Loading model '{size}'...")
            _MODEL_CACHE[size] = whisper.load_model(size)
            print(f"[JARVIS Whisper] Model '{size}' ready.")
        except ImportError:
            raise RuntimeError("openai-whisper not installed. Run: pip install openai-whisper")
    return _MODEL_CACHE[size]


class WhisperSTT:
    """Wake-word activated Whisper speech recognition."""

    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._running = False
        self._thread: threading.Thread | None = None

    def start_listening(self, on_transcript: Callable[[str], None]) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            args=(on_transcript,),
            daemon=True,
            name="jarvis-whisper",
        )
        self._thread.start()

    def stop_listening(self) -> None:
        self._running = False

    def _listen_loop(self, on_transcript: Callable[[str], None]) -> None:
        try:
            import speech_recognition as sr
            import numpy as np  # noqa: F401 — needed for whisper audio processing
        except ImportError as e:
            print(f"[JARVIS Whisper] Missing dependency: {e}")
            return

        model = _load_model(self.model_size)
        recogniser = sr.Recognizer()
        recogniser.energy_threshold = 300
        recogniser.dynamic_energy_threshold = True
        recogniser.pause_threshold = 0.8

        print(f"[JARVIS Whisper] Listening for wake word: '{cfg.WAKE_WORD}'")

        with sr.Microphone(sample_rate=16000) as source:
            recogniser.adjust_for_ambient_noise(source, duration=1)
            while self._running:
                try:
                    audio = recogniser.listen(source, timeout=5, phrase_time_limit=20)
                    # Convert to numpy float32 for Whisper
                    raw = np.frombuffer(audio.get_raw_data(convert_rate=16000, convert_width=2), dtype=np.int16)
                    audio_float = raw.astype(np.float32) / 32768.0
                    result = model.transcribe(audio_float, language="en", fp16=False)
                    text = result["text"].strip().lower()
                    if not text:
                        continue
                    if cfg.WAKE_WORD.lower() in text:
                        command = text.replace(cfg.WAKE_WORD.lower(), "").strip(" ,.")
                        on_transcript(command if command else "_wake_only_")
                except Exception:
                    pass

    async def transcribe_once(self, timeout: int = 10) -> str | None:
        """Record a single phrase and transcribe with Whisper."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._record_and_transcribe, timeout)

    def _record_and_transcribe(self, timeout: int) -> str | None:
        try:
            import speech_recognition as sr
            import numpy as np

            model = _load_model(self.model_size)
            recogniser = sr.Recognizer()
            with sr.Microphone(sample_rate=16000) as source:
                recogniser.adjust_for_ambient_noise(source, duration=0.5)
                audio = recogniser.listen(source, timeout=timeout, phrase_time_limit=25)
                raw = np.frombuffer(audio.get_raw_data(convert_rate=16000, convert_width=2), dtype=np.int16)
                audio_float = raw.astype(np.float32) / 32768.0
                result = model.transcribe(audio_float, language="en", fp16=False)
                return result["text"].strip()
        except Exception as exc:
            print(f"[JARVIS Whisper] Transcribe error: {exc}")
            return None
