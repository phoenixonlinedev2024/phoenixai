"""Speech-to-text engine — listens for JARVIS wake word and transcribes speech."""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Callable

from jarvis.config import cfg


class STTEngine:
    """Wake-word activated speech recognition using SpeechRecognition library."""

    def __init__(self) -> None:
        self._running = False
        self._thread: threading.Thread | None = None
        self._audio_queue: queue.Queue[str] = queue.Queue()

    def start_listening(self, on_transcript: Callable[[str], None]) -> None:
        """Start background wake-word listening loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            args=(on_transcript,),
            daemon=True,
            name="jarvis-stt",
        )
        self._thread.start()

    def stop_listening(self) -> None:
        self._running = False

    def _listen_loop(self, on_transcript: Callable[[str], None]) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            print("[JARVIS Voice] SpeechRecognition not installed. Voice input disabled.")
            return

        recogniser = sr.Recognizer()
        recogniser.energy_threshold = 300
        recogniser.dynamic_energy_threshold = True
        recogniser.pause_threshold = 0.8

        with sr.Microphone() as source:
            print(f"[JARVIS] Listening for wake word: '{cfg.WAKE_WORD}'")
            recogniser.adjust_for_ambient_noise(source, duration=1)

            while self._running:
                try:
                    audio = recogniser.listen(source, timeout=5, phrase_time_limit=15)
                    text = recogniser.recognize_google(audio).lower()

                    if cfg.WAKE_WORD.lower() in text:
                        # Strip the wake word and pass the rest
                        command = text.replace(cfg.WAKE_WORD.lower(), "").strip(" ,.")
                        if command:
                            on_transcript(command)
                        else:
                            # Prompt for follow-up
                            on_transcript("_wake_only_")

                except Exception:
                    pass  # Timeout or recognition error — keep looping

    async def listen_once(self, timeout: int = 10) -> str | None:
        """Record a single phrase and return its transcription (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._record_phrase, timeout)

    def _record_phrase(self, timeout: int) -> str | None:
        try:
            import speech_recognition as sr
            recogniser = sr.Recognizer()
            with sr.Microphone() as source:
                recogniser.adjust_for_ambient_noise(source, duration=0.5)
                audio = recogniser.listen(source, timeout=timeout, phrase_time_limit=20)
                return recogniser.recognize_google(audio)
        except Exception as exc:
            print(f"[JARVIS Voice] STT error: {exc}")
            return None
