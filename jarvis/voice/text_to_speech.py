"""Text-to-speech engine — JARVIS speaks with either pyttsx3 or ElevenLabs."""

from __future__ import annotations

import asyncio
import io
import os
import threading
from typing import Literal

from jarvis.config import cfg


class TTSEngine:
    """Unified TTS interface supporting pyttsx3 (offline) and ElevenLabs (cloud)."""

    def __init__(self, engine: Literal["pyttsx3", "elevenlabs"] | None = None) -> None:
        self.engine_name = engine or cfg.TTS_ENGINE
        self._pyttsx3_engine = None
        self._lock = threading.Lock()

    def _get_pyttsx3(self):
        if self._pyttsx3_engine is None:
            try:
                import pyttsx3
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty("rate", 175)
                self._pyttsx3_engine.setProperty("volume", 0.95)
                # Try to select a British male voice (closest to JARVIS)
                voices = self._pyttsx3_engine.getProperty("voices")
                for voice in voices:
                    name = voice.name.lower()
                    if "english" in name and ("uk" in name or "british" in name or "male" in name):
                        self._pyttsx3_engine.setProperty("voice", voice.id)
                        break
            except ImportError:
                raise RuntimeError("pyttsx3 not installed. Run: pip install pyttsx3")
        return self._pyttsx3_engine

    def speak(self, text: str) -> None:
        """Speak text synchronously."""
        if not cfg.VOICE_ENABLED:
            return
        if self.engine_name == "elevenlabs" and cfg.ELEVENLABS_API_KEY:
            self._speak_elevenlabs(text)
        else:
            self._speak_pyttsx3(text)

    async def speak_async(self, text: str) -> None:
        """Speak text without blocking the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.speak, text)

    def _speak_pyttsx3(self, text: str) -> None:
        with self._lock:
            engine = self._get_pyttsx3()
            engine.say(text)
            engine.runAndWait()

    def _speak_elevenlabs(self, text: str) -> None:
        try:
            from elevenlabs import ElevenLabs, play
            client = ElevenLabs(api_key=cfg.ELEVENLABS_API_KEY)
            voice_id = cfg.ELEVENLABS_VOICE_ID or "21m00Tcm4TlvDq8ikWAM"  # Rachel (neutral)
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2",
            )
            play(audio)
        except Exception as exc:
            print(f"[JARVIS Voice] ElevenLabs error: {exc}. Falling back to pyttsx3.")
            self._speak_pyttsx3(text)

    def set_voice_by_name(self, name_fragment: str) -> bool:
        """Select a pyttsx3 voice by partial name match."""
        try:
            engine = self._get_pyttsx3()
            voices = engine.getProperty("voices")
            for voice in voices:
                if name_fragment.lower() in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    return True
        except Exception:
            pass
        return False

    def list_voices(self) -> list[str]:
        try:
            engine = self._get_pyttsx3()
            return [v.name for v in engine.getProperty("voices")]
        except Exception:
            return []
