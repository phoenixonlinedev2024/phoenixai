"""Text-to-speech engine — pyttsx3, Piper (offline neural), or ElevenLabs."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Literal

from jarvis.config import cfg


class TTSEngine:
    """Unified TTS: pyttsx3 (offline) | piper (neural offline) | elevenlabs (cloud)."""

    def __init__(self, engine: Literal["pyttsx3", "piper", "elevenlabs"] | None = None) -> None:
        self.engine_name = engine or cfg.TTS_ENGINE
        self._pyttsx3_engine = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def speak(self, text: str) -> None:
        if not cfg.VOICE_ENABLED or not text.strip():
            return
        if self.engine_name == "elevenlabs" and cfg.ELEVENLABS_API_KEY:
            self._speak_elevenlabs(text)
        elif self.engine_name == "piper" and self._piper_available():
            self._speak_piper(text)
        else:
            self._speak_pyttsx3(text)

    async def speak_async(self, text: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.speak, text)

    # ------------------------------------------------------------------ #
    # pyttsx3 (offline, any platform)
    # ------------------------------------------------------------------ #

    def _get_pyttsx3(self):
        if self._pyttsx3_engine is None:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", 175)
            self._pyttsx3_engine.setProperty("volume", 0.95)
            voices = self._pyttsx3_engine.getProperty("voices")
            for voice in voices:
                name = voice.name.lower()
                if "english" in name and ("uk" in name or "british" in name or "male" in name):
                    self._pyttsx3_engine.setProperty("voice", voice.id)
                    break
        return self._pyttsx3_engine

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            with self._lock:
                engine = self._get_pyttsx3()
                engine.say(text)
                engine.runAndWait()
        except Exception as exc:
            print(f"[JARVIS TTS] pyttsx3 error: {exc}")

    # ------------------------------------------------------------------ #
    # Piper (high-quality offline neural TTS)
    # Requires: piper binary + model file
    # Install:  https://github.com/rhasspy/piper/releases
    # ------------------------------------------------------------------ #

    def _piper_available(self) -> bool:
        return shutil.which("piper") is not None or Path(cfg.PIPER_BINARY).exists()

    def _speak_piper(self, text: str) -> None:
        try:
            binary = cfg.PIPER_BINARY if Path(cfg.PIPER_BINARY).exists() else "piper"
            model = cfg.PIPER_MODEL
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            cmd = [binary, "--model", model, "--output_file", wav_path]
            proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                self._play_wav(wav_path)
            else:
                print(f"[JARVIS TTS] Piper error: {proc.stderr}. Falling back to pyttsx3.")
                self._speak_pyttsx3(text)
        except Exception as exc:
            print(f"[JARVIS TTS] Piper exception: {exc}. Falling back to pyttsx3.")
            self._speak_pyttsx3(text)

    def _play_wav(self, path: str) -> None:
        try:
            import wave
            import pyaudio
            with wave.open(path, "rb") as wf:
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )
                while chunk := wf.readframes(1024):
                    stream.write(chunk)
                stream.stop_stream()
                stream.close()
                p.terminate()
        except Exception:
            # Fallback: use system audio player
            for player in ("aplay", "afplay", "paplay"):
                if shutil.which(player):
                    subprocess.run([player, path], capture_output=True)
                    break

    # ------------------------------------------------------------------ #
    # ElevenLabs (cloud, premium quality)
    # ------------------------------------------------------------------ #

    def _speak_elevenlabs(self, text: str) -> None:
        try:
            from elevenlabs import ElevenLabs, play
            client = ElevenLabs(api_key=cfg.ELEVENLABS_API_KEY)
            voice_id = cfg.ELEVENLABS_VOICE_ID or "21m00Tcm4TlvDq8ikWAM"
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2",
            )
            play(audio)
        except Exception as exc:
            print(f"[JARVIS TTS] ElevenLabs error: {exc}. Falling back to pyttsx3.")
            self._speak_pyttsx3(text)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def set_voice_by_name(self, name_fragment: str) -> bool:
        try:
            engine = self._get_pyttsx3()
            for voice in engine.getProperty("voices"):
                if name_fragment.lower() in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    return True
        except Exception:
            pass
        return False

    def list_voices(self) -> list[str]:
        try:
            return [v.name for v in self._get_pyttsx3().getProperty("voices")]
        except Exception:
            return []
