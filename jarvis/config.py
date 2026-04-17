"""JARVIS configuration — loaded from environment variables and .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    # Anthropic / Claude
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")
    MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "8192"))

    # ElevenLabs TTS (optional)
    ELEVENLABS_API_KEY: str = os.environ.get("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = os.environ.get("ELEVENLABS_VOICE_ID", "")

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MEMORY_DB: Path = DATA_DIR / "jarvis_memory.db"
    TOOLS_DIR: Path = BASE_DIR / "jarvis" / "tools" / "dynamic"
    LOGS_DIR: Path = DATA_DIR / "logs"

    # Voice
    VOICE_ENABLED: bool = os.environ.get("VOICE_ENABLED", "true").lower() == "true"
    WAKE_WORD: str = os.environ.get("WAKE_WORD", "jarvis")
    TTS_ENGINE: str = os.environ.get("TTS_ENGINE", "pyttsx3")  # pyttsx3 | elevenlabs

    # Daemon / Scheduler
    DAEMON_ENABLED: bool = os.environ.get("DAEMON_ENABLED", "true").lower() == "true"
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("API_PORT", "8000"))

    # Self-improvement
    LEARNING_ENABLED: bool = os.environ.get("LEARNING_ENABLED", "true").lower() == "true"
    REFLECTION_INTERVAL_HOURS: int = int(os.environ.get("REFLECTION_INTERVAL_HOURS", "6"))

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)


cfg = Config()
cfg.ensure_dirs()
