"""JARVIS configuration — loaded from environment variables and .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    # ── Anthropic / Claude ─────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")
    MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "8192"))

    # ── Voice ──────────────────────────────────────────────────────────
    VOICE_ENABLED: bool = os.environ.get("VOICE_ENABLED", "true").lower() == "true"
    WAKE_WORD: str = os.environ.get("WAKE_WORD", "jarvis")
    TTS_ENGINE: str = os.environ.get("TTS_ENGINE", "pyttsx3")   # pyttsx3 | piper | elevenlabs
    STT_ENGINE: str = os.environ.get("STT_ENGINE", "whisper")   # whisper | google
    WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")  # tiny/base/small/medium/large
    PIPER_BINARY: str = os.environ.get("PIPER_BINARY", "piper")
    PIPER_MODEL: str = os.environ.get("PIPER_MODEL", "")
    ELEVENLABS_API_KEY: str = os.environ.get("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = os.environ.get("ELEVENLABS_VOICE_ID", "")

    # ── Paths ──────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MEMORY_DB: Path = DATA_DIR / "jarvis_memory.db"
    TOOLS_DIR: Path = BASE_DIR / "jarvis" / "tools" / "dynamic"
    LOGS_DIR: Path = DATA_DIR / "logs"

    # ── Daemon / API ───────────────────────────────────────────────────
    DAEMON_ENABLED: bool = os.environ.get("DAEMON_ENABLED", "true").lower() == "true"
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("API_PORT", "8000"))
    MONITOR_INTERVAL: int = int(os.environ.get("MONITOR_INTERVAL", "300"))  # seconds

    # ── Self-improvement ───────────────────────────────────────────────
    LEARNING_ENABLED: bool = os.environ.get("LEARNING_ENABLED", "true").lower() == "true"
    REFLECTION_INTERVAL_HOURS: int = int(os.environ.get("REFLECTION_INTERVAL_HOURS", "6"))
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
    AB_TESTING_ENABLED: bool = os.environ.get("AB_TESTING_ENABLED", "false").lower() == "true"

    # ── Personality ────────────────────────────────────────────────────
    DEFAULT_PROFILE: str = os.environ.get("DEFAULT_PROFILE", "default")

    # ── Integrations (optional) ────────────────────────────────────────
    GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")
    EMAIL_ADDRESS: str = os.environ.get("EMAIL_ADDRESS", "")
    EMAIL_PASSWORD: str = os.environ.get("EMAIL_PASSWORD", "")
    SMTP_HOST: str = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "465"))
    IMAP_HOST: str = os.environ.get("IMAP_HOST", "imap.gmail.com")
    TELEGRAM_TOKEN: str = os.environ.get("TELEGRAM_TOKEN", "")
    DISCORD_TOKEN: str = os.environ.get("DISCORD_TOKEN", "")
    OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3")

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)


cfg = Config()
cfg.ensure_dirs()
