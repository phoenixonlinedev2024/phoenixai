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

    # ── Multi-provider ─────────────────────────────────────────────────
    PROVIDER_ORDER: list = os.environ.get("PROVIDER_ORDER", "anthropic").split(",")
    OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free")
    OPENAI_COMPAT_BASE_URL: str = os.environ.get("OPENAI_COMPAT_BASE_URL", "")
    OPENAI_COMPAT_API_KEY: str = os.environ.get("OPENAI_COMPAT_API_KEY", "")
    OPENAI_COMPAT_MODEL: str = os.environ.get("OPENAI_COMPAT_MODEL", "gpt-3.5-turbo")
    OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3")

    # ── Voice ──────────────────────────────────────────────────────────
    VOICE_ENABLED: bool = os.environ.get("VOICE_ENABLED", "true").lower() == "true"
    WAKE_WORD: str = os.environ.get("WAKE_WORD", "jarvis")
    TTS_ENGINE: str = os.environ.get("TTS_ENGINE", "pyttsx3")
    STT_ENGINE: str = os.environ.get("STT_ENGINE", "whisper")
    WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")
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
    MONITOR_INTERVAL: int = int(os.environ.get("MONITOR_INTERVAL", "300"))
    # Comma-separated list of origins; empty list means CORS is disabled when
    # SECURITY_ENABLED=true, or wildcarded to "*" when SECURITY_ENABLED=false.
    CORS_ALLOWED_ORIGINS: list = [
        o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()
    ]

    # ── Self-improvement ───────────────────────────────────────────────
    LEARNING_ENABLED: bool = os.environ.get("LEARNING_ENABLED", "true").lower() == "true"
    REFLECTION_INTERVAL_HOURS: int = int(os.environ.get("REFLECTION_INTERVAL_HOURS", "6"))
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
    AB_TESTING_ENABLED: bool = os.environ.get("AB_TESTING_ENABLED", "false").lower() == "true"
    TRAJECTORY_COLLECTION: bool = os.environ.get("TRAJECTORY_COLLECTION", "true").lower() == "true"

    # ── Personality ────────────────────────────────────────────────────
    DEFAULT_PROFILE: str = os.environ.get("DEFAULT_PROFILE", "default")

    # ── Sandbox ────────────────────────────────────────────────────────
    SANDBOX_BACKEND: str = os.environ.get("SANDBOX_BACKEND", "local")
    DOCKER_IMAGE: str = os.environ.get("DOCKER_IMAGE", "python:3.11-slim")
    SSH_HOST: str = os.environ.get("SSH_HOST", "")
    SSH_PORT: int = int(os.environ.get("SSH_PORT", "22"))
    SSH_USER: str = os.environ.get("SSH_USER", "")
    SSH_KEY_PATH: str = os.environ.get("SSH_KEY_PATH", "")
    SSH_PASSWORD: str = os.environ.get("SSH_PASSWORD", "")

    # ── Platform bots ──────────────────────────────────────────────────
    TELEGRAM_TOKEN: str = os.environ.get("TELEGRAM_TOKEN", "")
    DISCORD_TOKEN: str = os.environ.get("DISCORD_TOKEN", "")
    SLACK_BOT_TOKEN: str = os.environ.get("SLACK_BOT_TOKEN", "")
    SLACK_APP_TOKEN: str = os.environ.get("SLACK_APP_TOKEN", "")
    # WhatsApp — Twilio
    TWILIO_ACCOUNT_SID: str = os.environ.get("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.environ.get("TWILIO_AUTH_TOKEN", "")
    TWILIO_WHATSAPP_FROM: str = os.environ.get("TWILIO_WHATSAPP_FROM", "")
    # WhatsApp — Meta Cloud API
    WHATSAPP_ACCESS_TOKEN: str = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    WHATSAPP_PHONE_ID: str = os.environ.get("WHATSAPP_PHONE_ID", "")
    # Must be explicitly set; no default — predictable defaults let attackers
    # complete the WhatsApp webhook handshake.
    WHATSAPP_VERIFY_TOKEN: str = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
    # Signal
    SIGNAL_PHONE_NUMBER: str = os.environ.get("SIGNAL_PHONE_NUMBER", "")
    SIGNAL_CLI_PATH: str = os.environ.get("SIGNAL_CLI_PATH", "signal-cli")
    # IRC
    IRC_SERVER: str = os.environ.get("IRC_SERVER", "")
    IRC_PORT: int = int(os.environ.get("IRC_PORT", "6667"))
    IRC_NICK: str = os.environ.get("IRC_NICK", "JARVIS")
    IRC_CHANNELS: str = os.environ.get("IRC_CHANNELS", "#jarvis")
    # Matrix
    MATRIX_HOMESERVER: str = os.environ.get("MATRIX_HOMESERVER", "")
    MATRIX_USER_ID: str = os.environ.get("MATRIX_USER_ID", "")
    MATRIX_ACCESS_TOKEN: str = os.environ.get("MATRIX_ACCESS_TOKEN", "")
    # Mattermost
    MATTERMOST_URL: str = os.environ.get("MATTERMOST_URL", "")
    MATTERMOST_TOKEN: str = os.environ.get("MATTERMOST_TOKEN", "")
    MATTERMOST_BOT_USER_ID: str = os.environ.get("MATTERMOST_BOT_USER_ID", "")
    MATTERMOST_BOT_NAME: str = os.environ.get("MATTERMOST_BOT_NAME", "jarvis")

    # ── Object storage (S3 / R2 / MinIO) ──────────────────────────────
    S3_ACCESS_KEY: str = os.environ.get("S3_ACCESS_KEY", "")
    S3_SECRET_KEY: str = os.environ.get("S3_SECRET_KEY", "")
    S3_REGION: str = os.environ.get("S3_REGION", "us-east-1")
    S3_ENDPOINT_URL: str = os.environ.get("S3_ENDPOINT_URL", "")

    # ── Security ──────────────────────────────────────────────────────
    SECURITY_ENABLED: bool = os.environ.get("SECURITY_ENABLED", "false").lower() == "true"
    RATE_LIMIT: int = int(os.environ.get("RATE_LIMIT", "120"))
    SELF_IMPROVE_INTERVAL_HOURS: float = float(os.environ.get("SELF_IMPROVE_INTERVAL_HOURS", "12"))
    # Risky-feature kill-switches. Both default ON for backward compatibility,
    # but operators in hardened environments should set them to "false".
    DYNAMIC_TOOLS_ENABLED: bool = os.environ.get("DYNAMIC_TOOLS_ENABLED", "true").lower() == "true"
    SHELL_TOOL_ENABLED: bool = os.environ.get("SHELL_TOOL_ENABLED", "true").lower() == "true"

    # ── External integrations ──────────────────────────────────────────
    GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")
    EMAIL_ADDRESS: str = os.environ.get("EMAIL_ADDRESS", "")
    EMAIL_PASSWORD: str = os.environ.get("EMAIL_PASSWORD", "")
    SMTP_HOST: str = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "465"))
    IMAP_HOST: str = os.environ.get("IMAP_HOST", "imap.gmail.com")

    # ── Media / AI ─────────────────────────────────────────────────────
    HF_API_TOKEN: str = os.environ.get("HF_API_TOKEN", "")

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)


cfg = Config()
cfg.ensure_dirs()
