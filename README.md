# JARVIS — OpenClaw AI

**J**ust **A** **R**ather **V**ery **I**ntelligent **S**ystem — a universal, self-improving, always-on AI agent built for OpenClaw, powered by Claude.

---

## Capabilities

| Capability | Description |
|---|---|
| **Universal Action** | Web search/fetch, browser automation (Playwright), file I/O, shell, code execution in any language, REST/GraphQL, email, GitHub, system monitoring |
| **Dynamic Creation** | If a tool or capability doesn't exist, JARVIS synthesises and registers it on the fly via Claude |
| **Semantic Memory** | ChromaDB vector store for meaning-based recall + SQLite for facts, lessons, history, gaps, and schedules |
| **Self-Learning** | Post-session reflection extracts lessons and facts fed back into every future prompt. Confidence scoring + A/B testing |
| **Voice** | Whisper STT (offline, free, accurate) + Piper neural TTS / pyttsx3 / ElevenLabs. Wake-word activated |
| **Always On** | FastAPI REST + WebSocket streaming + APScheduler + proactive URL/file monitor + Telegram + Discord bots |
| **Personality Profiles** | Switch tone at runtime: default / professional / casual / terse / verbose |
| **Plugin System** | Drop a `.py` into `jarvis/plugins/` — hot-reloaded without restart |
| **Web UI** | Built-in browser chat interface served at `/` |

---

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Chat

```bash
jarvis chat                        # Interactive terminal chat
jarvis chat "What's the weather?"  # One-shot message
jarvis chat --voice                # Chat with voice output
```

### 4. Voice Mode

```bash
jarvis voice    # Full voice interaction — say "JARVIS" to wake
```

### 5. Run as a 24/7 Daemon

```bash
jarvis daemon
# API available at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

---

## CLI Commands

```
jarvis chat      Interactive or one-shot conversation
jarvis daemon    Start the 24/7 REST API + voice + scheduler
jarvis voice     Voice-only mode
jarvis status    Show system status and memory summary
jarvis tools     List all registered tools (including dynamic ones)
jarvis memory    Show stored facts and lessons learned
```

---

## REST API (Daemon Mode)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/status` | Full status |
| POST | `/chat` | Send a message |
| POST | `/session/end` | End session and trigger reflection |
| POST | `/session/new` | Start a fresh session |
| GET | `/memory/facts` | All stored facts |
| GET | `/memory/lessons` | All lessons learned |
| GET | `/tools` | All registered tools |
| POST | `/schedule` | Add a scheduled task |
| GET | `/schedule` | List scheduled tasks |

### Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Search the web for the latest AI news and summarise it."}'
```

---

## Architecture

```
jarvis/
├── core.py           ← JARVIS agent (agentic tool-use loop)
├── personality.py    ← System prompt, tone, identity
├── config.py         ← Environment configuration
├── daemon.py         ← FastAPI + APScheduler + VoiceLoop
├── main.py           ← CLI entry point (typer)
├── tools/
│   ├── registry.py   ← Tool registry + all built-in tools
│   └── creator.py    ← Dynamic tool synthesis via Claude
├── memory/
│   ├── store.py      ← SQLite memory (facts, lessons, history, schedule)
│   └── learning.py   ← Self-reflection and improvement engine
└── voice/
    ├── text_to_speech.py   ← pyttsx3 + ElevenLabs TTS
    └── speech_to_text.py   ← Wake-word + Google STT
```

---

## Self-Learning Loop

1. Every conversation is stored in SQLite.
2. Every N hours (configurable), JARVIS runs a reflection pass using Claude.
3. Claude extracts: lessons learned, new facts, capability gaps.
4. Lessons and facts are injected into every future conversation as memory context.
5. Capability gaps trigger automatic tool synthesis on the next relevant request.

---

## Adding Custom Scheduled Tasks

```bash
curl -X POST http://localhost:8000/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "name": "daily_news",
    "cron": "0 8 * * *",
    "prompt": "Search for today'\''s top AI news and store a summary."
  }'
```

---

## Environment Variables

See [`.env.example`](.env.example) for full configuration reference.

---

## License

GPLv3 — see [LICENSE](LICENSE).