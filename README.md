# JARVIS — OpenClaw AI v4

**J**ust **A** **R**ather **V**ery **I**ntelligent **S**ystem — a universal, self-improving, always-on AI agent built for OpenClaw, powered by Claude.

---

## Capabilities

| Capability | Description |
|---|---|
| **Universal Action** | Web search/fetch, browser automation (Playwright), file I/O, shell, code execution, REST/GraphQL, email, GitHub, system monitoring |
| **Dynamic Creation** | If a tool doesn't exist, JARVIS synthesises and registers it on the fly via Claude |
| **Semantic Memory** | ChromaDB vector store + SQLite for facts, lessons, history, gaps, and schedules |
| **Self-Improving** | Automated benchmark suite + capability gap detection + auto-synthesis. Background self-improve loop every N hours |
| **Observability** | Prometheus-style metrics (counters, histograms, p95/p99), `/metrics` endpoint, JSONL log rotation |
| **Security** | API key auth (HMAC-SHA256), sliding-window rate limiting, RBAC roles (admin/user/readonly) |
| **ACP Message Bus** | Async pub/sub bus for internal event streaming; wildcard subscriptions; history replay |
| **NL Scheduler** | Natural-language cron: "every morning", "every 15 minutes", "every weekday" → cron expressions |
| **Voice** | Whisper STT (offline) + Piper TTS / pyttsx3 / ElevenLabs. Wake-word activated |
| **Multi-Platform Bots** | Telegram, Discord, Slack, Signal, WhatsApp, IRC, Matrix, Mattermost |
| **Personality Profiles** | Switch tone at runtime: default / professional / casual / terse / verbose |
| **Plugin System** | Drop a `.py` into `jarvis/plugins/` — hot-reloaded without restart |
| **Web UI** | Built-in browser chat + v4 dashboards (Metrics, API Keys, ACP Bus, Self-Improve, Scheduler) |

---

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY
```

### 3. Chat

```bash
jarvis chat                        # Interactive terminal chat
jarvis chat "What's the weather?"  # One-shot message
jarvis chat --voice                # Chat with voice output
```

### 4. Run as a 24/7 Daemon

```bash
jarvis daemon
# Web UI:  http://localhost:8000
# API docs: http://localhost:8000/docs
```

---

## CLI Commands

```
jarvis chat          Interactive or one-shot conversation
jarvis daemon        Start the 24/7 REST API + voice + scheduler + bots
jarvis voice         Voice-only mode (wake-word: JARVIS)
jarvis status        Show system status and memory summary
jarvis tools         List all registered tools (including dynamic)
jarvis memory        Show stored facts and lessons learned
jarvis benchmark     Run the capability benchmark suite
jarvis keys          Manage API keys (--create / --revoke / list)
```

---

## REST API (Daemon Mode)

### Core

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/status` | Full system status |
| POST | `/chat` | Send a message (REST fallback) |
| WS | `/ws` | WebSocket streaming chat |
| POST | `/session/new` | Start a fresh session |
| GET | `/memory/facts` | All stored facts |
| GET | `/memory/lessons` | All lessons learned |
| GET | `/tools` | All registered tools |

### Observability

| Method | Endpoint | Description |
|---|---|---|
| GET | `/metrics` | Prometheus text format |
| GET | `/metrics/json` | JSON snapshot (counters + histograms) |

### Security

| Method | Endpoint | Description |
|---|---|---|
| GET | `/security/keys` | List API keys |
| POST | `/security/keys` | Create API key `{name, role}` |
| DELETE | `/security/keys/{id}` | Revoke API key |

### ACP Bus

| Method | Endpoint | Description |
|---|---|---|
| POST | `/acp/publish` | Publish message `{topic, payload}` |
| GET | `/acp/history` | Recent message history |
| GET | `/acp/stats` | Bus statistics |

### Self-Improve

| Method | Endpoint | Description |
|---|---|---|
| POST | `/self-improve/run` | Run benchmark suite |
| GET | `/self-improve/benchmarks` | Benchmark history + trend |
| GET | `/self-improve/capabilities` | Capability map + gaps |

### NL Scheduler

| Method | Endpoint | Description |
|---|---|---|
| GET | `/schedule` | List scheduled jobs |
| POST | `/schedule/nl` | Add job via natural language `{phrase, prompt}` |
| GET | `/schedule/nl/parse` | Parse phrase to cron (dry run) |

### Example

```bash
# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Search for the latest AI news and summarise it."}'

# Add a scheduled task in plain English
curl -X POST http://localhost:8000/schedule/nl \
  -H "Content-Type: application/json" \
  -d '{"phrase": "every morning", "name": "daily_brief", "prompt": "Summarise AI news."}'

# Get Prometheus metrics
curl http://localhost:8000/metrics
```

---

## Architecture

```
jarvis/
├── core.py               ← JARVIS agent (agentic tool-use loop)
├── personality.py        ← System prompt, tone profiles, identity
├── config.py             ← Environment configuration (100+ settings)
├── daemon.py             ← FastAPI + WebSocket + all API endpoints
├── main.py               ← CLI entry point (typer)
├── acp/                  ← Async pub/sub message bus
├── observability/        ← Metrics (counters, histograms, Prometheus)
├── scheduling/           ← Priority task queue + NL cron parser
├── security/             ← API key auth + rate limiter + RBAC
├── self_improve/         ← Benchmark runner + capability evolver
├── tools/
│   ├── registry.py       ← Tool registry + 50+ built-in tools
│   ├── creator.py        ← Dynamic tool synthesis via Claude
│   └── transform_tools.py← JSON/YAML/regex/diff/jq helpers
├── memory/
│   ├── store.py          ← SQLite memory (facts, lessons, history)
│   └── learning.py       ← Self-reflection engine
├── agents/               ← SubAgent pool + RPC bus
├── skills/               ← Skill registry
├── bots/                 ← Telegram, Discord, Slack, IRC, Matrix, Mattermost
├── voice/
│   ├── text_to_speech.py ← pyttsx3 + ElevenLabs TTS
│   └── speech_to_text.py ← Whisper STT + wake-word
└── web_ui/
    └── index.html        ← Browser chat + v4 dashboards
```

---

## Self-Improvement Loop

1. Every conversation is stored in SQLite.
2. Every N hours (default 12, configurable via `SELF_IMPROVE_INTERVAL_HOURS`), JARVIS runs:
   - Built-in benchmark suite (math, reasoning, code, factual, memory, tool use).
   - Capability gap analysis — compares known capabilities vs. benchmark results.
   - Auto-synthesis of new tools to fill detected gaps.
3. Lessons and facts from reflection are injected into every future conversation.
4. Benchmark trends tracked over time — accessible at `/self-improve/benchmarks`.

---

## Security

API key authentication is enabled via `SECURITY_ENABLED=true` in `.env`.

```bash
# Create a key
jarvis keys --create --name my-app --role user

# Revoke a key
jarvis keys --revoke <key-id>

# Use in requests
curl -H "X-API-Key: <key>" http://localhost:8000/chat ...
```

Rate limiting defaults to 120 requests/minute per key (`RATE_LIMIT=120`).

---

## Environment Variables

See [`.env.example`](.env.example) for the full configuration reference (120+ settings covering LLM providers, platform bots, object storage, voice, security, and more).

---

## Testing

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

90 tests covering memory, tools, security, scheduling, ACP, observability, self-improve, and transform tools.

---

## License

GPLv3 — see [LICENSE](LICENSE).
