# CLAUDE.md — JARVIS Development Guide

## Architecture Overview

```
jarvis/
├── core.py              ← Jarvis agent: chat(), stream_chat(), _run_agent_loop()
├── personality.py       ← System prompts, 5 personality profiles
├── config.py            ← All settings loaded from .env via python-dotenv
├── daemon.py            ← FastAPI app (create_app), REST + WebSocket, scheduler, bots
├── main.py              ← CLI entry point (typer): chat, daemon, voice, benchmark, keys
│
├── memory/
│   ├── store.py         ← SQLite: facts, lessons, history, gaps, schedules
│   ├── learning.py      ← Post-session reflection engine
│   └── semantic.py      ← ChromaDB vector store for semantic recall
│
├── tools/
│   ├── registry.py      ← Tool dataclass + ToolRegistry + build_registry()
│   ├── creator.py       ← Dynamic tool synthesis via Claude API
│   └── *.py             ← 35+ built-in tools (web, files, shell, code, git, etc.)
│
├── acp/                 ← Async pub/sub message bus (MessageBus)
├── observability/       ← Metrics: counters, histograms, Prometheus export
├── scheduling/          ← Priority TaskQueue + NLScheduler (NL → cron)
├── security/            ← API key auth, rate limiting, RBAC middleware
├── self_improve/        ← BenchmarkRunner, CapabilityEvolver, SelfImproveEngine
│
├── agents/              ← SubAgent pool + RPC bus
├── bots/                ← Telegram, Discord, Slack, IRC, Matrix, Mattermost, Signal, WhatsApp
├── providers/           ← Multi-LLM router: Anthropic → OpenRouter → Ollama → OpenAI-compat
├── sandbox/             ← Execution backends: Local, Docker, SSH, Modal, Singularity
├── skills/              ← Composable skill registry (hot-reloaded)
├── plugins/             ← Drop-in plugin loader (watches jarvis/plugins/*.py)
├── research/            ← Trajectory collection + ShareGPT/Atropos dataset export
└── voice/               ← Whisper STT + pyttsx3/ElevenLabs TTS
```

## Request Flow

```
User → WebSocket /ws  →  stream_chat()  →  _stream_agent_loop()  →  Claude API (streaming)
                                                  ↓ tool calls?
                                          _execute_tools_parallel()
                                                  ↓
                                          repeat up to 12 iterations

User → POST /chat     →  chat()          →  _run_agent_loop()     →  Claude API (non-streaming)
```

## Running Locally

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Set ANTHROPIC_API_KEY at minimum

# Chat (terminal)
jarvis chat

# Start daemon (API + WebSocket + bots)
jarvis daemon
# → http://localhost:8000     Web UI
# → http://localhost:8000/docs  API docs

# Voice mode
jarvis voice

# Run benchmarks
jarvis benchmark

# Manage API keys
jarvis keys --create --name myapp --role user
jarvis keys
jarvis keys --revoke <id>
```

## Running Tests

```bash
pip install -r requirements-test.txt
pip install --no-deps -e .
pytest tests/ -v
```

**129 tests** covering: memory, tools/registry, security, scheduling, ACP, observability,
self-improve, transform tools, core agent, and daemon REST endpoints.

## Adding a New Tool

1. Add a function to an appropriate file in `jarvis/tools/`
2. Register it in `build_registry()` in `jarvis/tools/registry.py`

```python
registry.register(Tool(
    name="my_tool",
    description="What it does.",
    input_schema={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
    fn=my_function,
    category="general",
))
```

## Adding a New API Endpoint

Add the route inside `create_app()` in `daemon.py`. Pydantic request/response models
**must be defined at module level** (not inside `create_app`) — `from __future__ import annotations`
causes FastAPI to fail resolving locally-scoped models.

## Key Design Decisions

- **Parallel tool execution**: `_execute_tools_parallel()` uses `asyncio.gather` — all tool calls in a single Claude turn run concurrently.
- **12-iteration cap**: The agent loop runs at most 12 tool-call rounds per message to prevent runaway costs.
- **Dynamic tools**: If Claude requests a tool that doesn't exist, `synthesise_tool()` generates and registers it on the fly.
- **Security off by default**: `SECURITY_ENABLED=false` in `.env.example`. Set to `true` in production.
- **Pydantic models at module level**: Required due to `from __future__ import annotations` (PEP 563 deferred evaluation).

## Environment Variables

See `.env.example` for the full reference (120+ settings). Minimum required:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## CI

GitHub Actions runs on every push to `main` and `claude/**` branches:
- **Tests** (Python 3.11 + 3.12): `pytest tests/ -v`
- **Lint**: `ruff check jarvis/ --select E,F,W --ignore E501`
