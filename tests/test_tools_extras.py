"""Tests for jarvis.tools.nlp_cron and jarvis.tools.creator."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.tools.nlp_cron import _nl_to_cron, _validate_cron  # noqa: E402
from jarvis.tools.creator import _make_fn, synthesise_tool  # noqa: E402


# ── _nl_to_cron ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,expected", [
    ("every minute", "* * * * *"),
    ("every 5 minutes", "*/5 * * * *"),
    ("every hour", "0 * * * *"),
    ("every 6 hours", "0 */6 * * *"),
    ("every morning", "0 8 * * *"),
    ("every evening", "0 18 * * *"),
    ("every night", "0 22 * * *"),
    ("every monday", "0 9 * * 1"),
    ("every friday", "0 9 * * 5"),
    ("every weekend", "0 9 * * 6,0"),
    ("every weekday", "0 9 * * 1-5"),
    ("every week", "0 9 * * 1"),
    ("every month", "0 9 1 * *"),
    ("midnight", "0 0 * * *"),
    ("noon", "0 12 * * *"),
])
def test_nl_to_cron_known_phrases(phrase, expected):
    assert _nl_to_cron(phrase) == expected


def test_nl_to_cron_with_time_every_day_at():
    result = _nl_to_cron("every day at 9")
    assert result == "0 9 * * *"


def test_nl_to_cron_with_minutes_every_day_at():
    result = _nl_to_cron("every day at 14:30")
    assert result == "30 14 * * *"


def test_nl_to_cron_case_insensitive():
    assert _nl_to_cron("Every Morning") == "0 8 * * *"
    assert _nl_to_cron("EVERY MINUTE") == "* * * * *"


def test_nl_to_cron_unknown_returns_message():
    result = _nl_to_cron("whenever the moon is full")
    assert "Could not parse" in result


def test_nl_to_cron_every_1_minute():
    assert _nl_to_cron("every 1 minute") == "*/1 * * * *"


# ── _validate_cron ────────────────────────────────────────────────────────────

def test_validate_cron_valid():
    result = _validate_cron("0 8 * * 1")
    assert result.startswith("Valid cron")


def test_validate_cron_wildcard_fields():
    result = _validate_cron("* * * * *")
    assert "Valid cron" in result


def test_validate_cron_wrong_field_count():
    result = _validate_cron("0 8 * *")
    assert "Invalid cron" in result


def test_validate_cron_out_of_range_hour():
    result = _validate_cron("0 25 * * *")
    assert "hour" in result and "25" in result


def test_validate_cron_out_of_range_minute():
    result = _validate_cron("65 * * * *")
    assert "minute" in result


def test_validate_cron_step_expressions():
    result = _validate_cron("*/15 * * * *")
    assert "Valid cron" in result


# ── _make_fn ──────────────────────────────────────────────────────────────────

def test_make_fn_executes_code():
    code = "def run(**kwargs):\n    return 'result: ' + kwargs['x']"
    fn = _make_fn(code)
    assert fn(x="hello") == "result: hello"


def test_make_fn_arithmetic():
    code = "def run(**kwargs):\n    return str(kwargs['a'] + kwargs['b'])"
    fn = _make_fn(code)
    assert fn(a=3, b=4) == "7"


# ── synthesise_tool ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesise_tool_registers_and_returns(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path / "tools")
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    (tmp_path / "tools").mkdir()

    payload = {
        "name": "greet_tool",
        "description": "Greet the user.",
        "category": "general",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Name"}},
            "required": ["name"],
        },
        "python_code": "def run(**kwargs):\n    return f'Hello {kwargs[\"name\"]}!'",
    }
    import json
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=json.dumps(payload))]
    ))

    from jarvis.tools.registry import build_registry
    registry = build_registry()
    tool = await synthesise_tool("greet the user by name", client, registry)

    assert tool is not None
    assert tool.name == "greet_tool"
    assert registry.get("greet_tool") is not None
    assert (tmp_path / "tools" / "greet_tool.py").exists()


@pytest.mark.asyncio
async def test_synthesise_tool_returns_none_on_api_error(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))

    from jarvis.tools.registry import build_registry
    tool = await synthesise_tool("some capability", client, build_registry())
    assert tool is None


@pytest.mark.asyncio
async def test_synthesise_tool_strips_code_fence(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path / "t")
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    (tmp_path / "t").mkdir()

    import json
    payload = {
        "name": "fenced_tool",
        "description": "desc",
        "category": "general",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "python_code": "def run(**kwargs):\n    return 'ok'",
    }
    fenced = f"```json\n{json.dumps(payload)}\n```"
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=fenced)]
    ))

    from jarvis.tools.registry import build_registry
    tool = await synthesise_tool("fenced capability", client, build_registry())
    assert tool is not None
    assert tool.name == "fenced_tool"
