"""Tests for jarvis.tools.nlp_cron and jarvis.tools.creator."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

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


# ── Additional _nl_to_cron cases (remaining day names + "daily at") ───────────

@pytest.mark.parametrize("phrase,expected", [
    ("every tuesday",   "0 9 * * 2"),
    ("every wednesday", "0 9 * * 3"),
    ("every thursday",  "0 9 * * 4"),
    ("Every Tuesday",   "0 9 * * 2"),   # case-insensitive
    ("daily at 9",      "0 9 * * *"),
    ("daily at 14:30",  "30 14 * * *"),
    ("daily at 0",      "0 0 * * *"),   # midnight via daily-at
])
def test_nl_to_cron_remaining_patterns(phrase, expected):
    assert _nl_to_cron(phrase) == expected


# ── Additional _validate_cron boundary cases ──────────────────────────────────

@pytest.mark.parametrize("expr,keyword", [
    ("0 8 0 * *",   "day of month"),   # day < 1
    ("0 8 32 * *",  "day of month"),   # day > 31
    ("0 8 * 0 *",   "month"),          # month < 1
    ("0 8 * 13 *",  "month"),          # month > 12
])
def test_validate_cron_out_of_range_fields(expr, keyword):
    result = _validate_cron(expr)
    assert "issues" in result.lower() or keyword in result


def test_validate_cron_day_of_week_7_is_valid():
    """Day-of-week 7 (Sunday) is within [0,7]."""
    result = _validate_cron("0 9 * * 7")
    assert "Valid cron" in result


def test_validate_cron_range_expression():
    """'1-5' splits on '-' and takes first value for range check."""
    result = _validate_cron("0 9 * * 1-5")
    assert "Valid cron" in result


def test_validate_cron_comma_expression():
    """'6,0' splits on ',' and takes first value for range check."""
    result = _validate_cron("0 9 * * 6,0")
    assert "Valid cron" in result


def test_validate_cron_six_fields_invalid():
    result = _validate_cron("0 8 * * * *")
    assert "Invalid cron" in result


# ── subagent_tools: delegation edge cases ────────────────────────────────────

def test_delegate_no_pool_returns_error():
    import jarvis.tools.subagent_tools as st
    old_pool = st._pool
    try:
        st._pool = None
        out = st._delegate("do something")
        assert "not initialised" in out
    finally:
        st._pool = old_pool


def test_delegate_parallel_no_pool_returns_error():
    import jarvis.tools.subagent_tools as st
    old_pool = st._pool
    try:
        st._pool = None
        out = st._delegate_parallel([{"goal": "task1"}])
        assert "not initialised" in out
    finally:
        st._pool = old_pool


def test_delegate_exception_returns_error_string():
    import jarvis.tools.subagent_tools as st
    from unittest.mock import MagicMock, patch
    fake_pool = MagicMock()
    old_pool = st._pool
    try:
        st._pool = fake_pool
        with patch("asyncio.run", side_effect=RuntimeError("pool exploded")):
            out = st._delegate("do something")
        assert "Subagent error" in out
        assert "pool exploded" in out
    finally:
        st._pool = old_pool


def test_delegate_parallel_exception_returns_error_string():
    import jarvis.tools.subagent_tools as st
    from unittest.mock import MagicMock, patch
    fake_pool = MagicMock()
    old_pool = st._pool
    try:
        st._pool = fake_pool
        with patch("asyncio.run", side_effect=RuntimeError("parallel error")):
            out = st._delegate_parallel([{"goal": "task1"}])
        assert "Parallel subagent error" in out
    finally:
        st._pool = old_pool


def test_delegate_parallel_result_truncated_to_500():
    """Results longer than 500 chars are truncated per the slice in the implementation."""
    import jarvis.tools.subagent_tools as st
    import json
    from unittest.mock import MagicMock, patch
    fake_pool = MagicMock()
    old_pool = st._pool
    try:
        st._pool = fake_pool
        long_result = "x" * 600
        fake_results = {"task_0": long_result}
        with patch("asyncio.run", return_value=fake_results):
            out = st._delegate_parallel([{"goal": "do task"}])
        parsed = json.loads(out)
        for val in parsed.values():
            assert len(val) <= 500
    finally:
        st._pool = old_pool


def test_register_tools_with_jarvis_none_skips_pool_creation():
    """register_tools(registry, jarvis=None) should not create a pool."""
    import jarvis.tools.subagent_tools as st
    from jarvis.tools.registry import ToolRegistry
    old_pool = st._pool
    try:
        st._pool = None
        registry = ToolRegistry()
        st.register_tools(registry, jarvis=None)
        # Pool should still be None — jarvis=None skips SubagentPool creation
        assert st._pool is None
        # But tools should be registered
        assert registry.get("delegate_task") is not None
        assert registry.get("delegate_parallel") is not None
    finally:
        st._pool = old_pool


# ── creator: _make_fn edge cases ─────────────────────────────────────────────

def test_make_fn_no_run_function_raises():
    """Code that doesn't define `run` raises KeyError."""
    from jarvis.tools.creator import _make_fn
    code = "x = 42"  # no run() defined
    import pytest
    with pytest.raises(KeyError):
        _make_fn(code)


def test_make_fn_syntax_error_raises():
    """Syntactically invalid code raises SyntaxError."""
    from jarvis.tools.creator import _make_fn
    import pytest
    with pytest.raises(SyntaxError):
        _make_fn("def run(**kwargs):\n    return ??? invalid")


def test_make_fn_run_uses_kwargs():
    """The compiled run function can access all kwargs."""
    from jarvis.tools.creator import _make_fn
    code = "def run(**kwargs):\n    return f\"{kwargs['a']}-{kwargs['b']}\""
    fn = _make_fn(code)
    assert fn(a="hello", b="world") == "hello-world"


# ── personality: all-defaults call returns pure base prompt ──────────────────

def test_get_system_prompt_all_defaults_returns_base_only():
    from jarvis.personality import get_system_prompt, PERSONALITY_PROFILES
    prompt = get_system_prompt(profile="default", voice_mode=False, memory_context="", gap_context="")
    assert prompt == PERSONALITY_PROFILES["default"]


def test_get_system_prompt_both_memory_and_gap_context():
    from jarvis.personality import get_system_prompt
    prompt = get_system_prompt(memory_context="user hates spam", gap_context="can't parse PDFs")
    assert "user hates spam" in prompt
    assert "can't parse PDFs" in prompt
    assert "Your Current Memory" in prompt
    assert "Capability Gaps" in prompt


# ── geo_weather_tools ─────────────────────────────────────────────────────────

from jarvis.tools.geo_weather_tools import (  # noqa: E402
    _geocode, _get_weather, _read_rss, _ocr_image, _log_analyse,
)


def test_geocode_no_results():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = []
        out = _geocode("nowhere special")
    assert "No results" in out


def test_geocode_returns_lat_lon():
    result = [{"display_name": "London, UK", "lat": "51.5", "lon": "-0.1"}]
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = result
        out = _geocode("London")
    assert "lat:51.5" in out
    assert "lon:-0.1" in out


def test_geocode_exception_returns_error():
    with patch("requests.get", side_effect=Exception("network down")):
        out = _geocode("anywhere")
    assert "Geocode error" in out


def test_get_weather_location_not_found():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = []
        out = _get_weather("UnknownCity")
    assert "Location not found" in out


def test_get_weather_returns_lines():
    geo = [{"display_name": "Paris, France", "lat": "48.8", "lon": "2.3"}]
    daily = {
        "time": ["2026-05-26"],
        "temperature_2m_max": [22.0],
        "temperature_2m_min": [14.0],
        "precipitation_sum": [0.5],
    }
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.side_effect = [geo, {"daily": daily}]
        out = _get_weather("Paris", days=1)
    assert "Paris, France" in out
    assert "2026-05-26" in out


def test_get_weather_exception_returns_error():
    with patch("requests.get", side_effect=Exception("timeout")):
        out = _get_weather("Paris")
    assert "Weather error" in out


def test_read_rss_no_feedparser_fallback(tmp_path):
    xml = (
        "<rss><channel>"
        "<item><title>Hello</title><link>http://x.com</link></item>"
        "</channel></rss>"
    )
    fake_resp = MagicMock()
    fake_resp.text = xml
    with patch.dict(sys.modules, {"feedparser": None}):
        with patch("requests.get", return_value=fake_resp):
            out = _read_rss("http://example.com/feed")
    assert "Hello" in out


def test_read_rss_feedparser_present():
    fake_entry = MagicMock()
    fake_entry.get = lambda k, d="": {"title": "Item1", "link": "http://x", "summary": "desc"}.get(k, d)
    fake_feed = MagicMock()
    fake_feed.entries = [fake_entry]
    fake_fp = MagicMock()
    fake_fp.parse.return_value = fake_feed
    with patch.dict(sys.modules, {"feedparser": fake_fp}):
        out = _read_rss("http://example.com/rss")
    assert "Item1" in out


def test_ocr_image_no_pytesseract():
    with patch.dict(sys.modules, {"pytesseract": None}):
        out = _ocr_image("/tmp/fake.png")
    assert "not installed" in out.lower() or "pytesseract" in out


def test_log_analyse_tail_and_count(tmp_path):
    log = tmp_path / "app.log"
    log.write_text("INFO line\nERROR boom\nDEBUG ok\n")
    out = _log_analyse(str(log))
    assert "Total lines:" in out
    assert "Errors:" in out


def test_log_analyse_pattern_filter(tmp_path):
    log = tmp_path / "app.log"
    log.write_text("INFO foo\nERROR bar\nINFO baz\n")
    out = _log_analyse(str(log), pattern="ERROR")
    assert "ERROR" in out


def test_log_analyse_missing_file():
    out = _log_analyse("/nonexistent/file.log")
    assert "Log error" in out


def test_geo_weather_register_tools_populates_registry():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.geo_weather_tools import register_tools
    reg = build_registry()
    register_tools(reg)
    for name in ("geocode", "get_weather", "read_rss", "ocr_image", "analyse_logs"):
        assert reg.get(name) is not None
