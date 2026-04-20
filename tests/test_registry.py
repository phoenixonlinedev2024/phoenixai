"""Tests for jarvis.tools.registry — tool registration and execution."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jarvis.tools.registry import (
    Tool,
    ToolRegistry,
    _api_call,
    _execute_python,
    _get_system_info,
    _list_directory,
    _web_fetch,
    _web_search,
    _write_and_run_code,
)


def _make_tool(name="echo", fn=None):
    fn = fn or (lambda text="": text)
    return Tool(
        name=name,
        description=f"Echo tool: {name}",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        fn=fn,
        category="test",
    )


def test_register_and_get():
    registry = ToolRegistry()
    t = _make_tool("greet")
    registry.register(t)
    assert registry.get("greet") is t


def test_get_unknown_returns_none():
    registry = ToolRegistry()
    assert registry.get("nonexistent") is None


def test_all_returns_list():
    registry = ToolRegistry()
    registry.register(_make_tool("a"))
    registry.register(_make_tool("b"))
    names = [t.name for t in registry.all()]
    assert "a" in names and "b" in names


def test_tool_run():
    registry = ToolRegistry()
    registry.register(_make_tool("upper", fn=lambda text="": text.upper()))
    result = registry.get("upper").run(text="hello")
    assert result == "HELLO"


def test_anthropic_tools_schema():
    registry = ToolRegistry()
    registry.register(_make_tool("ping"))
    tools = registry.anthropic_tools()
    assert isinstance(tools, list)
    assert any(t["name"] == "ping" for t in tools)
    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "input_schema" in t


def test_build_registry_includes_core_tools(tool_registry):
    names = [t.name for t in tool_registry.all()]
    # A selection of tools that should always be present
    for expected in ("run_shell", "read_file", "write_file", "web_search"):
        assert expected in names, f"Missing built-in tool: {expected}"


def test_dynamic_tool_flag():
    t = Tool(name="dyn", description="d", input_schema={}, fn=lambda: "ok",
             category="test", dynamic=True)
    assert t.dynamic is True


def test_registry_categories(tool_registry):
    categories = {t.category for t in tool_registry.all()}
    assert len(categories) >= 2  # at least a few categories present


# ── ToolRegistry.names() ──────────────────────────────────────────────────────

def test_names_returns_registered_names():
    registry = ToolRegistry()
    registry.register(_make_tool("alpha"))
    registry.register(_make_tool("beta"))
    assert "alpha" in registry.names()
    assert "beta" in registry.names()


def test_names_empty_registry():
    assert ToolRegistry().names() == []


def test_names_matches_all():
    registry = ToolRegistry()
    for name in ("x", "y", "z"):
        registry.register(_make_tool(name))
    assert set(registry.names()) == {"x", "y", "z"}


# ── Tool.to_anthropic() ───────────────────────────────────────────────────────

def test_to_anthropic_structure():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    t = Tool(name="searcher", description="Searches stuff.", input_schema=schema,
             fn=lambda q="": q, category="web")
    d = t.to_anthropic()
    assert d["name"] == "searcher"
    assert d["description"] == "Searches stuff."
    assert d["input_schema"] is schema
    assert set(d.keys()) == {"name", "description", "input_schema"}


# ── ToolRegistry.load_dynamic_tools() ────────────────────────────────────────

def test_load_dynamic_tools_registers_tool(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    (tmp_path / "my_dyn_tool.py").write_text(
        "from jarvis.tools.registry import Tool\n"
        "def register_tools(reg):\n"
        "    reg.register(Tool('dyn_hello', 'A dynamic tool.', {}, fn=lambda: 'hi', category='test'))\n"
    )
    registry = ToolRegistry()
    registry.load_dynamic_tools()
    t = registry.get("dyn_hello")
    assert t is not None
    assert t.run() == "hi"


def test_load_dynamic_tools_skips_broken_file(tmp_path, monkeypatch, capsys):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    (tmp_path / "busted.py").write_text("this is NOT valid python !!!")
    registry = ToolRegistry()
    registry.load_dynamic_tools()  # must not raise
    assert len(registry.all()) == 0
    assert "Warning" in capsys.readouterr().out


def test_load_dynamic_tools_skips_file_without_register(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    (tmp_path / "plain.py").write_text("x = 42\n")
    registry = ToolRegistry()
    registry.load_dynamic_tools()
    assert len(registry.all()) == 0


def test_load_dynamic_tools_creates_dir_if_missing(tmp_path, monkeypatch):
    from jarvis.config import cfg
    missing = tmp_path / "nonexistent_dir"
    monkeypatch.setattr(cfg, "TOOLS_DIR", missing)
    ToolRegistry().load_dynamic_tools()
    assert missing.exists()


# ── _web_search ───────────────────────────────────────────────────────────────

def test_web_search_returns_formatted_results():
    fake_ddgs = MagicMock()
    fake_ddgs.__enter__ = MagicMock(return_value=fake_ddgs)
    fake_ddgs.__exit__ = MagicMock(return_value=False)
    fake_ddgs.text = MagicMock(return_value=[
        {"title": "Result One", "href": "https://example.com", "body": "Some info."},
    ])
    fake_module = MagicMock()
    fake_module.DDGS = MagicMock(return_value=fake_ddgs)
    with patch.dict(sys.modules, {"duckduckgo_search": fake_module}):
        out = _web_search("pytest")
    assert "Result One" in out
    assert "example.com" in out


def test_web_search_no_results():
    fake_ddgs = MagicMock()
    fake_ddgs.__enter__ = MagicMock(return_value=fake_ddgs)
    fake_ddgs.__exit__ = MagicMock(return_value=False)
    fake_ddgs.text = MagicMock(return_value=[])
    fake_module = MagicMock()
    fake_module.DDGS = MagicMock(return_value=fake_ddgs)
    with patch.dict(sys.modules, {"duckduckgo_search": fake_module}):
        out = _web_search("nothing")
    assert "No results" in out


def test_web_search_exception_returns_error():
    fake_module = MagicMock()
    fake_module.DDGS = MagicMock(side_effect=RuntimeError("DNS failure"))
    with patch.dict(sys.modules, {"duckduckgo_search": fake_module}):
        out = _web_search("boom")
    assert "Search error" in out


# ── _web_fetch ────────────────────────────────────────────────────────────────

def test_web_fetch_extracts_text():
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.text = "<html><body><p>Hello world</p></body></html>"
    fake_requests = MagicMock()
    fake_requests.get = MagicMock(return_value=fake_resp)

    fake_bs4 = MagicMock()

    class FakeSoup:
        def __init__(self, text, parser):
            self._text = text

        def __call__(self, tags):
            return []

        def get_text(self, separator="\n", strip=False):
            return "Hello world"

    fake_bs4.BeautifulSoup = FakeSoup

    with patch.dict(sys.modules, {"requests": fake_requests, "bs4": fake_bs4}):
        out = _web_fetch("https://example.com")
    assert "Hello world" in out


def test_web_fetch_exception_returns_error():
    fake_requests = MagicMock()
    fake_requests.get = MagicMock(side_effect=ConnectionError("refused"))
    fake_bs4 = MagicMock()
    with patch.dict(sys.modules, {"requests": fake_requests, "bs4": fake_bs4}):
        out = _web_fetch("https://bad.example")
    assert "Fetch error" in out


# ── _list_directory ───────────────────────────────────────────────────────────

def test_list_directory_shows_files_and_dirs(tmp_path):
    (tmp_path / "file.txt").write_text("hello")
    (tmp_path / "subdir").mkdir()
    out = _list_directory(str(tmp_path))
    assert "file.txt" in out
    assert "subdir" in out


def test_list_directory_empty(tmp_path):
    out = _list_directory(str(tmp_path))
    assert out == "(empty)"


def test_list_directory_missing_path():
    out = _list_directory("/no/such/dir/xyz123")
    assert "List error" in out


# ── _execute_python ───────────────────────────────────────────────────────────

def test_execute_python_simple_output():
    out = _execute_python("print(2 + 2)")
    assert "4" in out


def test_execute_python_no_output():
    out = _execute_python("x = 42")
    assert out == "(no output)"


def test_execute_python_syntax_error():
    out = _execute_python("def foo(")
    assert "Error" in out or "error" in out.lower()


def test_execute_python_timeout():
    import subprocess
    with patch("jarvis.tools.registry.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="python", timeout=30)):
        out = _execute_python("while True: pass")
    assert "timed out" in out.lower()


# ── _api_call ────────────────────────────────────────────────────────────────

def test_api_call_get_json_response():
    fake_resp = MagicMock()
    fake_resp.json = MagicMock(return_value={"status": "ok"})
    fake_requests = MagicMock()
    fake_requests.request = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _api_call("https://api.example.com/data")
    parsed = json.loads(out)
    assert parsed["status"] == "ok"


def test_api_call_post_with_body():
    fake_resp = MagicMock()
    fake_resp.json = MagicMock(return_value={"created": True})
    fake_requests = MagicMock()
    fake_requests.request = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        _api_call("https://api.example.com/items", method="POST", body={"name": "widget"})
    call_args = fake_requests.request.call_args
    assert call_args[0][0] == "POST"
    assert call_args[1]["json"] == {"name": "widget"}


def test_api_call_text_fallback_when_not_json():
    fake_resp = MagicMock()
    fake_resp.json = MagicMock(side_effect=ValueError("not JSON"))
    fake_resp.text = "plain text response"
    fake_requests = MagicMock()
    fake_requests.request = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _api_call("https://api.example.com/text")
    assert "plain text" in out


def test_api_call_network_error():
    fake_requests = MagicMock()
    fake_requests.request = MagicMock(side_effect=ConnectionError("no route"))
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _api_call("https://dead.example.com")
    assert "API call error" in out


# ── _get_system_info ──────────────────────────────────────────────────────────

def test_get_system_info_returns_nonempty():
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent = MagicMock(return_value=15.5)
    fake_mem = MagicMock(percent=60.0, used=2 * 1024**2, total=8 * 1024**2)
    fake_psutil.virtual_memory = MagicMock(return_value=fake_mem)
    fake_disk = MagicMock(percent=40.0, used=50 * 1024**3, total=200 * 1024**3)
    fake_psutil.disk_usage = MagicMock(return_value=fake_disk)
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        out = _get_system_info()
    assert "CPU" in out
    assert "Memory" in out
    assert "Disk" in out


def test_get_system_info_error_returns_message():
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent = MagicMock(side_effect=RuntimeError("no sensors"))
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        out = _get_system_info()
    assert "System info error" in out


# ── _write_and_run_code ───────────────────────────────────────────────────────

def test_write_and_run_code_python():
    out = _write_and_run_code("python", "print('hello from test')")
    assert "hello from test" in out


def test_write_and_run_code_no_output():
    out = _write_and_run_code("python", "x = 1 + 1")
    assert out == "(no output)"


def test_write_and_run_code_unknown_language():
    out = _write_and_run_code("cobol", "IDENTIFICATION DIVISION.")
    assert "No runner configured" in out


def test_write_and_run_code_timeout():
    import subprocess
    with patch("jarvis.tools.registry.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd=[], timeout=30)):
        out = _write_and_run_code("python", "import time; time.sleep(999)")
    assert "timed out" in out.lower()


def test_write_and_run_code_bash(tmp_path):
    out = _write_and_run_code("bash", "echo 'bash works'")
    assert "bash works" in out
