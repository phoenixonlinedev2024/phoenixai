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
    _read_file,
    _run_shell,
    _web_fetch,
    _web_search,
    _write_and_run_code,
    _write_file,
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


# ── _read_file ────────────────────────────────────────────────────────────────

def test_read_file_returns_content(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    out = _read_file(str(f))
    assert out == "hello world"


def test_read_file_missing_returns_error():
    out = _read_file("/no/such/file/xyz_missing.txt")
    assert "Read error" in out


# ── _write_file ───────────────────────────────────────────────────────────────

def test_write_file_creates_and_writes(tmp_path):
    dest = str(tmp_path / "output.txt")
    out = _write_file(dest, "test content")
    assert "Written" in out
    assert Path(dest).read_text() == "test content"


def test_write_file_creates_parent_dirs(tmp_path):
    dest = str(tmp_path / "a" / "b" / "c.txt")
    _write_file(dest, "nested")
    assert Path(dest).exists()


def test_write_file_reports_char_count(tmp_path):
    dest = str(tmp_path / "count.txt")
    out = _write_file(dest, "12345")
    assert "5" in out


# ── _run_shell ────────────────────────────────────────────────────────────────

def test_run_shell_captures_stdout():
    out = _run_shell("echo hello_from_shell")
    assert "hello_from_shell" in out


def test_run_shell_exit_code_included():
    out = _run_shell("exit 0", timeout=5)
    assert "Exit code: 0" in out


def test_run_shell_nonzero_exit_code():
    out = _run_shell("exit 42", timeout=5)
    assert "42" in out


def test_run_shell_timeout():
    import subprocess
    with patch("jarvis.tools.registry.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="sleep", timeout=30)):
        out = _run_shell("sleep 999")
    assert "timed out" in out.lower()


def test_run_shell_exception():
    with patch("jarvis.tools.registry.subprocess.run", side_effect=OSError("no shell")):
        out = _run_shell("impossible")
    assert "Shell error" in out


def test_run_shell_captures_stderr():
    """Line 161: STDERR branch in _run_shell is exercised by a command writing to stderr."""
    out = _run_shell("echo 'err msg' >&2")
    assert "STDERR" in out
    assert "err msg" in out


def test_web_fetch_decomposes_nav_tags():
    """Line 111: tag.decompose() runs when HTML contains nav/script/aside tags."""
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.text = "<html><nav>junk</nav><p>content</p></html>"
    fake_requests = MagicMock()
    fake_requests.get = MagicMock(return_value=fake_resp)

    fake_tag = MagicMock()

    class FakeSoup:
        def __init__(self, text, parser):
            pass

        def __call__(self, tags):
            return [fake_tag]

        def get_text(self, separator="\n", strip=False):
            return "content"

    fake_bs4 = MagicMock()
    fake_bs4.BeautifulSoup = FakeSoup

    with patch.dict(sys.modules, {"requests": fake_requests, "bs4": fake_bs4}):
        out = _web_fetch("https://example.com")
    assert "content" in out
    fake_tag.decompose.assert_called_once()


def test_write_file_error(tmp_path):
    """Lines 131-132: Write error path when the target path cannot be written."""
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file")
    out = _write_file(str(blocker / "subfile.txt"), "content")
    assert "Write error" in out


def test_execute_python_exception():
    """Lines 189-190: Execution error when subprocess.run raises an OS-level error."""
    import subprocess
    with patch("jarvis.tools.registry.subprocess.run", side_effect=OSError("no interpreter")):
        out = _execute_python("print(1)")
    assert "Execution error" in out


def test_write_and_run_code_stderr_only():
    """Line 264: Error/stderr branch in _write_and_run_code when code writes to stderr."""
    out = _write_and_run_code("python", "import sys; sys.stderr.write('oops\\n')")
    assert "oops" in out


def test_write_and_run_code_exception():
    """Lines 268-269: Generic exception path in _write_and_run_code."""
    import subprocess
    with patch("jarvis.tools.registry.subprocess.run", side_effect=OSError("runner gone")):
        out = _write_and_run_code("python", "print(1)")
    assert "Error" in out


def test_build_registry_swallows_tool_module_exceptions(capsys):
    """Lines 431-468: build_registry continues even when optional tool modules fail."""
    import jarvis.tools.browser as browser_mod
    import jarvis.tools.email_tool as email_mod
    import jarvis.tools.github_tool as github_mod
    import jarvis.tools.package_installer as pkg_mod
    import jarvis.tools.image_gen as image_mod
    import jarvis.tools.nlp_cron as nlp_mod
    import jarvis.tools.sandbox_tools as sandbox_mod

    boom = MagicMock(side_effect=RuntimeError("unavailable"))
    with patch.object(browser_mod, "register_tools", boom), \
         patch.object(email_mod, "register_tools", boom), \
         patch.object(github_mod, "register_tools", boom), \
         patch.object(pkg_mod, "register_tools", boom), \
         patch.object(image_mod, "register_tools", boom), \
         patch.object(nlp_mod, "register_tools", boom), \
         patch.object(sandbox_mod, "register_tools", boom):
        from jarvis.tools.registry import build_registry
        reg = build_registry()
    assert reg is not None
    out = capsys.readouterr().out
    assert "unavailable" in out


# ── Tool dataclass ─────────────────────────────────────────────────────────────

def test_tool_dynamic_defaults_to_false():
    t = _make_tool("mytest")
    assert t.dynamic is False


def test_tool_dynamic_can_be_set_true():
    t = Tool(
        name="dynamic_one",
        description="runtime-created",
        input_schema={"type": "object"},
        fn=lambda: "ok",
        dynamic=True,
    )
    assert t.dynamic is True


def test_tool_category_stored():
    t = Tool(
        name="cat_tool",
        description="d",
        input_schema={"type": "object"},
        fn=lambda: "x",
        category="networking",
    )
    assert t.category == "networking"


def test_tool_to_anthropic_structure():
    t = _make_tool("schema_test")
    d = t.to_anthropic()
    assert d["name"] == "schema_test"
    assert d["description"].startswith("Echo tool")
    assert "input_schema" in d
    assert d["input_schema"]["type"] == "object"


# ── ToolRegistry.names() ──────────────────────────────────────────────────────

def test_registry_names_returns_all():
    registry = ToolRegistry()
    registry.register(_make_tool("x"))
    registry.register(_make_tool("y"))
    names = registry.names()
    assert "x" in names
    assert "y" in names
    assert len(names) == 2


def test_registry_names_empty():
    registry = ToolRegistry()
    assert registry.names() == []


# ── Tool.run() with various argument patterns ─────────────────────────────────

def test_tool_run_with_multiple_args():
    def adder(a=0, b=0):
        return a + b

    t = Tool(
        name="adder",
        description="adds",
        input_schema={"type": "object", "properties": {"a": {}, "b": {}}},
        fn=adder,
    )
    assert t.run(a=3, b=4) == 7


def test_tool_run_no_args():
    t = Tool(
        name="noop",
        description="d",
        input_schema={"type": "object"},
        fn=lambda: "static result",
    )
    assert t.run() == "static result"


# ── ToolRegistry: register overwrites existing tool ────────────────────────────

def test_register_overwrites_existing():
    registry = ToolRegistry()
    registry.register(Tool("same_name", "first", {}, fn=lambda: "first"))
    registry.register(Tool("same_name", "second", {}, fn=lambda: "second"))
    assert registry.get("same_name").description == "second"
    assert len(registry.all()) == 1


# ── anthropic_tools format ────────────────────────────────────────────────────

def test_anthropic_tools_returns_all_tools():
    registry = ToolRegistry()
    for i in range(5):
        registry.register(_make_tool(f"tool_{i}"))
    tools = registry.anthropic_tools()
    assert len(tools) == 5
    names = {t["name"] for t in tools}
    assert {f"tool_{i}" for i in range(5)} == names


# ── _run_shell with cwd parameter ────────────────────────────────────────────

def test_run_shell_with_cwd(tmp_path):
    """cwd parameter sets the working directory for the command."""
    (tmp_path / "testfile.txt").write_text("content")
    out = _run_shell("ls testfile.txt", cwd=str(tmp_path))
    assert "testfile.txt" in out


def test_run_shell_both_stdout_and_stderr():
    """When command produces both stdout and stderr both are included in output."""
    cmd = "echo stdout_text; echo stderr_text >&2"
    out = _run_shell(cmd)
    assert "STDOUT" in out
    assert "STDERR" in out
    assert "stdout_text" in out
    assert "stderr_text" in out


# ── _list_directory empty directory ──────────────────────────────────────────

def test_list_directory_empty_dir_returns_empty_marker(tmp_path):
    """Empty directory returns the '(empty)' placeholder."""
    out = _list_directory(str(tmp_path))
    assert out == "(empty)"


# ── _api_call text fallback when response is not JSON ────────────────────────

def test_api_call_text_fallback_non_json():
    """When resp.json() raises, text[:4000] is returned instead."""
    from unittest.mock import MagicMock, patch
    fake_resp = MagicMock()
    fake_resp.json.side_effect = ValueError("not json")
    fake_resp.text = "plain text response"

    fake_requests = MagicMock()
    fake_requests.request.return_value = fake_resp

    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _api_call("https://example.com/endpoint")
    assert "plain text response" in out


# ── Tool.to_anthropic() structure ────────────────────────────────────────────

def test_tool_to_anthropic_includes_input_schema():
    """to_anthropic() includes input_schema key (for Claude API)."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    t = Tool("test_t", "desc", schema, fn=lambda x: x)
    d = t.to_anthropic()
    assert d["input_schema"] == schema
    assert "description" in d
    assert d["name"] == "test_t"


# ── Additional coverage ────────────────────────────────────────────────────────

def test_web_search_max_results_forwarded_to_ddgs():
    """_web_search passes max_results to DDGS.text()."""
    fake_ddgs = MagicMock()
    fake_ddgs.__enter__ = MagicMock(return_value=fake_ddgs)
    fake_ddgs.__exit__ = MagicMock(return_value=False)
    fake_ddgs.text = MagicMock(return_value=[{"title": "T", "href": "http://x", "body": "b"}])
    fake_module = MagicMock()
    fake_module.DDGS = MagicMock(return_value=fake_ddgs)
    with patch.dict(sys.modules, {"duckduckgo_search": fake_module}):
        _web_search("test", max_results=7)
    fake_ddgs.text.assert_called_once_with("test", max_results=7)


def test_web_fetch_http_error_returns_fetch_error():
    """_web_fetch returns error string when raise_for_status raises."""
    fake_resp = MagicMock()
    fake_resp.raise_for_status.side_effect = Exception("404 Not Found")
    fake_requests = MagicMock()
    fake_requests.get = MagicMock(return_value=fake_resp)
    fake_bs4 = MagicMock()
    with patch.dict(sys.modules, {"requests": fake_requests, "bs4": fake_bs4}):
        out = _web_fetch("https://example.com/notfound")
    assert "Fetch error" in out
    assert "404" in out


def test_api_call_put_method():
    """_api_call sends PUT request and returns JSON response."""
    fake_requests = MagicMock()
    mock_req = fake_requests.request
    resp = MagicMock()
    resp.json.return_value = {"updated": True}
    mock_req.return_value = resp
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _api_call("https://api.example.com/item/1", method="PUT", body={"name": "new"})
    mock_req.assert_called_once()
    call_args = mock_req.call_args
    assert call_args[0][0] == "PUT"
    assert "updated" in out


def test_api_call_custom_headers_forwarded():
    """_api_call passes the headers dict to requests.request."""
    fake_requests = MagicMock()
    mock_req = fake_requests.request
    resp = MagicMock()
    resp.json.return_value = {}
    mock_req.return_value = resp
    custom_headers = {"Authorization": "Bearer token123"}
    with patch.dict(sys.modules, {"requests": fake_requests}):
        _api_call("https://api.example.com/data", headers=custom_headers)
    _, kwargs = mock_req.call_args
    assert kwargs["headers"] == custom_headers


def test_api_call_delete_method_returns_json():
    """_api_call sends DELETE request and returns JSON."""
    fake_requests = MagicMock()
    mock_req = fake_requests.request
    resp = MagicMock()
    resp.json.return_value = {"deleted": True}
    mock_req.return_value = resp
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _api_call("https://api.example.com/item/99", method="DELETE")
    assert "deleted" in out
    call_method = mock_req.call_args[0][0]
    assert call_method == "DELETE"


def test_get_system_info_format_has_cpu_memory_disk():
    """_get_system_info returns string with CPU, Memory, Disk sections."""
    fake_mem = MagicMock()
    fake_mem.percent = 42.0
    fake_mem.used = 1024 ** 3  # 1 GB
    fake_mem.total = 4 * 1024 ** 3
    fake_disk = MagicMock()
    fake_disk.percent = 75.0
    fake_disk.used = 50 * 1024 ** 3
    fake_disk.total = 100 * 1024 ** 3
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent = MagicMock(return_value=33.5)
    fake_psutil.virtual_memory = MagicMock(return_value=fake_mem)
    fake_psutil.disk_usage = MagicMock(return_value=fake_disk)
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        out = _get_system_info()
    assert "CPU" in out
    assert "33.5" in out
    assert "Memory" in out
    assert "42.0" in out
    assert "Disk" in out
    assert "75.0" in out


def test_build_registry_nlp_cron_and_sandbox_tools_registered():
    """build_registry registers tools from nlp_cron and sandbox_tools modules."""
    from jarvis.tools.registry import build_registry
    reg = build_registry()
    names = reg.names()
    assert "nl_to_cron" in names
    assert "sandbox_run" in names
    assert "sandbox_run_code" in names


def test_web_fetch_truncates_long_content():
    """_web_fetch caps output at 8000 chars for very large pages."""
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.text = "<p>" + "x" * 9000 + "</p>"
    fake_requests = MagicMock()
    fake_requests.get = MagicMock(return_value=fake_resp)

    class FakeSoup:
        def __init__(self, text, parser): pass
        def __call__(self, tags): return []
        def get_text(self, separator="\n", strip=False): return "y" * 9000

    fake_bs4 = MagicMock()
    fake_bs4.BeautifulSoup = FakeSoup

    with patch.dict(sys.modules, {"requests": fake_requests, "bs4": fake_bs4}):
        out = _web_fetch("https://example.com/large")
    assert len(out) == 8000


def test_list_directory_lists_dirs_before_files(tmp_path):
    """_list_directory sorts dirs before files (key=lambda: (is_file, name))."""
    (tmp_path / "a_file.txt").write_text("x")
    (tmp_path / "b_subdir").mkdir()
    out = _list_directory(str(tmp_path))
    lines = out.splitlines()
    # Directory should appear before file in the output
    dir_pos = next(i for i, l in enumerate(lines) if "[DIR]" in l)
    file_pos = next(i for i, l in enumerate(lines) if "[FILE]" in l)
    assert dir_pos < file_pos


def test_run_shell_no_output_shows_exit_code():
    """_run_shell with no stdout/stderr still returns the exit code line."""
    out = _run_shell("true")  # no output, exit 0
    assert "Exit code:" in out
    assert "0" in out


def test_run_shell_exception_returns_error():
    """_run_shell exception (e.g. FileNotFoundError) returns 'Shell error:'."""
    import subprocess
    with patch("subprocess.run", side_effect=OSError("bad cmd")):
        out = _run_shell("nonexistent_binary")
    assert "Shell error" in out


def test_execute_python_stderr_included():
    """_execute_python captures stderr from the executed code."""
    out = _execute_python("import sys; print('err', file=sys.stderr)")
    assert "err" in out


def test_write_and_run_code_typescript_no_runner():
    """TypeScript maps to .ts but has no runner — returns 'No runner configured'."""
    out = _write_and_run_code("typescript", "const x: number = 1;")
    assert "No runner" in out


def test_web_search_returns_error_on_exception():
    """_web_search returns an error string when DDGS raises."""
    fake_ddgs_instance = MagicMock()
    fake_ddgs_instance.__enter__ = MagicMock(side_effect=RuntimeError("ddgs down"))
    fake_ddgs_instance.__exit__ = MagicMock(return_value=False)
    fake_module = MagicMock()
    fake_module.DDGS = MagicMock(return_value=fake_ddgs_instance)
    with patch.dict(sys.modules, {"duckduckgo_search": fake_module}):
        out = _web_search("test query")
    assert "Search error" in out


def test_read_file_not_found_returns_error():
    """_read_file on a nonexistent path returns a 'Read error:' string."""
    out = _read_file("/nonexistent/path/file_xyz.txt")
    assert "Read error" in out


def test_api_call_json_parse_error_returns_text_response():
    """When response is not JSON, _api_call returns resp.text."""
    fake_requests = MagicMock()
    mock_req = fake_requests.request
    resp = MagicMock()
    resp.json.side_effect = ValueError("not json")
    resp.text = "plain text response"
    mock_req.return_value = resp
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _api_call("https://example.com")
    assert "plain text response" in out


# ── Tool.to_anthropic() format ────────────────────────────────────────────────

def test_tool_to_anthropic_includes_required_keys():
    """Tool.to_anthropic() returns name, description, and input_schema."""
    from jarvis.tools.registry import Tool
    t = Tool(
        name="my_tool",
        description="Does something.",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        fn=lambda x: x,
    )
    d = t.to_anthropic()
    assert d["name"] == "my_tool"
    assert d["description"] == "Does something."
    assert "input_schema" in d


# ── Tool.run() calls fn with kwargs ──────────────────────────────────────────

def test_tool_run_passes_kwargs_to_fn():
    """Tool.run() forwards all keyword arguments to the wrapped function."""
    from jarvis.tools.registry import Tool
    results = []
    def my_fn(x, y):
        results.append((x, y))
        return f"{x}+{y}"
    t = Tool(name="add", description="add", input_schema={}, fn=my_fn)
    out = t.run(x=3, y=4)
    assert out == "3+4"
    assert results == [(3, 4)]


# ── ToolRegistry.anthropic_tools() returns list of dicts ─────────────────────

def test_anthropic_tools_returns_list_of_dicts():
    """anthropic_tools() returns a list with name, description, input_schema keys."""
    from jarvis.tools.registry import Tool, ToolRegistry
    reg = ToolRegistry()
    reg.register(Tool("t1", "desc1", {"type": "object"}, fn=lambda: None))
    reg.register(Tool("t2", "desc2", {"type": "object"}, fn=lambda: None))
    tools = reg.anthropic_tools()
    assert len(tools) == 2
    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "input_schema" in t


# ── ToolRegistry.names() returns registered tool names ────────────────────────

def test_registry_names_returns_all_registered():
    from jarvis.tools.registry import Tool, ToolRegistry
    reg = ToolRegistry()
    reg.register(Tool("alpha", "d", {}, fn=lambda: None))
    reg.register(Tool("beta", "d", {}, fn=lambda: None))
    names = reg.names()
    assert "alpha" in names
    assert "beta" in names
    assert len(names) == 2


# ── _web_search with no results returns the 'No results' message ─────────────

def test_web_search_no_results_message():
    """_web_search returns 'No results found.' when DDGS returns empty list."""
    fake_ddgs_instance = MagicMock()
    fake_ddgs_instance.__enter__ = MagicMock(return_value=fake_ddgs_instance)
    fake_ddgs_instance.__exit__ = MagicMock(return_value=False)
    fake_ddgs_instance.text = MagicMock(return_value=[])
    fake_module = MagicMock()
    fake_module.DDGS = MagicMock(return_value=fake_ddgs_instance)
    with patch.dict(sys.modules, {"duckduckgo_search": fake_module}):
        out = _web_search("obscure query xyz")
    assert out == "No results found."


# ── _run_shell with cwd parameter ────────────────────────────────────────────

def test_run_shell_with_cwd(tmp_path):
    """_run_shell passes cwd to subprocess.run."""
    out = _run_shell("pwd", cwd=str(tmp_path))
    assert str(tmp_path) in out


# ── Tool.dynamic flag defaults to False ──────────────────────────────────────

def test_tool_dynamic_defaults_to_false():
    from jarvis.tools.registry import Tool
    t = Tool("t", "d", {}, fn=lambda: None)
    assert t.dynamic is False


def test_tool_dynamic_can_be_set_true():
    from jarvis.tools.registry import Tool
    t = Tool("t", "d", {}, fn=lambda: None, dynamic=True)
    assert t.dynamic is True


# ── Tool.category default ─────────────────────────────────────────────────────

def test_tool_category_default_is_general():
    from jarvis.tools.registry import Tool
    t = Tool("t", "d", {}, fn=lambda: None)
    assert t.category == "general"


def test_tool_category_custom():
    from jarvis.tools.registry import Tool
    t = Tool("t", "d", {}, fn=lambda: None, category="web")
    assert t.category == "web"


# ── ToolRegistry.all() and get() ─────────────────────────────────────────────

def test_registry_all_returns_registered_tools():
    from jarvis.tools.registry import Tool, ToolRegistry
    reg = ToolRegistry()
    t1 = Tool("a", "d", {}, fn=lambda: None)
    t2 = Tool("b", "d", {}, fn=lambda: None)
    reg.register(t1)
    reg.register(t2)
    all_tools = reg.all()
    assert len(all_tools) == 2
    assert t1 in all_tools
    assert t2 in all_tools


def test_registry_get_returns_none_for_missing():
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    assert reg.get("nonexistent_tool") is None


def test_registry_overwrites_on_same_name():
    from jarvis.tools.registry import Tool, ToolRegistry
    reg = ToolRegistry()
    t1 = Tool("name", "first", {}, fn=lambda: "first")
    t2 = Tool("name", "second", {}, fn=lambda: "second")
    reg.register(t1)
    reg.register(t2)
    assert reg.get("name").description == "second"
    assert len(reg.all()) == 1


# ── _execute_python no output ─────────────────────────────────────────────────

def test_execute_python_no_output_returns_no_output_marker():
    from jarvis.tools.registry import _execute_python
    out = _execute_python("x = 1 + 1")
    assert "(no output)" in out


def test_execute_python_output_and_error():
    from jarvis.tools.registry import _execute_python
    out = _execute_python("print('hello'); import sys; sys.stderr.write('err')")
    assert "hello" in out


# ── _list_directory error handling ───────────────────────────────────────────

def test_list_directory_nonexistent_path_returns_error():
    from jarvis.tools.registry import _list_directory
    out = _list_directory("/nonexistent/path/that/does/not/exist/xyz")
    assert "List error" in out or "error" in out.lower()


# ── _get_system_info error path ───────────────────────────────────────────────

def test_get_system_info_error_returns_system_info_error():
    with patch.dict(sys.modules, {"psutil": None}):
        out = _get_system_info()
    assert "System info error" in out or "error" in out.lower()


def test_get_system_info_with_psutil_returns_cpu():
    fake_psutil = MagicMock()
    fake_psutil.cpu_percent.return_value = 42.0
    mem = MagicMock()
    mem.percent = 55.0
    mem.used = 2 * 1024**2
    mem.total = 8 * 1024**2
    fake_psutil.virtual_memory.return_value = mem
    disk = MagicMock()
    disk.percent = 30.0
    disk.used = 50 * 1024**3
    disk.total = 200 * 1024**3
    fake_psutil.disk_usage.return_value = disk
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        out = _get_system_info()
    assert "CPU:" in out
    assert "Memory:" in out
    assert "Disk:" in out


# ── _api_call: successful JSON vs text response ───────────────────────────────

def test_api_call_returns_json_string():
    fake_resp = MagicMock()
    fake_resp.headers = {"Content-Type": "application/json"}
    fake_resp.json.return_value = {"status": "ok"}
    fake_requests = MagicMock()
    fake_requests.request.return_value = fake_resp
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _api_call("https://api.example.com/status")
    assert "status" in out


def test_api_call_returns_text_when_not_json():
    fake_resp = MagicMock()
    fake_resp.headers = {"Content-Type": "text/plain"}
    fake_resp.json.side_effect = ValueError("not json")
    fake_resp.text = "pong"
    fake_requests = MagicMock()
    fake_requests.request.return_value = fake_resp
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _api_call("https://api.example.com/ping")
    assert "pong" in out


# ── _write_file creates file with correct content ─────────────────────────────

def test_write_file_creates_file(tmp_path):
    from jarvis.tools.registry import _write_file
    p = str(tmp_path / "out.txt")
    out = _write_file(p, "hello world")
    assert "Written" in out or "Wrote" in out
    assert (tmp_path / "out.txt").read_text() == "hello world"


# ── _read_file reads file content ─────────────────────────────────────────────

def test_read_file_returns_content(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("sample content")
    out = _read_file(str(f))
    assert "sample content" in out


# ── _list_directory returns file names ────────────────────────────────────────

def test_list_directory_returns_file_names(tmp_path):
    (tmp_path / "alpha.txt").write_text("a")
    (tmp_path / "beta.txt").write_text("b")
    out = _list_directory(str(tmp_path))
    assert "alpha.txt" in out
    assert "beta.txt" in out
