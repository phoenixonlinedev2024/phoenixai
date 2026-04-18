"""Tests for jarvis.tools.registry — tool registration and execution."""

import pytest
from jarvis.tools.registry import Tool, ToolRegistry


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
