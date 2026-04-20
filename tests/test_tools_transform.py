"""Tests for jarvis.tools.transform_tools — JSON, YAML, regex, diff."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


def test_validate_json_valid():
    from jarvis.tools.transform_tools import _validate_json
    result = _validate_json('{"key": "value", "num": 42}')
    assert "Valid" in result
    assert "dict" in result


def test_validate_json_invalid():
    from jarvis.tools.transform_tools import _validate_json
    result = _validate_json('{bad json}')
    assert "Invalid" in result


def test_regex_test_match():
    from jarvis.tools.transform_tools import _regex_test
    result = _regex_test(r"\d+", "hello 42 world 99")
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["match"] == "42"


def test_regex_test_no_match():
    from jarvis.tools.transform_tools import _regex_test
    result = _regex_test(r"\d+", "no numbers here")
    assert "No matches" in result


def test_regex_test_case_insensitive():
    from jarvis.tools.transform_tools import _regex_test
    result = _regex_test("hello", "HELLO world", flags="i")
    data = json.loads(result)
    assert data[0]["match"] == "HELLO"


def test_regex_bad_pattern():
    from jarvis.tools.transform_tools import _regex_test
    result = _regex_test("[unclosed", "text")
    assert "error" in result.lower()


def test_text_diff_identical():
    from jarvis.tools.transform_tools import _text_diff
    result = _text_diff("same text", "same text")
    assert result == "No differences."


def test_text_diff_shows_changes():
    from jarvis.tools.transform_tools import _text_diff
    result = _text_diff("line one\nline two\n", "line one\nline THREE\n")
    assert "-line two" in result or "+line THREE" in result


def test_yaml_to_json():
    from jarvis.tools.transform_tools import _yaml_to_json
    yaml_str = "name: jarvis\nversion: 4"
    result = _yaml_to_json(yaml_str)
    try:
        data = json.loads(result)
        assert data["name"] == "jarvis"
        assert data["version"] == 4
    except ImportError:
        pytest.skip("PyYAML not installed")


def test_json_to_yaml():
    from jarvis.tools.transform_tools import _json_to_yaml
    json_str = '{"name": "jarvis", "version": 4}'
    result = _json_to_yaml(json_str)
    try:
        assert "jarvis" in result
        assert "version" in result
    except ImportError:
        pytest.skip("PyYAML not installed")


def test_jq_query_fallback():
    from jarvis.tools.transform_tools import _jq_query
    data = '{"user": {"name": "Tony"}}'
    result = _jq_query(data, ".user.name")
    # Either jq binary, pyjq, or pure-python fallback should work
    assert "Tony" in result or "jq not available" in result


def test_jq_query_pure_python_nested():
    from jarvis.tools.transform_tools import _jq_query
    import sys
    from unittest.mock import patch
    data = '{"a": {"b": "hello"}}'
    # Force both jq binary and pyjq to be unavailable so pure-python runs
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with patch.dict(sys.modules, {"jq": None}):
            result = _jq_query(data, ".a.b")
    assert "hello" in result


def test_jq_query_unavailable_returns_message():
    from jarvis.tools.transform_tools import _jq_query
    import sys
    from unittest.mock import patch
    data = '{"x": 1}'
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with patch.dict(sys.modules, {"jq": None}):
            result = _jq_query(data, ".nonexistent.deeply.nested.key")
    # Falls through all paths — should not crash
    assert isinstance(result, str)


def test_yaml_to_json_no_pyyaml():
    from jarvis.tools.transform_tools import _yaml_to_json
    with patch.dict(sys.modules, {"yaml": None}):
        result = _yaml_to_json("key: value")
    assert "PyYAML not installed" in result


def test_yaml_to_json_parse_error():
    from jarvis.tools.transform_tools import _yaml_to_json
    fake_yaml = MagicMock()
    fake_yaml.safe_load = MagicMock(side_effect=Exception("bad yaml"))
    with patch.dict(sys.modules, {"yaml": fake_yaml}):
        result = _yaml_to_json(": bad: {")
    assert "YAML error" in result


def test_json_to_yaml_no_pyyaml():
    from jarvis.tools.transform_tools import _json_to_yaml
    with patch.dict(sys.modules, {"yaml": None}):
        result = _json_to_yaml('{"a": 1}')
    assert "PyYAML not installed" in result


def test_json_to_yaml_parse_error():
    from jarvis.tools.transform_tools import _json_to_yaml
    fake_yaml = MagicMock()
    fake_yaml.dump = MagicMock(side_effect=Exception("dump failed"))
    with patch.dict(sys.modules, {"yaml": fake_yaml}):
        result = _json_to_yaml('{"a": 1}')
    assert "Conversion error" in result


def test_file_diff_success(tmp_path):
    from jarvis.tools.transform_tools import _file_diff
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("line1\nline2\n")
    b.write_text("line1\nchanged\n")
    result = _file_diff(str(a), str(b))
    assert "-line2" in result or "+changed" in result


def test_file_diff_identical(tmp_path):
    from jarvis.tools.transform_tools import _file_diff
    a = tmp_path / "same.txt"
    a.write_text("same content\n")
    result = _file_diff(str(a), str(a))
    assert result == "No differences."


def test_file_diff_missing_file():
    from jarvis.tools.transform_tools import _file_diff
    result = _file_diff("/nonexistent/a.txt", "/nonexistent/b.txt")
    assert "Diff error" in result


def test_transform_tools_register():
    from jarvis.tools.transform_tools import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    register_tools(reg)
    names = {t.name for t in reg.all()}
    assert {"jq_query", "yaml_to_json", "json_to_yaml", "validate_json",
            "regex_test", "text_diff", "file_diff"} <= names
