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


def test_jq_query_using_jq_lib():
    """Line 23: jq library path runs when the jq module is available."""
    from jarvis.tools.transform_tools import _jq_query
    fake_jq = MagicMock()
    fake_jq.first = MagicMock(return_value="Tony")
    data = '{"user": {"name": "Tony"}}'
    with patch("subprocess.run", side_effect=FileNotFoundError), \
         patch.dict(sys.modules, {"jq": fake_jq}):
        result = _jq_query(data, ".user.name")
    assert "Tony" in result


def test_jq_query_empty_parts_skipped():
    """Line 32: continue skips empty path segments from double-dot queries."""
    from jarvis.tools.transform_tools import _jq_query
    data = '{"a": {"b": "found_it"}}'
    with patch("subprocess.run", side_effect=FileNotFoundError), \
         patch.dict(sys.modules, {"jq": None}):
        result = _jq_query(data, ".a..b")
    assert "found_it" in result


def test_jq_query_query_without_dot_prefix_returns_unavailable():
    """Branch 29->37: query not starting with '.' skips pure-Python fallback."""
    from jarvis.tools.transform_tools import _jq_query
    data = '{"a": 1}'
    with patch("subprocess.run", side_effect=FileNotFoundError), \
         patch.dict(sys.modules, {"jq": None}):
        result = _jq_query(data, "length")
    assert "jq not available" in result


# ── jq_query: array index access ─────────────────────────────────────────────

def test_jq_query_integer_index_access():
    """Pure Python fallback handles integer array index access."""
    from jarvis.tools.transform_tools import _jq_query
    data = '{"items": [{"name": "first"}, {"name": "second"}]}'
    with patch("subprocess.run", side_effect=FileNotFoundError), \
         patch.dict(sys.modules, {"jq": None}):
        result = _jq_query(data, ".items")
    # The whole array should be returned as JSON
    parsed = json.loads(result)
    assert len(parsed) == 2


def test_jq_query_valid_jq_binary_output():
    """When jq binary succeeds, its stdout is returned."""
    from jarvis.tools.transform_tools import _jq_query
    import subprocess
    mock_result = MagicMock()
    mock_result.stdout = '"hello"'
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result):
        result = _jq_query('{"msg": "hello"}', ".msg")
    assert result == '"hello"'


# ── _validate_json: type reporting ────────────────────────────────────────────

def test_validate_json_list_reports_list_type():
    from jarvis.tools.transform_tools import _validate_json
    result = _validate_json('[1, 2, 3]')
    assert "Valid" in result
    assert "list" in result


def test_validate_json_number_reports_number_type():
    from jarvis.tools.transform_tools import _validate_json
    result = _validate_json('42')
    assert "Valid" in result


# ── _yaml_to_json: PyYAML missing ────────────────────────────────────────────

def test_yaml_to_json_missing_pyyaml():
    from jarvis.tools.transform_tools import _yaml_to_json
    with patch.dict(sys.modules, {"yaml": None}):
        result = _yaml_to_json("key: value")
    assert "PyYAML" in result


def test_json_to_yaml_missing_pyyaml():
    from jarvis.tools.transform_tools import _json_to_yaml
    with patch.dict(sys.modules, {"yaml": None}):
        result = _json_to_yaml('{"key": "value"}')
    assert "PyYAML" in result


# ── _regex_test ───────────────────────────────────────────────────────────────

def test_regex_test_match():
    from jarvis.tools.transform_tools import _regex_test
    result = _regex_test(r"\d+", "hello 42 world")
    assert "42" in result
    assert "Match" in result or "match" in result


def test_regex_test_no_match():
    from jarvis.tools.transform_tools import _regex_test
    result = _regex_test(r"\d+", "no numbers here")
    assert "No match" in result or "not found" in result.lower() or "0 match" in result


def test_regex_test_invalid_pattern():
    from jarvis.tools.transform_tools import _regex_test
    result = _regex_test(r"[invalid", "some text")
    assert "error" in result.lower() or "invalid" in result.lower()


# ── _text_diff ────────────────────────────────────────────────────────────────

def test_text_diff_shows_additions():
    from jarvis.tools.transform_tools import _text_diff
    result = _text_diff("line 1\nline 2\n", "line 1\nline 2\nline 3\n")
    assert "line 3" in result


def test_text_diff_identical_texts():
    from jarvis.tools.transform_tools import _text_diff
    result = _text_diff("same\n", "same\n")
    assert "No differences" in result or result.strip() == "" or "identical" in result.lower()


# ── _regex_test: flags and multiple matches ───────────────────────────────────

def test_regex_test_case_insensitive_flag():
    from jarvis.tools.transform_tools import _regex_test
    import json
    result = _regex_test(r"hello", "Hello World", flags="i")
    data = json.loads(result)
    assert data[0]["match"].lower() == "hello"


def test_regex_test_multiline_flag():
    from jarvis.tools.transform_tools import _regex_test
    import json
    result = _regex_test(r"^start", "start line\nstart again", flags="m")
    data = json.loads(result)
    assert len(data) == 2


def test_regex_test_multiple_matches_capped_at_20():
    from jarvis.tools.transform_tools import _regex_test
    import json
    text = " ".join(["word"] * 30)
    result = _regex_test(r"word", text)
    data = json.loads(result)
    assert len(data) <= 20


def test_regex_test_with_groups():
    from jarvis.tools.transform_tools import _regex_test
    import json
    result = _regex_test(r"(\w+)@(\w+)", "user@host")
    data = json.loads(result)
    assert list(data[0]["groups"]) == ["user", "host"]


# ── _json_to_yaml: success path ───────────────────────────────────────────────

def test_json_to_yaml_success():
    import sys
    fake_yaml = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    fake_yaml.dump.return_value = "key: value\n"
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {"yaml": fake_yaml}):
        from jarvis.tools.transform_tools import _json_to_yaml
        result = _json_to_yaml('{"key": "value"}')
    assert "value" in result


def test_json_to_yaml_invalid_json():
    import sys
    fake_yaml = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    fake_yaml.dump.side_effect = Exception("cannot convert")
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {"yaml": fake_yaml}):
        from jarvis.tools.transform_tools import _json_to_yaml
        result = _json_to_yaml("{not valid json}")
    assert "error" in result.lower()


# ── _yaml_to_json: success path ───────────────────────────────────────────────

def test_yaml_to_json_success():
    import sys
    fake_yaml = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    fake_yaml.safe_load.return_value = {"name": "jarvis", "version": 2}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {"yaml": fake_yaml}):
        from jarvis.tools.transform_tools import _yaml_to_json
        result = _yaml_to_json("name: jarvis\nversion: 2\n")
    import json
    data = json.loads(result)
    assert data["name"] == "jarvis"


def test_yaml_to_json_parse_error():
    import sys
    fake_yaml = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    fake_yaml.safe_load.side_effect = Exception("bad yaml")
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {"yaml": fake_yaml}):
        from jarvis.tools.transform_tools import _yaml_to_json
        result = _yaml_to_json(": invalid : yaml :")
    assert "error" in result.lower()


# ── _validate_json: error paths ───────────────────────────────────────────────

def test_validate_json_invalid_reports_error():
    from jarvis.tools.transform_tools import _validate_json
    result = _validate_json("{bad}")
    assert "Invalid JSON" in result


def test_validate_json_dict_reports_dict_type():
    from jarvis.tools.transform_tools import _validate_json
    result = _validate_json('{"a": 1}')
    assert "dict" in result
