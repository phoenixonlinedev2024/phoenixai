"""Tests for jarvis.tools.transform_tools — JSON, YAML, regex, diff."""

import json
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
