"""Data transformation tools — JSON, YAML, TOML, jq-style queries, regex, diff."""

from __future__ import annotations
import difflib
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _jq_query(data: str, query: str) -> str:
    """Apply a jq-style query to JSON data. Uses jq binary or pyjq."""
    try:
        import subprocess
        result = subprocess.run(["jq", query], input=data, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or result.stderr.strip()
    except FileNotFoundError:
        pass
    try:
        import jq as jq_lib
        return json.dumps(jq_lib.first(query, json.loads(data)), indent=2)
    except ImportError:
        pass
    # Pure Python fallback for simple .key access
    try:
        obj = json.loads(data)
        if query.startswith("."):
            for part in query.lstrip(".").split("."):
                if not part:
                    continue
                obj = obj[part] if isinstance(obj, dict) else obj[int(part)]
            return json.dumps(obj, indent=2)
    except Exception:
        pass
    return "jq not available. Install: apt install jq"


def _yaml_to_json(yaml_str: str) -> str:
    try:
        import yaml
        return json.dumps(yaml.safe_load(yaml_str), indent=2, default=str)
    except ImportError:
        return "PyYAML not installed. Run: pip install pyyaml"
    except Exception as exc:
        return f"YAML error: {exc}"


def _json_to_yaml(json_str: str) -> str:
    try:
        import yaml
        return yaml.dump(json.loads(json_str), default_flow_style=False, allow_unicode=True)
    except ImportError:
        return "PyYAML not installed. Run: pip install pyyaml"
    except Exception as exc:
        return f"Conversion error: {exc}"


def _validate_json(json_str: str) -> str:
    try:
        obj = json.loads(json_str)
        return f"Valid JSON. Type: {type(obj).__name__}, Size: {len(json_str)} chars."
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"


def _regex_test(pattern: str, text: str, flags: str = "") -> str:
    flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}
    fl = 0
    for c in flags:
        fl |= flag_map.get(c, 0)
    try:
        matches = list(re.finditer(pattern, text, fl))
        if not matches:
            return "No matches found."
        results = [{"match": m.group(), "start": m.start(), "end": m.end(), "groups": m.groups()} for m in matches[:20]]
        return json.dumps(results, indent=2)
    except re.error as exc:
        return f"Regex error: {exc}"


def _text_diff(text_a: str, text_b: str, context: int = 3) -> str:
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines_a, lines_b, fromfile="before", tofile="after", n=context))
    return "".join(diff)[:6000] or "No differences."


def _file_diff(path_a: str, path_b: str) -> str:
    try:
        a = open(path_a, encoding="utf-8", errors="replace").read()
        b = open(path_b, encoding="utf-8", errors="replace").read()
        return _text_diff(a, b)
    except Exception as exc:
        return f"Diff error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(name="jq_query", description="Apply a jq query to JSON string. E.g. query='.users[0].name'",
        input_schema={"type":"object","properties":{"data":{"type":"string"},"query":{"type":"string"}},"required":["data","query"]},
        fn=_jq_query, category="code"))
    registry.register(Tool(name="yaml_to_json", description="Convert YAML string to JSON.",
        input_schema={"type":"object","properties":{"yaml_str":{"type":"string"}},"required":["yaml_str"]},
        fn=_yaml_to_json, category="code"))
    registry.register(Tool(name="json_to_yaml", description="Convert JSON string to YAML.",
        input_schema={"type":"object","properties":{"json_str":{"type":"string"}},"required":["json_str"]},
        fn=_json_to_yaml, category="code"))
    registry.register(Tool(name="validate_json", description="Validate and describe a JSON string.",
        input_schema={"type":"object","properties":{"json_str":{"type":"string"}},"required":["json_str"]},
        fn=_validate_json, category="code"))
    registry.register(Tool(name="regex_test", description="Test a regex pattern against text. flags: i=ignore case, m=multiline, s=dotall.",
        input_schema={"type":"object","properties":{"pattern":{"type":"string"},"text":{"type":"string"},"flags":{"type":"string","default":""}},"required":["pattern","text"]},
        fn=_regex_test, category="code"))
    registry.register(Tool(name="text_diff", description="Compute unified diff between two text strings.",
        input_schema={"type":"object","properties":{"text_a":{"type":"string"},"text_b":{"type":"string"},"context":{"type":"integer","default":3}},"required":["text_a","text_b"]},
        fn=_text_diff, category="files"))
    registry.register(Tool(name="file_diff", description="Compute unified diff between two files.",
        input_schema={"type":"object","properties":{"path_a":{"type":"string"},"path_b":{"type":"string"}},"required":["path_a","path_b"]},
        fn=_file_diff, category="files"))
