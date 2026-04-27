"""Natural-language cron scheduling — converts phrases to cron expressions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry

# Simple pattern-based NL → cron (fast, no LLM call needed for common phrases)
_NL_MAP = {
    r"every minute": "* * * * *",
    r"every (\d+) minutes?": lambda m: f"*/{m.group(1)} * * * *",
    r"every hour": "0 * * * *",
    r"every (\d+) hours?": lambda m: f"0 */{m.group(1)} * * *",
    r"every day at (\d+)(?::(\d+))?": lambda m: f"{m.group(2) or 0} {m.group(1)} * * *",
    r"daily at (\d+)(?::(\d+))?": lambda m: f"{m.group(2) or 0} {m.group(1)} * * *",
    r"every morning": "0 8 * * *",
    r"every evening": "0 18 * * *",
    r"every night": "0 22 * * *",
    r"every monday": "0 9 * * 1",
    r"every tuesday": "0 9 * * 2",
    r"every wednesday": "0 9 * * 3",
    r"every thursday": "0 9 * * 4",
    r"every friday": "0 9 * * 5",
    r"every weekend": "0 9 * * 6,0",
    r"every weekday": "0 9 * * 1-5",
    r"every week": "0 9 * * 1",
    r"every month": "0 9 1 * *",
    r"midnight": "0 0 * * *",
    r"noon": "0 12 * * *",
}


def _nl_to_cron(phrase: str) -> str:
    """Convert a natural language scheduling phrase to a cron expression."""
    phrase_lower = phrase.lower().strip()
    for pattern, result in _NL_MAP.items():
        m = re.search(pattern, phrase_lower)
        if m:
            if callable(result):
                return result(m)
            return result
    return f"Could not parse '{phrase}'. Provide a cron expression (e.g. '0 9 * * 1')."


def _validate_cron(expression: str) -> str:
    """Validate a cron expression and describe what it means."""
    parts = expression.strip().split()
    if len(parts) != 5:
        return f"Invalid cron: must have 5 parts (minute hour day month weekday), got {len(parts)}."
    descriptions = {
        0: "minute", 1: "hour", 2: "day of month", 3: "month", 4: "day of week"
    }
    issues = []
    ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
    for i, (part, (lo, hi)) in enumerate(zip(parts, ranges)):
        if part == "*":
            continue
        try:
            val = int(part.split("/")[-1].split(",")[0].split("-")[0])
            if not (lo <= val <= hi):
                issues.append(f"{descriptions[i]} value {val} out of range [{lo},{hi}]")
        except ValueError:
            pass
    if issues:
        return "Cron issues: " + "; ".join(issues)
    return f"Valid cron: {expression}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool

    registry.register(Tool(
        name="nl_to_cron",
        description="Convert a natural language scheduling phrase to a cron expression. E.g. 'every morning' → '0 8 * * *'.",
        input_schema={
            "type": "object",
            "properties": {
                "phrase": {"type": "string", "description": "Natural language schedule (e.g. 'every Monday at 9am')"},
            },
            "required": ["phrase"],
        },
        fn=lambda phrase: _nl_to_cron(phrase),
        category="system",
    ))

    registry.register(Tool(
        name="validate_cron",
        description="Validate a cron expression and describe when it will fire.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
        fn=lambda expression: _validate_cron(expression),
        category="system",
    ))
