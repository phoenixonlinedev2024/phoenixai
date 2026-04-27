"""Subagent delegation tools — lets JARVIS spawn isolated subagents."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry

_pool = None


def _get_pool(jarvis=None):
    global _pool
    if _pool is None and jarvis is not None:
        from jarvis.agents.subagent import SubagentPool
        _pool = SubagentPool(jarvis)
    return _pool


def _delegate(goal: str, context: str = "", max_turns: int = 8) -> str:
    """Delegate a task to a subagent."""
    pool = _get_pool()
    if not pool:
        return "Subagent pool not initialised."
    try:
        return asyncio.run(pool.dispatch_one(goal, context=context, max_turns=max_turns))
    except Exception as exc:
        return f"Subagent error: {exc}"


def _delegate_parallel(tasks: list) -> str:
    """Dispatch multiple tasks in parallel to subagents."""
    pool = _get_pool()
    if not pool:
        return "Subagent pool not initialised."
    try:
        from jarvis.agents.subagent import SubagentTask
        task_objs = [SubagentTask(goal=t.get("goal", ""), context=t.get("context", "")) for t in tasks]
        results = asyncio.run(pool.dispatch(task_objs))
        return json.dumps({tid: res[:500] for tid, res in results.items()}, indent=2)
    except Exception as exc:
        return f"Parallel subagent error: {exc}"


def register_tools(registry: "ToolRegistry", jarvis=None) -> None:
    from jarvis.tools.registry import Tool

    global _pool
    if jarvis is not None:
        from jarvis.agents.subagent import SubagentPool
        _pool = SubagentPool(jarvis)

    registry.register(Tool(
        name="delegate_task",
        description="Delegate a complex subtask to an isolated subagent with its own context. Returns the subagent's result. Use for long-running or independent tasks that shouldn't consume your context.",
        input_schema={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The task for the subagent to complete"},
                "context": {"type": "string", "description": "Background context to give the subagent"},
                "max_turns": {"type": "integer", "default": 8},
            },
            "required": ["goal"],
        },
        fn=_delegate,
        category="agents",
    ))

    registry.register(Tool(
        name="delegate_parallel",
        description="Dispatch multiple independent tasks to subagents in parallel. Returns all results.",
        input_schema={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "List of {goal, context} objects",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "context": {"type": "string"},
                        },
                        "required": ["goal"],
                    },
                },
            },
            "required": ["tasks"],
        },
        fn=_delegate_parallel,
        category="agents",
    ))
