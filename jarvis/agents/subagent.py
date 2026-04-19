"""Subagent system — isolated agents with own conversation context, zero cost to parent."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.core import Jarvis


@dataclass
class SubagentTask:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    goal: str = ""
    context: str = ""
    max_turns: int = 10
    sandbox_backend: str | None = None
    result: str | None = None
    error: str | None = None
    done: bool = False


class SubAgent:
    """An isolated JARVIS instance with its own session, memory-free from parent."""

    def __init__(self, task: SubagentTask, parent: "Jarvis") -> None:
        self.task = task
        self.parent = parent
        self._messages: list[dict] = []

    async def run(self) -> str:
        """Execute the task autonomously and return the result."""
        from jarvis.personality import get_system_prompt

        system = get_system_prompt(profile="terse") + (
            f"\n\n## Subagent Task\nYou are an isolated subagent. Complete this specific goal autonomously:\n{self.task.goal}"
        )
        if self.task.context:
            system += f"\n\n## Context from Parent\n{self.task.context}"

        self._messages = [{"role": "user", "content": f"Begin task: {self.task.goal}"}]

        for _ in range(self.task.max_turns):
            response = await self.parent.client.messages.create(
                model=self.parent._get_model(),
                max_tokens=4096,
                system=system,
                tools=self.parent.registry.anthropic_tools(),
                messages=self._messages,
            )

            text_parts = []
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

            if not tool_calls:
                return " ".join(text_parts).strip()

            self._messages.append({"role": "assistant", "content": response.content})
            tool_results = await self.parent._execute_tools_parallel(tool_calls)
            self._messages.append({"role": "user", "content": tool_results})

        return "Subagent reached max turns without completing task."


class SubagentPool:
    """Manages a pool of subagents running concurrently."""

    def __init__(self, parent: "Jarvis", max_concurrent: int = 5) -> None:
        self.parent = parent
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._results: dict[str, str] = {}

    async def dispatch(self, tasks: list[SubagentTask]) -> dict[str, str]:
        """Dispatch multiple tasks in parallel, return {task_id: result}."""
        async def _run_one(task: SubagentTask) -> tuple[str, str]:
            async with self._semaphore:
                try:
                    agent = SubAgent(task, self.parent)
                    result = await agent.run()
                    task.result = result
                    task.done = True
                    return task.id, result
                except Exception as exc:
                    task.error = str(exc)
                    task.done = True
                    return task.id, f"Error: {exc}"

        pairs = await asyncio.gather(*[_run_one(t) for t in tasks])
        return dict(pairs)

    async def dispatch_one(self, goal: str, context: str = "", max_turns: int = 10) -> str:
        task = SubagentTask(goal=goal, context=context, max_turns=max_turns)
        results = await self.dispatch([task])
        return results[task.id]
