"""Task planner — decomposition, parallel execution, confidence scoring, A/B testing."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from jarvis.config import cfg

_DECOMPOSE_PROMPT = """\
You are JARVIS's task planning subsystem.
Given a user request, decompose it into an ordered list of atomic subtasks.
Each subtask should be independently executable.
Mark subtasks that can run in parallel with the same "group" number.

Output ONLY JSON:
{
  "subtasks": [
    {"id": 1, "group": 1, "description": "..."},
    {"id": 2, "group": 1, "description": "..."},
    {"id": 3, "group": 2, "description": "..."}
  ],
  "requires_decomposition": true
}

If the task is simple (single step), set requires_decomposition to false and return one subtask.
"""

_CONFIDENCE_PROMPT = """\
Rate the quality of the following AI response on a scale of 0.0 to 1.0.
Consider: accuracy, completeness, relevance, and actionability.
Output ONLY a JSON object: {"score": 0.85, "reason": "one sentence"}
"""

_AB_JUDGE_PROMPT = """\
You are an impartial judge comparing two AI responses (A and B) to the same task.
Pick the better one. Output ONLY JSON: {"winner": "A" or "B", "score_a": 0.0-1.0, "score_b": 0.0-1.0, "reason": "one sentence"}
"""


class TaskPlanner:
    def __init__(self, client: Any, memory: Any) -> None:
        self.client = client
        self.memory = memory

    async def decompose(self, task: str) -> list[dict]:
        """Break a complex task into parallel-aware subtasks."""
        try:
            resp = await self.client.messages.create(
                model=cfg.CLAUDE_MODEL,
                max_tokens=1024,
                system=_DECOMPOSE_PROMPT,
                messages=[{"role": "user", "content": task}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            if not data.get("requires_decomposition", True):
                return [{"id": 1, "group": 1, "description": task}]
            return data.get("subtasks", [{"id": 1, "group": 1, "description": task}])
        except Exception:
            return [{"id": 1, "group": 1, "description": task}]

    async def score_confidence(self, task: str, response: str) -> float:
        """Return a 0–1 confidence score for a response."""
        try:
            resp = await self.client.messages.create(
                model="claude-haiku-4-5-20251001",  # Use fast/cheap model for scoring
                max_tokens=128,
                system=_CONFIDENCE_PROMPT,
                messages=[{"role": "user", "content": f"Task: {task}\n\nResponse: {response[:2000]}"}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            return float(json.loads(raw).get("score", 0.7))
        except Exception:
            return 0.7

    async def ab_test(self, task: str, response_a: str, response_b: str) -> tuple[str, dict]:
        """Judge two responses and return the winner + scores."""
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:12]
        # Check if we've seen this task type before
        cached = self.memory.get_ab_winner(task_hash)
        if cached:
            return cached, {}

        try:
            prompt = f"Task: {task}\n\nResponse A:\n{response_a[:2000]}\n\nResponse B:\n{response_b[:2000]}"
            resp = await self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=128,
                system=_AB_JUDGE_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            result = json.loads(raw)
            winner = result.get("winner", "A")
            self.memory.record_ab_result(
                task_hash, response_a[:200], response_b[:200],
                winner, result.get("score_a", 0.5), result.get("score_b", 0.5),
            )
            return winner, result
        except Exception:
            return "A", {}

    def group_subtasks(self, subtasks: list[dict]) -> list[list[dict]]:
        """Group subtasks by their group number for parallel execution."""
        groups: dict[int, list[dict]] = {}
        for st in subtasks:
            g = st.get("group", 1)
            groups.setdefault(g, []).append(st)
        return [groups[k] for k in sorted(groups.keys())]
