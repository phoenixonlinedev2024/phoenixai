"""Self-learning engine — JARVIS reflects on sessions and improves."""

from __future__ import annotations

import json
from typing import Any

from jarvis.config import cfg
from jarvis.memory.store import MemoryStore

_REFLECTION_PROMPT = """\
You are JARVIS's self-improvement subsystem.
You are given a session transcript and task outcomes.
Analyse them and extract:
1. Lessons learned (what worked, what failed, how to do better next time)
2. New facts to remember (user preferences, environment details, domain knowledge)
3. Capability gaps (things the user needed that JARVIS could not do well)

Respond ONLY with JSON:
{
  "lessons": ["lesson 1", "lesson 2"],
  "facts": {"key": "value"},
  "capability_gaps": ["description of gap 1"]
}
"""


class LearningEngine:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    async def reflect(self, session_transcript: list[dict], client: Any) -> dict:
        """Run a reflection pass over a completed session."""
        if not cfg.LEARNING_ENABLED:
            return {}

        transcript_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content'][:500]}"
            for msg in session_transcript[-30:]  # last 30 turns
        )

        try:
            response = await client.messages.create(
                model=cfg.CLAUDE_MODEL,
                max_tokens=1024,
                system=_REFLECTION_PROMPT,
                messages=[{"role": "user", "content": transcript_text}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
        except Exception as exc:
            print(f"[JARVIS] Reflection error: {exc}")
            return {}

        # Persist lessons
        for lesson in result.get("lessons", []):
            self.memory.store_lesson(lesson, context="session_reflection")

        # Persist facts
        for key, value in result.get("facts", {}).items():
            self.memory.store_fact(key, value, source="reflection", confidence=0.8)

        return result

    def build_context_prompt(self) -> str:
        """Build a short memory context to prepend to each conversation."""
        facts = self.memory.all_facts()
        lessons = self.memory.get_lessons(limit=10)

        parts = []
        if facts:
            fact_lines = [f"  - {f['key']}: {f['value']}" for f in facts[:20]]
            parts.append("## Known Facts\n" + "\n".join(fact_lines))
        if lessons:
            lesson_lines = [f"  - {lesson}" for lesson in lessons]
            parts.append("## Lessons Learned\n" + "\n".join(lesson_lines))

        if not parts:
            return ""
        return "\n\n".join(parts)
