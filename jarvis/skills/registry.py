"""Skills registry — reusable prompt-based skills JARVIS can invoke or create."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis


@dataclass
class Skill:
    name: str
    description: str
    system_prompt: str
    tags: list[str] = field(default_factory=list)
    usage_count: int = 0
    created_by: str = "builtin"  # builtin | generated | user

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tags": self.tags,
            "usage_count": self.usage_count,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SkillRegistry:
    """Catalog of reusable skills. JARVIS picks the best skill for a task
    or creates a new one when none match.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._persist_path = cfg.DATA_DIR / "skills.json"
        self._load()

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill
        self._save()

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def search(self, query: str) -> list[Skill]:
        q = query.lower()
        return [
            s for s in self._skills.values()
            if q in s.name.lower() or q in s.description.lower()
            or any(q in t for t in s.tags)
        ]

    def top(self, n: int = 5) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.usage_count, reverse=True)[:n]

    def increment_usage(self, name: str) -> None:
        if name in self._skills:
            self._skills[name].usage_count += 1
            self._save()

    async def auto_create(self, task: str, client: Any) -> Skill | None:
        """Ask Claude to create a new skill for a task type."""
        prompt = f"""Create a reusable JARVIS skill for this type of task: {task}

Output ONLY JSON:
{{
  "name": "snake_case_name",
  "description": "One sentence",
  "tags": ["tag1", "tag2"],
  "system_prompt": "A focused system prompt that makes JARVIS excellent at this task type."
}}"""
        try:
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            data = json.loads(raw)
            skill = Skill(created_by="generated", **data)
            self.register(skill)
            return skill
        except Exception as exc:
            print(f"[JARVIS Skills] Auto-create failed: {exc}")
            return None

    def _save(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {name: s.to_dict() for name, s in self._skills.items()
                    if s.created_by != "builtin"}
            self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for d in data.values():
                self._skills[d["name"]] = Skill.from_dict(d)
        except Exception:
            pass

    def summary(self) -> str:
        total = len(self._skills)
        generated = sum(1 for s in self._skills.values() if s.created_by == "generated")
        return f"{total} skills ({generated} auto-generated)"
