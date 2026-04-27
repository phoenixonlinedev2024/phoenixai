"""Tests for jarvis.skills — SkillRegistry, Skill dataclass, auto_create."""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Inject fakes before heavy imports ────────────────────────────────────────

def _inject_fakes():
    for name in ("chromadb", "sentence_transformers", "openai", "modal", "anthropic"):
        mod = sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.skills.registry import Skill, SkillRegistry  # noqa: E402
from jarvis.skills.builtin import BUILTIN_SKILLS, load_builtin_skills  # noqa: E402


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    return SkillRegistry()


# ── Skill dataclass ───────────────────────────────────────────────────────────

def test_skill_to_dict_roundtrip():
    s = Skill(name="test", description="d", system_prompt="sp", tags=["t"])
    d = s.to_dict()
    s2 = Skill.from_dict(d)
    assert s2.name == "test"
    assert s2.tags == ["t"]


def test_skill_defaults():
    s = Skill(name="x", description="d", system_prompt="sp")
    assert s.usage_count == 0
    assert s.created_by == "builtin"
    assert s.tags == []


# ── SkillRegistry CRUD ────────────────────────────────────────────────────────

def test_register_and_get(registry):
    s = Skill(name="s1", description="d", system_prompt="sp")
    registry.register(s)
    assert registry.get("s1") is s


def test_get_missing_returns_none(registry):
    assert registry.get("no_such_skill") is None


def test_all_returns_all_registered(registry):
    registry.register(Skill(name="a", description="d", system_prompt="sp"))
    registry.register(Skill(name="b", description="d", system_prompt="sp"))
    names = [s.name for s in registry.all()]
    assert "a" in names and "b" in names


def test_search_by_name(registry):
    registry.register(Skill(name="web_scraper", description="scrape web", system_prompt="sp"))
    results = registry.search("web")
    assert any(s.name == "web_scraper" for s in results)


def test_search_by_description(registry):
    registry.register(Skill(name="x", description="handles PDF extraction", system_prompt="sp"))
    results = registry.search("pdf")
    assert any(s.name == "x" for s in results)


def test_search_by_tag(registry):
    registry.register(Skill(name="ml", description="d", system_prompt="sp", tags=["pytorch"]))
    results = registry.search("pytorch")
    assert any(s.name == "ml" for s in results)


def test_search_no_match_returns_empty(registry):
    assert registry.search("zzz_no_match_xyz") == []


def test_top_returns_sorted_by_usage(registry):
    for name, count in (("low", 1), ("high", 10), ("mid", 5)):
        s = Skill(name=name, description="d", system_prompt="sp", usage_count=count)
        registry.register(s)
    top = registry.top(2)
    assert top[0].name == "high"
    assert top[1].name == "mid"


def test_increment_usage(registry):
    registry.register(Skill(name="u", description="d", system_prompt="sp"))
    registry.increment_usage("u")
    registry.increment_usage("u")
    assert registry.get("u").usage_count == 2


def test_increment_usage_missing_skill_noop(registry):
    registry.increment_usage("ghost")  # should not raise


def test_summary_string(registry):
    registry.register(Skill(name="a", description="d", system_prompt="sp"))
    s = registry.summary()
    assert isinstance(s, str)
    assert "1" in s


# ── Persistence (non-builtin skills saved to JSON) ────────────────────────────

def test_generated_skills_persisted(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    r1 = SkillRegistry()
    s = Skill(name="gen", description="d", system_prompt="sp", created_by="generated")
    r1.register(s)

    # New registry instance reads the same file
    r2 = SkillRegistry()
    assert r2.get("gen") is not None


def test_builtin_skills_not_persisted(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    r = SkillRegistry()
    r.register(Skill(name="b", description="d", system_prompt="sp", created_by="builtin"))
    persist_file = tmp_path / "skills.json"
    if persist_file.exists():
        saved = json.loads(persist_file.read_text())
        assert "b" not in saved


# ── load_builtin_skills ───────────────────────────────────────────────────────

def test_load_builtin_skills_populates_registry(registry):
    load_builtin_skills(registry)
    names = [s.name for s in registry.all()]
    for skill in BUILTIN_SKILLS:
        assert skill.name in names


def test_builtin_skills_count():
    assert len(BUILTIN_SKILLS) >= 8


def test_each_builtin_has_non_empty_system_prompt():
    for s in BUILTIN_SKILLS:
        assert len(s.system_prompt) > 20, f"Skill '{s.name}' has a too-short system_prompt"


# ── auto_create ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_create_registers_skill(registry):
    payload = '{"name":"csv_analyst","description":"analyse CSV","tags":["csv"],"system_prompt":"You are a data expert."}'
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=payload)]
    ))
    skill = await registry.auto_create("analyse a CSV file", client)
    assert skill is not None
    assert skill.name == "csv_analyst"
    assert registry.get("csv_analyst") is not None


@pytest.mark.asyncio
async def test_auto_create_on_error_returns_none(registry):
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))
    skill = await registry.auto_create("some task", client)
    assert skill is None


@pytest.mark.asyncio
async def test_auto_create_strips_backtick_fence(registry):
    """Line 94: raw.split('```')[1] branch when response starts with backticks."""
    payload = '{"name":"fenced","description":"d","tags":[],"system_prompt":"sp"}'
    fenced = f"```json\n{payload}\n```"
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=fenced)]
    ))
    skill = await registry.auto_create("do fenced thing", client)
    assert skill is not None
    assert skill.name == "fenced"


def test_save_swallows_exception(tmp_path, monkeypatch):
    """Lines 109-110: _save() passes silently when the file cannot be written."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    r = SkillRegistry()
    r.register(Skill(name="s1", description="d", system_prompt="sp", created_by="generated"))
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a directory")
    r._persist_path = blocker / "skills.json"
    r._save()  # should not raise despite mkdir failing


def test_load_swallows_corrupt_json(tmp_path, monkeypatch):
    """Lines 119-120: _load() passes silently when the persisted file is corrupt."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    persist = tmp_path / "skills.json"
    persist.write_text("NOT VALID JSON")
    r = SkillRegistry()
    assert r.all() == []


# ── Skill.from_dict() roundtrip ───────────────────────────────────────────────

def test_skill_from_dict_roundtrip():
    from jarvis.skills.registry import Skill
    original = Skill(
        name="code_review",
        description="Review code for quality",
        system_prompt="You are a code review expert.",
        tags=["code", "review"],
        usage_count=7,
        created_by="generated",
    )
    restored = Skill.from_dict(original.to_dict())
    assert restored.name == original.name
    assert restored.description == original.description
    assert restored.system_prompt == original.system_prompt
    assert restored.tags == original.tags
    assert restored.usage_count == original.usage_count
    assert restored.created_by == original.created_by


# ── SkillRegistry.top() edge cases ───────────────────────────────────────────

def test_top_n_larger_than_skill_count_returns_all(registry):
    from jarvis.skills.registry import Skill
    registry.register(Skill("s1", "d", "p", usage_count=3))
    registry.register(Skill("s2", "d", "p", usage_count=1))
    result = registry.top(n=100)
    assert len(result) == 2


def test_top_zero_returns_empty(registry):
    from jarvis.skills.registry import Skill
    registry.register(Skill("s1", "d", "p", usage_count=5))
    result = registry.top(n=0)
    assert result == []


def test_top_respects_n_limit(registry):
    from jarvis.skills.registry import Skill
    for i in range(10):
        registry.register(Skill(f"skill_{i}", "d", "p", usage_count=i))
    result = registry.top(n=3)
    assert len(result) == 3
    # Highest usage should be first
    assert result[0].usage_count >= result[1].usage_count >= result[2].usage_count


# ── SkillRegistry.search() matches all three fields ──────────────────────────

def test_search_matches_name_description_and_tag(registry):
    from jarvis.skills.registry import Skill
    registry.register(Skill("unique_name_xyz", "ordinary desc", "prompt", tags=["ordinary"]))
    registry.register(Skill("ordinary", "unique_desc_xyz", "prompt", tags=["ordinary"]))
    registry.register(Skill("other", "ordinary", "prompt", tags=["unique_tag_xyz"]))

    assert any(s.name == "unique_name_xyz" for s in registry.search("unique_name_xyz"))
    assert any(s.name == "ordinary" for s in registry.search("unique_desc_xyz"))
    assert any(s.name == "other" for s in registry.search("unique_tag_xyz"))


# ── SkillRegistry.summary() includes counts ──────────────────────────────────

def test_summary_counts_correctly(registry):
    from jarvis.skills.registry import Skill
    registry.register(Skill("builtin_s", "d", "p", created_by="builtin"))
    registry.register(Skill("gen_s1", "d", "p", created_by="generated"))
    registry.register(Skill("gen_s2", "d", "p", created_by="generated"))
    summary = registry.summary()
    assert "3 skills" in summary
    assert "2 auto-generated" in summary
