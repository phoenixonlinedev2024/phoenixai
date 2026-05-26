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


# ── SkillRegistry.increment_usage edge cases ─────────────────────────────────

def test_increment_usage_nonexistent_skill_is_silent(registry):
    """increment_usage on a name not in _skills should not raise."""
    registry.increment_usage("does_not_exist")  # should not raise


def test_increment_usage_increments_count(registry):
    """increment_usage bumps usage_count by 1 each call."""
    from jarvis.skills.registry import Skill
    registry.register(Skill("counter", "d", "p"))
    assert registry.get("counter").usage_count == 0
    registry.increment_usage("counter")
    assert registry.get("counter").usage_count == 1
    registry.increment_usage("counter")
    assert registry.get("counter").usage_count == 2


# ── SkillRegistry persistence: generated skills are saved but builtins are not ─

def test_generated_skill_persisted_on_register(tmp_path, monkeypatch):
    """Generated skills are written to the JSON persist file."""
    from jarvis.config import cfg
    from jarvis.skills.registry import Skill, SkillRegistry
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    reg = SkillRegistry()
    reg.register(Skill("gen_skill", "gen desc", "gen prompt", created_by="generated"))
    persist = tmp_path / "skills.json"
    assert persist.exists()
    data = json.loads(persist.read_text())
    assert "gen_skill" in data


def test_builtin_skill_not_persisted(tmp_path, monkeypatch):
    """Builtin skills are NOT written to the JSON persist file."""
    from jarvis.config import cfg
    from jarvis.skills.registry import Skill, SkillRegistry
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    reg = SkillRegistry()
    reg.register(Skill("builtin_tool", "desc", "prompt", created_by="builtin"))
    persist = tmp_path / "skills.json"
    if persist.exists():
        data = json.loads(persist.read_text())
        assert "builtin_tool" not in data


# ── SkillRegistry._load: persisted skills survive reload ─────────────────────

def test_persisted_generated_skill_loads_on_new_registry(tmp_path, monkeypatch):
    """Generated skills saved on one SkillRegistry instance load on a fresh one."""
    from jarvis.config import cfg
    from jarvis.skills.registry import Skill, SkillRegistry
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    r1 = SkillRegistry()
    r1.register(Skill("persistent_gen", "desc", "prompt", created_by="generated"))

    r2 = SkillRegistry()
    assert r2.get("persistent_gen") is not None


# ── SkillRegistry.search empty query returns all ─────────────────────────────

def test_search_empty_query_returns_all(registry):
    """Empty query string matches everything (all fields contain '')."""
    from jarvis.skills.registry import Skill
    registry.register(Skill("alpha", "desc a", "p"))
    registry.register(Skill("beta", "desc b", "p"))
    results = registry.search("")
    names = [s.name for s in results]
    assert "alpha" in names
    assert "beta" in names


# ── Skill.from_dict roundtrip for user-created skill ─────────────────────────

def test_skill_from_dict_user_created():
    """from_dict works with created_by='user'."""
    from jarvis.skills.registry import Skill
    d = {
        "name": "user_skill",
        "description": "user desc",
        "system_prompt": "user prompt",
        "tags": ["custom"],
        "usage_count": 5,
        "created_by": "user",
    }
    skill = Skill.from_dict(d)
    assert skill.created_by == "user"


# ── SkillRegistry.summary() generated count ───────────────────────────────────

def test_summary_generated_count_zero_when_only_builtins(registry):
    """summary() shows 0 auto-generated when only builtin skills are registered."""
    registry.register(Skill("bi", "desc", "prompt", created_by="builtin"))
    s = registry.summary()
    assert "0" in s


def test_summary_generated_count_one_after_generated_skill(registry):
    """summary() shows 1 auto-generated after registering a generated skill."""
    registry.register(Skill("gen_one", "desc", "prompt", created_by="generated"))
    s = registry.summary()
    assert "1" in s


def test_summary_user_skill_not_counted_as_generated(registry):
    """User-created skills are not counted as auto-generated in summary()."""
    registry.register(Skill("user_skill", "desc", "prompt", created_by="user"))
    s = registry.summary()
    # 0 auto-generated
    assert "(0 auto-generated)" in s


def test_summary_total_includes_all_skill_types(registry):
    """summary() total count includes builtin, generated, and user skills."""
    registry.register(Skill("b", "d", "p", created_by="builtin"))
    registry.register(Skill("g", "d", "p", created_by="generated"))
    registry.register(Skill("u", "d", "p", created_by="user"))
    s = registry.summary()
    assert "3 skills" in s


# ── Skill.tags default to empty list ─────────────────────────────────────────

def test_skill_tags_default_to_empty_list():
    skill = Skill("no_tags", "description", "prompt")
    assert skill.tags == []


def test_skill_tags_are_searchable(registry):
    """Search finds a skill by its tag."""
    registry.register(Skill("tagged_skill", "some desc", "some prompt", tags=["mlops", "ai"]))
    results = registry.search("mlops")
    assert any(s.name == "tagged_skill" for s in results)
