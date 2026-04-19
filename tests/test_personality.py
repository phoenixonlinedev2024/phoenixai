"""Tests for jarvis.personality — profiles, system prompts, constants."""

from __future__ import annotations

from jarvis.personality import (
    JARVIS_CAPABILITY_CREATED_TEMPLATE,
    JARVIS_LEARNING_TEMPLATE,
    JARVIS_SYSTEM_PROMPT,
    JARVIS_VOICE_INTRO,
    JARVIS_WAKE_RESPONSES,
    PERSONALITY_PROFILES,
    get_system_prompt,
)


# ── Profile registry ──────────────────────────────────────────────────────────

def test_all_five_profiles_exist():
    for name in ("default", "professional", "casual", "terse", "verbose"):
        assert name in PERSONALITY_PROFILES


def test_each_profile_is_non_empty_string():
    for name, prompt in PERSONALITY_PROFILES.items():
        assert isinstance(prompt, str)
        assert len(prompt) > 50, f"Profile '{name}' looks too short"


def test_all_profiles_contain_jarvis_identity():
    for name, prompt in PERSONALITY_PROFILES.items():
        assert "JARVIS" in prompt, f"Profile '{name}' missing JARVIS identity"


def test_default_profile_equals_jarvis_system_prompt():
    assert PERSONALITY_PROFILES["default"] == JARVIS_SYSTEM_PROMPT


def test_non_default_profiles_extend_base():
    base = PERSONALITY_PROFILES["default"]
    for name in ("professional", "casual", "terse", "verbose"):
        assert PERSONALITY_PROFILES[name].startswith(base)


# ── get_system_prompt ─────────────────────────────────────────────────────────

def test_get_system_prompt_default():
    prompt = get_system_prompt()
    assert prompt == PERSONALITY_PROFILES["default"]


def test_get_system_prompt_valid_profile():
    for name in PERSONALITY_PROFILES:
        prompt = get_system_prompt(profile=name)
        assert prompt.startswith(PERSONALITY_PROFILES[name])


def test_get_system_prompt_unknown_profile_falls_back_to_default():
    prompt = get_system_prompt(profile="does_not_exist")
    assert prompt == PERSONALITY_PROFILES["default"]


def test_get_system_prompt_voice_mode_appended():
    prompt = get_system_prompt(voice_mode=True)
    assert "Voice Mode" in prompt


def test_get_system_prompt_voice_mode_not_in_default():
    prompt = get_system_prompt(voice_mode=False)
    assert "Voice Mode" not in prompt


def test_get_system_prompt_memory_context_included():
    ctx = "User prefers Python. Dislikes Java."
    prompt = get_system_prompt(memory_context=ctx)
    assert ctx in prompt
    assert "Current Memory" in prompt


def test_get_system_prompt_empty_memory_context_not_injected():
    prompt = get_system_prompt(memory_context="")
    assert "Current Memory" not in prompt


def test_get_system_prompt_gap_context_included():
    gap = "Missing: SQL query tool."
    prompt = get_system_prompt(gap_context=gap)
    assert gap in prompt
    assert "Capability Gaps" in prompt


def test_get_system_prompt_empty_gap_context_not_injected():
    prompt = get_system_prompt(gap_context="")
    assert "Capability Gaps" not in prompt


def test_get_system_prompt_all_options_combined():
    prompt = get_system_prompt(
        profile="terse",
        voice_mode=True,
        memory_context="fact A",
        gap_context="gap B",
    )
    assert PERSONALITY_PROFILES["terse"] in prompt
    assert "Voice Mode" in prompt
    assert "fact A" in prompt
    assert "gap B" in prompt


# ── Constants ─────────────────────────────────────────────────────────────────

def test_voice_intro_is_non_empty():
    assert isinstance(JARVIS_VOICE_INTRO, str)
    assert len(JARVIS_VOICE_INTRO) > 10


def test_wake_responses_is_list_of_strings():
    assert isinstance(JARVIS_WAKE_RESPONSES, list)
    assert len(JARVIS_WAKE_RESPONSES) >= 3
    for r in JARVIS_WAKE_RESPONSES:
        assert isinstance(r, str)


def test_capability_created_template_has_placeholder():
    result = JARVIS_CAPABILITY_CREATED_TEMPLATE.format(name="my_tool")
    assert "my_tool" in result


def test_learning_template_has_count_placeholder():
    result = JARVIS_LEARNING_TEMPLATE.format(count=5)
    assert "5" in result
