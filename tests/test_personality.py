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


# ── Profile content verification ──────────────────────────────────────────────

def test_professional_profile_contains_formal_guidance():
    p = PERSONALITY_PROFILES["professional"]
    assert "formal" in p.lower() or "Professional" in p


def test_casual_profile_contains_casual_guidance():
    p = PERSONALITY_PROFILES["casual"]
    assert "friendly" in p.lower() or "Casual" in p


def test_terse_profile_contains_brevity_guidance():
    p = PERSONALITY_PROFILES["terse"]
    assert "few words" in p.lower() or "Terse" in p


def test_verbose_profile_contains_detail_guidance():
    p = PERSONALITY_PROFILES["verbose"]
    assert "detail" in p.lower() or "Verbose" in p


def test_all_profiles_contain_base_identity():
    """Every profile includes JARVIS identity text."""
    for name, profile in PERSONALITY_PROFILES.items():
        assert "JARVIS" in profile, f"Profile '{name}' missing JARVIS identity"


# ── get_system_prompt combinations ────────────────────────────────────────────

def test_get_system_prompt_voice_mode_adds_voice_section():
    prompt = get_system_prompt(voice_mode=True)
    assert "Voice Mode" in prompt or "spoken aloud" in prompt


def test_get_system_prompt_non_voice_no_voice_section():
    prompt = get_system_prompt(voice_mode=False)
    assert "spoken aloud" not in prompt


def test_get_system_prompt_unknown_profile_falls_back_to_default():
    default_prompt = get_system_prompt(profile="default")
    unknown_prompt = get_system_prompt(profile="nonexistent_profile")
    assert default_prompt == unknown_prompt


def test_get_system_prompt_memory_context_section_label():
    prompt = get_system_prompt(memory_context="user prefers dark mode")
    assert "Your Current Memory" in prompt
    assert "dark mode" in prompt


def test_get_system_prompt_gap_context_section_label():
    prompt = get_system_prompt(gap_context="need PDF tool")
    assert "Capability Gaps" in prompt
    assert "PDF tool" in prompt


# ── JARVIS_SYSTEM_PROMPT is the default profile ────────────────────────────────

def test_jarvis_system_prompt_is_default():
    assert JARVIS_SYSTEM_PROMPT == PERSONALITY_PROFILES["default"]


# ── Additional personality constant tests ─────────────────────────────────────

def test_personality_profiles_is_dict():
    assert isinstance(PERSONALITY_PROFILES, dict)


def test_personality_profiles_count_is_five():
    assert len(PERSONALITY_PROFILES) == 5


def test_all_profile_keys_are_lowercase():
    for key in PERSONALITY_PROFILES:
        assert key == key.lower(), f"Profile key '{key}' is not lowercase"


def test_capability_created_template_has_name_placeholder():
    assert "{name}" in JARVIS_CAPABILITY_CREATED_TEMPLATE


def test_learning_template_has_count_placeholder():
    assert "{count}" in JARVIS_LEARNING_TEMPLATE


def test_capability_created_template_is_string():
    assert isinstance(JARVIS_CAPABILITY_CREATED_TEMPLATE, str)
    assert len(JARVIS_CAPABILITY_CREATED_TEMPLATE) > 10


def test_jarvis_wake_responses_all_non_empty():
    for response in JARVIS_WAKE_RESPONSES:
        assert len(response) > 0, "Wake response should not be empty"


def test_jarvis_wake_responses_at_least_three():
    assert len(JARVIS_WAKE_RESPONSES) >= 3


def test_jarvis_voice_intro_contains_jarvis():
    assert "JARVIS" in JARVIS_VOICE_INTRO


def test_get_system_prompt_no_duplicate_newlines_at_start():
    prompt = get_system_prompt()
    assert not prompt.startswith("\n\n")


def test_get_system_prompt_gap_and_memory_both_included():
    prompt = get_system_prompt(
        memory_context="user is an engineer",
        gap_context="no OCR tool",
    )
    assert "engineer" in prompt
    assert "OCR tool" in prompt
    assert "Your Current Memory" in prompt
    assert "Capability Gaps" in prompt
