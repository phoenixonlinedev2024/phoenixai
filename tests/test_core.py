"""Tests for jarvis.core — Jarvis agent: profiles, skills, sessions, memory."""

from __future__ import annotations

import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module-level: inject fake heavy dependencies before jarvis.core is imported.
# This mirrors how CI runs (no anthropic/chromadb/openai installed).
# ---------------------------------------------------------------------------

def _inject_fakes():
    fakes = {
        "anthropic": MagicMock(),
        "chromadb": MagicMock(),
        "sentence_transformers": MagicMock(),
        "openai": MagicMock(),
        "modal": MagicMock(),
    }
    fakes["anthropic"].AsyncAnthropic = MagicMock
    for name, mod in fakes.items():
        sys.modules.setdefault(name, mod)


_inject_fakes()


@pytest.fixture()
def jarvis_instance(tmp_data_dir, monkeypatch):
    """Jarvis with all network/IO patched out."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setattr("jarvis.config.cfg.TRAJECTORY_COLLECTION", False)

    with patch("jarvis.plugins.loader.PluginLoader") as MockLoader:
        MockLoader.return_value.load_all.return_value = 0
        MockLoader.return_value.start_hot_reload.return_value = None
        from jarvis.core import Jarvis
        return Jarvis()


# ── Personality profiles ──────────────────────────────────────────────────

def test_default_profile(jarvis_instance):
    assert jarvis_instance._profile == "default"


def test_set_valid_profile(jarvis_instance):
    jarvis_instance.set_profile("terse")
    assert jarvis_instance._profile == "terse"


def test_set_all_profiles(jarvis_instance):
    for profile in ("default", "professional", "casual", "terse", "verbose"):
        jarvis_instance.set_profile(profile)
        assert jarvis_instance._profile == profile


def test_set_invalid_profile_keeps_current(jarvis_instance):
    jarvis_instance.set_profile("terse")
    jarvis_instance.set_profile("nonexistent_profile")
    assert jarvis_instance._profile == "terse"


# ── Skill activation ──────────────────────────────────────────────────────

def test_activate_unknown_skill_returns_false(jarvis_instance):
    assert jarvis_instance.activate_skill("does_not_exist") is False


def test_activate_builtin_skill(jarvis_instance):
    # Built-in skills are loaded; pick the first one
    skills = jarvis_instance.skills.all()
    if not skills:
        pytest.skip("No built-in skills loaded")
    ok = jarvis_instance.activate_skill(skills[0].name)
    assert ok is True
    assert jarvis_instance._active_skill == skills[0].name


def test_deactivate_skill(jarvis_instance):
    skills = jarvis_instance.skills.all()
    if not skills:
        pytest.skip("No built-in skills loaded")
    jarvis_instance.activate_skill(skills[0].name)
    jarvis_instance.deactivate_skill()
    assert jarvis_instance._active_skill is None


# ── Session management ────────────────────────────────────────────────────

def test_session_id_is_uuid(jarvis_instance):
    sid = jarvis_instance._session_id
    parsed = uuid.UUID(sid)
    assert str(parsed) == sid


def test_new_session_changes_id(jarvis_instance):
    old_id = jarvis_instance._session_id
    jarvis_instance.new_session()
    assert jarvis_instance._session_id != old_id


def test_new_session_clears_transcript(jarvis_instance):
    jarvis_instance._session_transcript.append({"role": "user", "content": "hi"})
    jarvis_instance.new_session()
    assert jarvis_instance._session_transcript == []


# ── Status ────────────────────────────────────────────────────────────────

def test_status_returns_string(jarvis_instance):
    status = jarvis_instance.status()
    assert isinstance(status, str)
    assert len(status) > 0


def test_status_mentions_jarvis(jarvis_instance):
    status = jarvis_instance.status().upper()
    assert "JARVIS" in status or "TOOLS" in status or "FACTS" in status


# ── Memory integration ────────────────────────────────────────────────────

def test_memory_saves_message(jarvis_instance):
    sid = jarvis_instance._session_id
    jarvis_instance.memory.save_message(sid, "user", "hello from test")
    history = jarvis_instance.memory.get_history(sid, limit=5)
    assert any(m.get("content") == "hello from test" for m in history)


def test_memory_fact_round_trip(jarvis_instance):
    jarvis_instance.memory.store_fact("test_key", "test_value")
    facts = {f["key"]: f["value"] for f in jarvis_instance.memory.all_facts()}
    assert facts.get("test_key") == "test_value"


# ── Streaming (mocked client) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_chat_yields_tokens(jarvis_instance, monkeypatch):
    """stream_chat should yield all text tokens and save to memory."""
    # Create event classes whose type().__name__ matches what _stream_agent_loop checks
    RawDeltaEvent = type("RawContentBlockDeltaEvent", (), {})

    def make_text_event(text):
        evt = RawDeltaEvent()
        evt.delta = type("Delta", (), {"text": text, "partial_json": None})()
        return evt

    class FakeStreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for text in ("Hello", ", ", "Sir."):
                yield make_text_event(text)

        async def get_final_message(self):
            return MagicMock(content=[])

    jarvis_instance.client.messages.stream = MagicMock(return_value=FakeStreamCtx())
    jarvis_instance.semantic.store_conversation_snippet = MagicMock()
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")

    tokens = []
    async for token in jarvis_instance.stream_chat("say hello"):
        tokens.append(token)

    assert len(tokens) == 3
    assert "".join(tokens) == "Hello, Sir."


@pytest.mark.asyncio
async def test_chat_saves_to_memory(jarvis_instance, monkeypatch):
    """chat() should store both user and assistant messages in memory."""
    async def fake_loop(msg, voice):
        return "Certainly, Sir."

    monkeypatch.setattr(jarvis_instance, "_run_agent_loop", fake_loop)
    jarvis_instance.semantic.store_conversation_snippet = MagicMock()
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")

    await jarvis_instance.chat("test message")

    sid = jarvis_instance._session_id
    history = jarvis_instance.memory.get_history(sid, limit=10)
    contents = [m["content"] for m in history]
    assert "test message" in contents
    assert "Certainly, Sir." in contents


# ── Additional coverage ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_chat_delegates_to_chat(jarvis_instance, monkeypatch):
    called_with = []

    async def fake_chat(msg, voice_mode=False):
        called_with.append((msg, voice_mode))
        return "voice reply"

    monkeypatch.setattr(jarvis_instance, "chat", fake_chat)
    result = await jarvis_instance.voice_chat("wake word detected")
    assert result == "voice reply"
    assert called_with == [("wake word detected", True)]


@pytest.mark.asyncio
async def test_end_session_calls_reflect_and_resets(jarvis_instance, monkeypatch):
    old_sid = jarvis_instance._session_id
    jarvis_instance._session_transcript = [{"role": "user", "content": "hello"}]

    reflect_called = []

    async def fake_reflect(transcript, client):
        reflect_called.append(transcript)
        return {"lessons": ["always test your code"]}

    monkeypatch.setattr(jarvis_instance.learner, "reflect", fake_reflect)
    jarvis_instance.semantic.store_lesson = MagicMock()

    summary = await jarvis_instance.end_session()

    assert len(reflect_called) == 1
    assert jarvis_instance._session_id != old_sid  # new_session() was called
    assert jarvis_instance._session_transcript == []
    # Should mention the lesson count
    assert "1" in summary or "lesson" in summary.lower()


@pytest.mark.asyncio
async def test_end_session_no_lessons_returns_default(jarvis_instance, monkeypatch):
    async def fake_reflect(transcript, client):
        return {}

    monkeypatch.setattr(jarvis_instance.learner, "reflect", fake_reflect)
    summary = await jarvis_instance.end_session()
    assert "Session ended" in summary or "Sir" in summary


def test_build_system_includes_active_skill(jarvis_instance):
    """System prompt should include the skill's prompt when a skill is active."""
    from jarvis.skills.registry import Skill
    skill = Skill(name="test_skill", description="d", system_prompt="## Special Mode\nBe extra helpful.")
    jarvis_instance.skills.register(skill)
    jarvis_instance.activate_skill("test_skill")
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")
    system = jarvis_instance._build_system("hello", voice_mode=False)
    assert "Special Mode" in system
    assert "Be extra helpful" in system


def test_build_system_no_active_skill(jarvis_instance):
    jarvis_instance.deactivate_skill()
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")
    system = jarvis_instance._build_system("hello", voice_mode=False)
    assert isinstance(system, str)
    assert len(system) > 0


@pytest.mark.asyncio
async def test_execute_tool_known_tool(jarvis_instance):
    from jarvis.tools.registry import Tool
    jarvis_instance.registry.register(Tool(
        name="echo",
        description="Echo input",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        fn=lambda text: f"echo: {text}",
        category="test",
    ))
    result = await jarvis_instance._execute_tool("echo", {"text": "hi"})
    assert result == "echo: hi"


@pytest.mark.asyncio
async def test_execute_tool_unknown_synthesises(jarvis_instance, monkeypatch):
    from jarvis.tools.registry import Tool
    synthesised = Tool(
        name="new_cap", description="desc",
        input_schema={"type": "object", "properties": {}},
        fn=lambda **kwargs: "synthesised result",
        category="general", dynamic=True,
    )
    monkeypatch.setattr(
        "jarvis.core.synthesise_tool",
        AsyncMock(return_value=synthesised)
    )
    result = await jarvis_instance._execute_tool("new_cap", {"description": "some capability"})
    assert result == "synthesised result"


@pytest.mark.asyncio
async def test_execute_tool_synthesise_fails_returns_message(jarvis_instance, monkeypatch):
    monkeypatch.setattr("jarvis.core.synthesise_tool", AsyncMock(return_value=None))
    result = await jarvis_instance._execute_tool("phantom_tool", {})
    assert "Unable to synthesise" in result or "phantom_tool" in result


@pytest.mark.asyncio
async def test_execute_tools_parallel_collects_results(jarvis_instance):
    from jarvis.tools.registry import Tool
    jarvis_instance.registry.register(Tool(
        name="double",
        description="Doubles",
        input_schema={"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        fn=lambda n: str(n * 2),
        category="test",
    ))
    calls = [
        {"id": "t1", "name": "double", "input": {"n": 3}},
        {"id": "t2", "name": "double", "input": {"n": 5}},
    ]
    results = await jarvis_instance._execute_tools_parallel(calls)
    assert len(results) == 2
    contents = {r["tool_use_id"]: r["content"] for r in results}
    assert contents["t1"] == "6"
    assert contents["t2"] == "10"


@pytest.mark.asyncio
async def test_execute_tools_parallel_exception_captured(jarvis_instance):
    from jarvis.tools.registry import Tool
    jarvis_instance.registry.register(Tool(
        name="explode",
        description="Always fails",
        input_schema={"type": "object", "properties": {}},
        fn=lambda: (_ for _ in ()).throw(RuntimeError("bang")),
        category="test",
    ))
    calls = [{"id": "e1", "name": "explode", "input": {}}]
    results = await jarvis_instance._execute_tools_parallel(calls)
    assert "error" in results[0]["content"].lower() or "bang" in results[0]["content"].lower()


@pytest.mark.asyncio
async def test_run_agent_loop_max_iterations(jarvis_instance, monkeypatch):
    """After 12 tool-call rounds the loop should return the max-depth message."""
    # Always return a tool_use block so we never exit normally
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "t"
    tool_block.name = "echo"
    tool_block.input = {"text": "x"}
    resp = MagicMock(content=[tool_block])
    jarvis_instance.client.messages.create = AsyncMock(return_value=resp)
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")

    # Register echo so the tool executes without synthesise
    from jarvis.tools.registry import Tool
    jarvis_instance.registry.register(Tool(
        name="echo",
        description="Echo",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        fn=lambda text="": text,
        category="test",
    ))

    result = await jarvis_instance._run_agent_loop("keep going", voice_mode=False)
    assert "maximum reasoning depth" in result.lower() or "max" in result.lower()


# ── __init__ edge cases ───────────────────────────────────────────────────────

def test_init_raises_without_api_key(monkeypatch, tmp_data_dir):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setattr("jarvis.config.cfg.ANTHROPIC_API_KEY", "")
    from jarvis.core import Jarvis
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
        Jarvis()


def test_get_model_returns_config_model(jarvis_instance, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.CLAUDE_MODEL", "claude-test-4-0")
    assert jarvis_instance._get_model() == "claude-test-4-0"


def test_init_with_trajectory_collection(monkeypatch, tmp_data_dir):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setattr("jarvis.config.cfg.TRAJECTORY_COLLECTION", True)
    with patch("jarvis.plugins.loader.PluginLoader") as MockLoader:
        MockLoader.return_value.load_all.return_value = 0
        MockLoader.return_value.start_hot_reload.return_value = None
        from jarvis.core import Jarvis
        j = Jarvis()
    assert j.trajectories is not None


def test_new_session_with_trajectory_collection(jarvis_instance, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.TRAJECTORY_COLLECTION", True)
    old_id = jarvis_instance._session_id
    jarvis_instance.new_session(task="new task")
    assert jarvis_instance._session_id != old_id


@pytest.mark.asyncio
async def test_chat_with_trajectory_collection(jarvis_instance, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.TRAJECTORY_COLLECTION", True)
    jarvis_instance.semantic.store_conversation_snippet = MagicMock()
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")

    async def fake_loop(msg, voice):
        return "trajectory response"

    monkeypatch.setattr(jarvis_instance, "_run_agent_loop", fake_loop)
    result = await jarvis_instance.chat("trajectory test")
    assert result == "trajectory response"


# ── Confidence threshold ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_agent_loop_confidence_retry(jarvis_instance, monkeypatch):
    """When confidence is below threshold, the loop should retry once."""
    monkeypatch.setattr("jarvis.config.cfg.CONFIDENCE_THRESHOLD", 0.9)
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")

    call_count = [0]

    def make_text_response(text):
        block = MagicMock()
        block.type = "text"
        block.text = text
        return MagicMock(content=[block], stop_reason="end_turn")

    async def fake_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return make_text_response("low quality answer")
        return make_text_response("high quality answer")

    jarvis_instance.client.messages.create = fake_create

    # score_confidence returns 0.5 on first call (below threshold) then not called
    jarvis_instance.planner.score_confidence = AsyncMock(side_effect=[0.5, 0.95])

    result = await jarvis_instance._run_agent_loop("test question", voice_mode=False)
    assert call_count[0] == 2
    assert result == "high quality answer"


@pytest.mark.asyncio
async def test_run_agent_loop_no_confidence_check_in_voice_mode(jarvis_instance, monkeypatch):
    """Voice mode should skip confidence check."""
    monkeypatch.setattr("jarvis.config.cfg.CONFIDENCE_THRESHOLD", 0.9)
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")

    block = MagicMock()
    block.type = "text"
    block.text = "voice answer"
    jarvis_instance.client.messages.create = AsyncMock(return_value=MagicMock(content=[block]))

    jarvis_instance.planner.score_confidence = AsyncMock()
    result = await jarvis_instance._run_agent_loop("hello", voice_mode=True)
    assert result == "voice answer"
    jarvis_instance.planner.score_confidence.assert_not_called()


# ── Stream chat with tool calls ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_chat_with_tool_calls(jarvis_instance, monkeypatch):
    """Stream loop should handle tool-use events and continue."""
    from jarvis.tools.registry import Tool
    jarvis_instance.registry.register(Tool(
        name="ping",
        description="ping",
        input_schema={"type": "object", "properties": {}},
        fn=lambda: "pong",
        category="test",
    ))
    jarvis_instance.semantic.store_conversation_snippet = MagicMock()
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")

    StartEvent = type("RawContentBlockStartEvent", (), {})
    StopEvent = type("RawContentBlockStopEvent", (), {})
    DeltaEvent = type("RawContentBlockDeltaEvent", (), {})

    def make_tool_start():
        e = StartEvent()
        cb = MagicMock()
        cb.type = "tool_use"
        cb.id = "t1"
        cb.name = "ping"
        e.content_block = cb
        return e

    def make_tool_json():
        e = DeltaEvent()
        e.delta = MagicMock(text=None, partial_json="{}")
        return e

    def make_stop():
        return StopEvent()

    def make_text_event(text):
        e = DeltaEvent()
        e.delta = MagicMock(text=text, partial_json=None)
        return e

    iteration = [0]

    class FakeStreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            if iteration[0] == 0:
                # First iteration: tool call
                iteration[0] += 1
                yield make_tool_start()
                yield make_tool_json()
                yield make_stop()
            else:
                # Second iteration: text response
                yield make_text_event("done")

        async def get_final_message(self):
            return MagicMock(content=[])

    jarvis_instance.client.messages.stream = MagicMock(return_value=FakeStreamCtx())

    tokens = []
    async for token in jarvis_instance.stream_chat("use a tool"):
        tokens.append(token)

    assert "done" in tokens


# ── _execute_tool async function ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_tool_async_function(jarvis_instance):
    from jarvis.tools.registry import Tool

    async def async_fn(x):
        return f"async:{x}"

    jarvis_instance.registry.register(Tool(
        name="async_echo",
        description="Async echo",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        fn=async_fn,
        category="test",
    ))
    result = await jarvis_instance._execute_tool("async_echo", {"x": "hello"})
    assert result == "async:hello"


@pytest.mark.asyncio
async def test_execute_tools_parallel_gather_exception(jarvis_instance):
    """Line 283: Exception from gather is converted to 'Tool error:' string."""
    async def _always_raise(name, inputs):
        raise RuntimeError("task exploded")

    jarvis_instance._execute_tool = _always_raise
    calls = [{"id": "x1", "name": "any", "input": {}}]
    results = await jarvis_instance._execute_tools_parallel(calls)
    assert "Tool error" in results[0]["content"]
    assert "task exploded" in results[0]["content"]


# ── Jarvis.__init__ coverage for lines 62-63 and 68 ──────────────────────────

def test_init_subagent_tools_exception_swallowed(tmp_data_dir, monkeypatch, capsys):
    """Lines 62-63: subagent tools failure during init is printed and swallowed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setattr("jarvis.config.cfg.TRAJECTORY_COLLECTION", False)

    import jarvis.tools.subagent_tools as submod
    with patch("jarvis.core.PluginLoader") as MockLoader, \
         patch.object(submod, "register_tools", side_effect=RuntimeError("no agents")):
        MockLoader.return_value.load_all.return_value = 0
        MockLoader.return_value.start_hot_reload.return_value = None
        from jarvis.core import Jarvis
        j = Jarvis()
    out = capsys.readouterr().out
    assert "Subagent tools unavailable" in out
    assert j is not None


def test_init_prints_plugin_count_when_nonzero(tmp_data_dir, monkeypatch, capsys):
    """Line 68: 'Loaded N plugin(s).' is printed when load_all() returns > 0."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setattr("jarvis.config.cfg.TRAJECTORY_COLLECTION", False)

    with patch("jarvis.core.PluginLoader") as MockLoader:
        MockLoader.return_value.load_all.return_value = 3
        MockLoader.return_value.start_hot_reload.return_value = None
        from jarvis.core import Jarvis
        j = Jarvis()
    out = capsys.readouterr().out
    assert "Loaded 3 plugin(s)." in out
    assert j is not None


# ── _stream_agent_loop malformed JSON input_raw (lines 257-258) ──────────────

@pytest.mark.asyncio
async def test_stream_agent_loop_malformed_tool_json(jarvis_instance):
    """Lines 257-258: bad JSON in tool input_raw falls back to empty dict."""
    StartEvent = type("RawContentBlockStartEvent", (), {})
    StopEvent = type("RawContentBlockStopEvent", (), {})
    TextDeltaEvent = type("RawContentBlockDeltaEvent", (), {})

    def make_start_event():
        evt = StartEvent()
        block = MagicMock()
        block.type = "tool_use"
        block.id = "t1"
        block.name = "echo"
        evt.content_block = block
        return evt

    def make_delta_event(partial_json):
        evt = TextDeltaEvent()
        evt.delta = MagicMock()
        evt.delta.text = None
        evt.delta.partial_json = partial_json
        return evt

    def make_stop_event():
        return StopEvent()

    class FakeStreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield make_start_event()
            yield make_delta_event("{INVALID JSON")
            yield make_stop_event()

        async def get_final_message(self):
            final_msg = MagicMock()
            final_msg.content = []
            return final_msg

    jarvis_instance.client.messages.stream = MagicMock(return_value=FakeStreamCtx())
    jarvis_instance.semantic.recall_relevant = MagicMock(return_value="")
    jarvis_instance.learner.build_context_prompt = MagicMock(return_value="")
    jarvis_instance.semantic.store_conversation_snippet = MagicMock()

    tokens = []
    async for token in jarvis_instance.stream_chat("trigger malformed json"):
        tokens.append(token)
    # No exception raised — malformed JSON was swallowed and input defaulted to {}
