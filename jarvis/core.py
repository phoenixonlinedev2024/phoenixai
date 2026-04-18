"""JARVIS core agent — universal, self-improving, always-on AI assistant."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import anthropic

from jarvis.config import cfg
from jarvis.memory.learning import LearningEngine
from jarvis.memory.semantic import SemanticMemory
from jarvis.memory.store import MemoryStore
from jarvis.personality import (
    JARVIS_CAPABILITY_CREATED_TEMPLATE,
    JARVIS_LEARNING_TEMPLATE,
    get_system_prompt,
)
from jarvis.planner import TaskPlanner
from jarvis.tools.creator import synthesise_tool
from jarvis.tools.registry import ToolRegistry, build_registry
from jarvis.plugins.loader import PluginLoader
from jarvis.skills.registry import SkillRegistry
from jarvis.skills.builtin import load_builtin_skills
from jarvis.research.trajectory import TrajectoryCollector
from jarvis.providers.router import ProviderRouter


class Jarvis:
    """The JARVIS agent — universal, self-improving, always-on AI assistant."""

    def __init__(self) -> None:
        if not cfg.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file or environment."
            )
        self.client = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)
        self.provider_router = ProviderRouter()
        self.registry: ToolRegistry = build_registry()
        self.memory = MemoryStore()
        self.semantic = SemanticMemory()
        self.learner = LearningEngine(self.memory)
        self.planner = TaskPlanner(self.client, self.memory)
        self.skills = SkillRegistry()
        self.trajectories = TrajectoryCollector()
        self._session_id: str = str(uuid.uuid4())
        self._session_transcript: list[dict] = []
        self._profile: str = cfg.DEFAULT_PROFILE
        self._active_skill: str | None = None
        self._plugin_loader = PluginLoader(self.registry)

        # Load built-in skills
        load_builtin_skills(self.skills)

        # Register subagent tools (needs self reference)
        try:
            from jarvis.tools.subagent_tools import register_tools as reg_subagent
            reg_subagent(self.registry, jarvis=self)
        except Exception as exc:
            print(f"[JARVIS] Subagent tools unavailable: {exc}")

        # Load plugins and start hot reload
        n = self._plugin_loader.load_all()
        if n:
            print(f"[JARVIS] Loaded {n} plugin(s).")
        self._plugin_loader.start_hot_reload()

        # Start trajectory collection
        if cfg.TRAJECTORY_COLLECTION:
            self.trajectories.start(self._session_id)

    # ------------------------------------------------------------------ #
    # Personality profile
    # ------------------------------------------------------------------ #

    def set_profile(self, profile: str) -> None:
        from jarvis.personality import PERSONALITY_PROFILES
        if profile in PERSONALITY_PROFILES:
            self._profile = profile

    def _get_model(self) -> str:
        return cfg.CLAUDE_MODEL

    def activate_skill(self, skill_name: str) -> bool:
        skill = self.skills.get(skill_name)
        if skill:
            self._active_skill = skill_name
            self.skills.increment_usage(skill_name)
            return True
        return False

    def deactivate_skill(self) -> None:
        self._active_skill = None

    # ------------------------------------------------------------------ #
    # Public chat interface
    # ------------------------------------------------------------------ #

    async def chat(self, user_message: str, voice_mode: bool = False) -> str:
        self.memory.save_message(self._session_id, "user", user_message)
        self._session_transcript.append({"role": "user", "content": user_message})
        self.semantic.store_conversation_snippet(user_message, self._session_id)
        if cfg.TRAJECTORY_COLLECTION:
            self.trajectories.record_turn(self._session_id, "user", user_message)

        response = await self._run_agent_loop(user_message, voice_mode)

        self.memory.save_message(self._session_id, "assistant", response)
        self._session_transcript.append({"role": "assistant", "content": response})
        self.semantic.store_conversation_snippet(response, self._session_id)
        if cfg.TRAJECTORY_COLLECTION:
            self.trajectories.record_turn(self._session_id, "assistant", response)

        return response

    async def stream_chat(self, user_message: str, voice_mode: bool = False) -> AsyncIterator[str]:
        """Stream tokens as they're generated."""
        self.memory.save_message(self._session_id, "user", user_message)
        self._session_transcript.append({"role": "user", "content": user_message})
        self.semantic.store_conversation_snippet(user_message, self._session_id)

        full_response = ""
        async for token in self._stream_agent_loop(user_message, voice_mode):
            full_response += token
            yield token

        self.memory.save_message(self._session_id, "assistant", full_response)
        self._session_transcript.append({"role": "assistant", "content": full_response})
        self.semantic.store_conversation_snippet(full_response, self._session_id)

    async def voice_chat(self, transcript: str) -> str:
        return await self.chat(transcript, voice_mode=True)

    async def end_session(self) -> str:
        result = await self.learner.reflect(self._session_transcript, self.client)
        n_lessons = len(result.get("lessons", []))
        # Store lessons in semantic memory too
        for lesson in result.get("lessons", []):
            self.semantic.store_lesson(lesson)
        self.new_session()
        return JARVIS_LEARNING_TEMPLATE.format(count=n_lessons) if n_lessons else "Session ended, Sir."

    def new_session(self, task: str = "") -> None:
        self._session_id = str(uuid.uuid4())
        self._session_transcript = []
        self._active_skill = None
        if cfg.TRAJECTORY_COLLECTION:
            self.trajectories.start(self._session_id, task=task)

    # ------------------------------------------------------------------ #
    # Agent loop
    # ------------------------------------------------------------------ #

    def _build_system(self, user_message: str, voice_mode: bool) -> str:
        memory_ctx = self.learner.build_context_prompt()
        semantic_ctx = self.semantic.recall_relevant(user_message, n=6)
        gap_ctx = "\n".join(f"- {g['description']}" for g in self.memory.get_open_gaps()[:5])

        # Inject active skill system prompt
        skill_addon = ""
        if self._active_skill:
            skill = self.skills.get(self._active_skill)
            if skill:
                skill_addon = f"\n\n## Active Skill: {skill.name}\n{skill.system_prompt}"

        return get_system_prompt(
            profile=self._profile,
            voice_mode=voice_mode,
            memory_context="\n".join(filter(None, [memory_ctx, semantic_ctx])),
            gap_context=gap_ctx,
        ) + skill_addon

    async def _run_agent_loop(self, user_message: str, voice_mode: bool) -> str:
        history = self.memory.get_history(self._session_id, limit=40)
        system = self._build_system(user_message, voice_mode)
        messages = history + [{"role": "user", "content": user_message}]

        for _iteration in range(12):
            response = await self.client.messages.create(
                model=cfg.CLAUDE_MODEL,
                max_tokens=cfg.MAX_TOKENS,
                system=system,
                tools=self.registry.anthropic_tools(),
                messages=messages,
            )

            text_parts: list[str] = []
            tool_calls: list[dict] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

            if not tool_calls:
                result_text = " ".join(text_parts).strip()
                # Confidence check — retry once if below threshold
                if cfg.CONFIDENCE_THRESHOLD > 0 and not voice_mode:
                    score = await self.planner.score_confidence(user_message, result_text)
                    if score < cfg.CONFIDENCE_THRESHOLD:
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({"role": "user", "content": (
                            f"Your response had a confidence score of {score:.2f}. "
                            "Please reconsider and provide a better, more complete answer."
                        )})
                        continue
                return result_text

            messages.append({"role": "assistant", "content": response.content})

            # Execute tools in parallel where possible
            tool_results = await self._execute_tools_parallel(tool_calls)
            messages.append({"role": "user", "content": tool_results})

        return "I've reached the maximum reasoning depth for this task, Sir. Shall I continue with a fresh approach?"

    async def _stream_agent_loop(self, user_message: str, voice_mode: bool) -> AsyncIterator[str]:
        """Stream text tokens while still handling tool calls."""
        history = self.memory.get_history(self._session_id, limit=40)
        system = self._build_system(user_message, voice_mode)
        messages = history + [{"role": "user", "content": user_message}]

        for _iteration in range(12):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            current_tool: dict = {}

            async with self.client.messages.stream(
                model=cfg.CLAUDE_MODEL,
                max_tokens=cfg.MAX_TOKENS,
                system=system,
                tools=self.registry.anthropic_tools(),
                messages=messages,
            ) as stream:
                async for event in stream:
                    etype = type(event).__name__
                    if etype == "RawContentBlockDeltaEvent":
                        delta = event.delta
                        if hasattr(delta, "text") and delta.text:
                            text_parts.append(delta.text)
                            yield delta.text
                        elif hasattr(delta, "partial_json") and delta.partial_json:
                            current_tool.setdefault("input_raw", "")
                            current_tool["input_raw"] += delta.partial_json
                    elif etype == "RawContentBlockStartEvent":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool = {"id": block.id, "name": block.name, "input_raw": ""}
                    elif etype == "RawContentBlockStopEvent":
                        if current_tool.get("name"):
                            try:
                                current_tool["input"] = json.loads(current_tool.pop("input_raw", "{}"))
                            except Exception:
                                current_tool["input"] = {}
                            tool_calls.append(current_tool.copy())
                            current_tool = {}

            if not tool_calls:
                return

            # Execute tools and continue
            final = await stream.get_final_message()
            messages.append({"role": "assistant", "content": final.content})
            tool_results = await self._execute_tools_parallel(tool_calls)
            messages.append({"role": "user", "content": tool_results})

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #

    async def _execute_tools_parallel(self, tool_calls: list[dict]) -> list[dict]:
        """Execute all tool calls concurrently and return results."""
        tasks = [self._execute_tool(call["name"], call["input"]) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_results = []
        for call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                result = f"Tool error: {result}"
            self.memory.record_skill_use(call["name"], call["name"], success=not isinstance(result, str) or "error" not in result.lower())
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": str(result),
            })
        return tool_results

    async def _execute_tool(self, name: str, inputs: dict) -> Any:
        tool = self.registry.get(name)

        if tool is None:
            # Log the gap
            self.memory.log_gap(f"Tool '{name}' requested but not available", context=str(inputs))
            # Synthesise
            description = inputs.get("description", name.replace("_", " "))
            new_tool = await synthesise_tool(description, self.client, self.registry)
            if new_tool:
                print(JARVIS_CAPABILITY_CREATED_TEMPLATE.format(name=name))
                # Mark gap resolved
                gaps = self.memory.get_open_gaps()
                for g in gaps:
                    if name in g["description"]:
                        self.memory.resolve_gap(g["id"])
                tool = new_tool
            else:
                return f"Unable to synthesise tool '{name}'. Attempting alternative approach."

        try:
            result = tool.run(**inputs)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as exc:
            self.memory.record_skill_use(name, name, success=False)
            return f"Tool '{name}' error: {exc}"

    # ------------------------------------------------------------------ #
    # Status / diagnostics
    # ------------------------------------------------------------------ #

    def status(self) -> str:
        dynamic = sum(1 for t in self.registry.all() if t.dynamic)
        return (
            f"JARVIS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Model      : {cfg.CLAUDE_MODEL}\n"
            f"Profile    : {self._profile}\n"
            f"Tools      : {len(self.registry.all())} registered ({dynamic} dynamic)\n"
            f"Session    : {self._session_id[:8]}...\n"
            f"Voice      : {'enabled' if cfg.VOICE_ENABLED else 'disabled'} ({cfg.STT_ENGINE} STT / {cfg.TTS_ENGINE} TTS)\n"
            f"Semantic   : {self.semantic.count()} vectors stored\n"
            f"{self.memory.summary()}"
        )
