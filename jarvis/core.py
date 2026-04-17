"""JARVIS core agent — the brain that ties everything together."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import anthropic

from jarvis.config import cfg
from jarvis.memory.learning import LearningEngine
from jarvis.memory.store import MemoryStore
from jarvis.personality import (
    JARVIS_CAPABILITY_CREATED_TEMPLATE,
    JARVIS_LEARNING_TEMPLATE,
    JARVIS_SYSTEM_PROMPT,
)
from jarvis.tools.creator import synthesise_tool
from jarvis.tools.registry import ToolRegistry, build_registry


class Jarvis:
    """The JARVIS agent — universal, self-improving, always-on AI assistant."""

    def __init__(self) -> None:
        if not cfg.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file or environment."
            )
        self.client = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)
        self.registry: ToolRegistry = build_registry()
        self.memory = MemoryStore()
        self.learner = LearningEngine(self.memory)
        self._session_id: str = str(uuid.uuid4())
        self._session_transcript: list[dict] = []

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    async def chat(self, user_message: str, voice_mode: bool = False) -> str:
        """Process a user message and return JARVIS's response."""
        self.memory.save_message(self._session_id, "user", user_message)
        self._session_transcript.append({"role": "user", "content": user_message})

        response = await self._run_agent_loop(user_message, voice_mode)

        self.memory.save_message(self._session_id, "assistant", response)
        self._session_transcript.append({"role": "assistant", "content": response})

        return response

    async def end_session(self) -> str:
        """Run reflection and return a summary."""
        result = await self.learner.reflect(self._session_transcript, self.client)
        n_lessons = len(result.get("lessons", []))
        self._session_id = str(uuid.uuid4())
        self._session_transcript = []
        return JARVIS_LEARNING_TEMPLATE.format(count=n_lessons) if n_lessons else "Session ended, Sir."

    def new_session(self) -> None:
        self._session_id = str(uuid.uuid4())
        self._session_transcript = []

    # ------------------------------------------------------------------ #
    # Agent loop (tool-use agentic loop with synthesis fallback)
    # ------------------------------------------------------------------ #

    async def _run_agent_loop(self, user_message: str, voice_mode: bool) -> str:
        history = self.memory.get_history(self._session_id, limit=40)

        # Inject memory context into system prompt
        memory_ctx = self.learner.build_context_prompt()
        system = JARVIS_SYSTEM_PROMPT
        if memory_ctx:
            system = f"{JARVIS_SYSTEM_PROMPT}\n\n## Your Current Memory\n{memory_ctx}"

        messages = history + [{"role": "user", "content": user_message}]

        for _iteration in range(10):  # max 10 tool-call rounds
            response = await self.client.messages.create(
                model=cfg.CLAUDE_MODEL,
                max_tokens=cfg.MAX_TOKENS,
                system=system,
                tools=self.registry.anthropic_tools(),
                messages=messages,
            )

            # Collect text and tool-use blocks
            text_parts: list[str] = []
            tool_calls: list[dict] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

            if not tool_calls:
                # Final response — return text
                return " ".join(text_parts).strip()

            # Append assistant turn with all content blocks
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and collect results
            tool_results = []
            for call in tool_calls:
                result = await self._execute_tool(call["name"], call["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": str(result),
                })
                # Track skill use
                self.memory.record_skill_use(call["name"], call["name"], success=True)

            messages.append({"role": "user", "content": tool_results})

        return "I've reached the maximum reasoning depth for this task, Sir. Shall I continue with a fresh approach?"

    async def _execute_tool(self, name: str, inputs: dict) -> Any:
        tool = self.registry.get(name)

        if tool is None:
            # Tool doesn't exist — synthesise it
            print(f"[JARVIS] Tool '{name}' not found. Synthesising...")
            description = inputs.get("description", name.replace("_", " "))
            new_tool = await synthesise_tool(description, self.client, self.registry)
            if new_tool:
                print(JARVIS_CAPABILITY_CREATED_TEMPLATE.format(name=name))
                tool = new_tool
            else:
                return f"Unable to synthesise tool '{name}'. Attempting alternative approach."

        try:
            result = tool.run(**inputs)
            # If the function is a coroutine
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as exc:
            self.memory.record_skill_use(name, name, success=False)
            return f"Tool '{name}' error: {exc}"

    # ------------------------------------------------------------------ #
    # Voice convenience
    # ------------------------------------------------------------------ #

    async def voice_chat(self, transcript: str) -> str:
        """Process a voice transcript (shorter, spoken-word-optimised response)."""
        voice_hint = "\n\n[Voice mode: respond conversationally in short spoken sentences. No markdown.]"
        return await self.chat(transcript + voice_hint, voice_mode=True)

    # ------------------------------------------------------------------ #
    # Status / diagnostics
    # ------------------------------------------------------------------ #

    def status(self) -> str:
        return (
            f"JARVIS Status — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Model      : {cfg.CLAUDE_MODEL}\n"
            f"Tools      : {len(self.registry.all())} registered "
            f"({sum(1 for t in self.registry.all() if t.dynamic)} dynamic)\n"
            f"Session    : {self._session_id[:8]}...\n"
            f"Voice      : {'enabled' if cfg.VOICE_ENABLED else 'disabled'}\n"
            f"{self.memory.summary()}"
        )
