"""JARVIS 24/7 daemon — background service, API server, voice loop, and scheduler."""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from jarvis.config import cfg
from jarvis.personality import JARVIS_VOICE_INTRO, JARVIS_WAKE_RESPONSES

if TYPE_CHECKING:
    from jarvis.core import Jarvis


# ---------------------------------------------------------------------------
# FastAPI REST interface
# ---------------------------------------------------------------------------

def create_app(jarvis: "Jarvis") -> FastAPI:
    app = FastAPI(
        title="JARVIS — OpenClaw AI",
        description="Just A Rather Very Intelligent System REST API",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ChatRequest(BaseModel):
        message: str
        session_id: str | None = None
        voice_mode: bool = False

    class ChatResponse(BaseModel):
        response: str
        session_id: str
        timestamp: str

    class ScheduleRequest(BaseModel):
        name: str
        cron: str
        prompt: str

    @app.get("/health")
    async def health():
        return {"status": "online", "jarvis": "operational", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/status")
    async def status():
        return {"status": jarvis.status()}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        try:
            response = await jarvis.chat(req.message, voice_mode=req.voice_mode)
            return ChatResponse(
                response=response,
                session_id=jarvis._session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/session/end")
    async def end_session():
        summary = await jarvis.end_session()
        return {"summary": summary}

    @app.post("/session/new")
    async def new_session():
        jarvis.new_session()
        return {"message": "New session started.", "session_id": jarvis._session_id}

    @app.get("/memory/facts")
    async def get_facts():
        return {"facts": jarvis.memory.all_facts()}

    @app.get("/memory/lessons")
    async def get_lessons():
        return {"lessons": jarvis.memory.get_lessons(limit=50)}

    @app.get("/tools")
    async def list_tools():
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "dynamic": t.dynamic,
                }
                for t in jarvis.registry.all()
            ]
        }

    @app.post("/schedule")
    async def schedule_task(req: ScheduleRequest):
        jarvis.memory.add_scheduled_task(req.name, req.cron, req.prompt)
        return {"message": f"Task '{req.name}' scheduled with cron: {req.cron}"}

    @app.get("/schedule")
    async def list_scheduled():
        return {"tasks": jarvis.memory.get_scheduled_tasks()}

    return app


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class JarvisScheduler:
    def __init__(self, jarvis: "Jarvis") -> None:
        self.jarvis = jarvis
        self.scheduler = AsyncIOScheduler()

    def start(self) -> None:
        self._load_tasks()
        # Periodic self-reflection
        self.scheduler.add_job(
            self._run_reflection,
            "interval",
            hours=cfg.REFLECTION_INTERVAL_HOURS,
            id="self_reflection",
            replace_existing=True,
        )
        self.scheduler.start()
        print(f"[JARVIS Scheduler] Running. Reflection every {cfg.REFLECTION_INTERVAL_HOURS}h.")

    def _load_tasks(self) -> None:
        for task in self.jarvis.memory.get_scheduled_tasks():
            try:
                self.scheduler.add_job(
                    self._run_task,
                    CronTrigger.from_crontab(task["cron"]),
                    id=task["name"],
                    args=[task["name"], task["prompt"]],
                    replace_existing=True,
                )
                print(f"[JARVIS Scheduler] Loaded task: {task['name']} ({task['cron']})")
            except Exception as exc:
                print(f"[JARVIS Scheduler] Failed to load task {task['name']}: {exc}")

    async def _run_task(self, name: str, prompt: str) -> None:
        print(f"[JARVIS Scheduler] Running task: {name}")
        try:
            result = await self.jarvis.chat(prompt)
            print(f"[JARVIS Scheduler] Task '{name}' result: {result[:200]}")
            now = datetime.now(timezone.utc).isoformat()
            self.jarvis.memory.update_task_run(name, now, now)
        except Exception as exc:
            print(f"[JARVIS Scheduler] Task '{name}' error: {exc}")

    async def _run_reflection(self) -> None:
        print("[JARVIS] Running scheduled self-reflection...")
        result = await self.jarvis.learner.reflect(
            self.jarvis._session_transcript, self.jarvis.client
        )
        n = len(result.get("lessons", []))
        print(f"[JARVIS] Reflection complete. {n} new lesson(s) stored.")

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Voice loop (runs in background thread)
# ---------------------------------------------------------------------------

class VoiceLoop:
    def __init__(self, jarvis: "Jarvis") -> None:
        self.jarvis = jarvis
        self._tts = None
        self._stt = None

    def _get_tts(self):
        if self._tts is None:
            from jarvis.voice.text_to_speech import TTSEngine
            self._tts = TTSEngine()
        return self._tts

    def _get_stt(self):
        if self._stt is None:
            from jarvis.voice.speech_to_text import STTEngine
            self._stt = STTEngine()
        return self._stt

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if not cfg.VOICE_ENABLED:
            return
        tts = self._get_tts()
        stt = self._get_stt()
        tts.speak(JARVIS_VOICE_INTRO)
        stt.start_listening(lambda transcript: self._on_transcript(transcript, loop))

    def _on_transcript(self, transcript: str, loop: asyncio.AbstractEventLoop) -> None:
        tts = self._get_tts()
        if transcript == "_wake_only_":
            import random
            tts.speak(random.choice(JARVIS_WAKE_RESPONSES))
            return
        print(f"[JARVIS Voice] Heard: {transcript}")
        future = asyncio.run_coroutine_threadsafe(
            self.jarvis.voice_chat(transcript), loop
        )
        try:
            response = future.result(timeout=60)
            tts.speak(response)
        except Exception as exc:
            print(f"[JARVIS Voice] Error: {exc}")
            tts.speak("I encountered an error, Sir. Please check the logs.")

    def stop(self) -> None:
        if self._stt:
            self._stt.stop_listening()


# ---------------------------------------------------------------------------
# Daemon entry point
# ---------------------------------------------------------------------------

async def run_daemon(jarvis: "Jarvis") -> None:
    """Start all JARVIS subsystems and run forever."""
    loop = asyncio.get_event_loop()

    # Scheduler
    scheduler = JarvisScheduler(jarvis)
    scheduler.start()

    # Voice loop (background thread)
    voice = VoiceLoop(jarvis)
    voice.start(loop)

    # FastAPI server
    app = create_app(jarvis)
    config = uvicorn.Config(
        app,
        host=cfg.API_HOST,
        port=cfg.API_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    # Graceful shutdown
    def _shutdown(sig, frame):
        print("\n[JARVIS] Shutdown signal received. Standing down, Sir.")
        voice.stop()
        scheduler.stop()
        server.should_exit = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(
        f"[JARVIS] All systems online. API: http://{cfg.API_HOST}:{cfg.API_PORT}\n"
        f"[JARVIS] {jarvis.status()}"
    )
    await server.serve()
