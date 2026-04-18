"""JARVIS 24/7 daemon — API server, WebSocket, voice loop, scheduler, bots, monitor."""

from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from jarvis.config import cfg
from jarvis.export import export_markdown, export_pdf
from jarvis.monitor import ProactiveMonitor
from jarvis.personality import JARVIS_VOICE_INTRO, JARVIS_WAKE_RESPONSES
from jarvis.observability import metrics
from jarvis.security import key_store, rate_limiter, SecurityMiddleware
from jarvis.acp import bus as acp_bus

if TYPE_CHECKING:
    from jarvis.core import Jarvis

WEB_UI_DIR = Path(__file__).parent / "web_ui"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(jarvis: "Jarvis") -> FastAPI:
    app = FastAPI(
        title="JARVIS — OpenClaw AI",
        description="Just A Rather Very Intelligent System REST + WebSocket API",
        version="2.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Pydantic models ──────────────────────────────────────────────────

    class ChatRequest(BaseModel):
        message: str
        voice_mode: bool = False
        profile: str = "default"

    class ChatResponse(BaseModel):
        response: str
        session_id: str
        timestamp: str

    class ScheduleRequest(BaseModel):
        name: str
        cron: str
        prompt: str

    class MonitorRequest(BaseModel):
        name: str
        target_type: str  # url | file
        target: str
        action: str

    class ProfileRequest(BaseModel):
        profile: str

    # ── REST endpoints ───────────────────────────────────────────────────

    @app.get("/")
    async def root():
        html_path = WEB_UI_DIR / "index.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return {"message": "JARVIS API online. See /docs."}

    @app.get("/health")
    async def health():
        return {"status": "online", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/status")
    async def status():
        return {"status": jarvis.status()}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        jarvis.set_profile(req.profile)
        metrics.inc("chat.requests")
        try:
            async with metrics.atime("chat.latency"):
                response = await jarvis.chat(req.message, voice_mode=req.voice_mode)
            metrics.inc("chat.success")
            acp_bus.publish_sync("chat.completed", {"message": req.message[:80], "session": jarvis._session_id})
            return ChatResponse(
                response=response,
                session_id=jarvis._session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            metrics.inc("chat.errors")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/session/end")
    async def end_session():
        summary = await jarvis.end_session()
        return {"summary": summary}

    @app.post("/session/new")
    async def new_session():
        jarvis.new_session()
        return {"message": "New session started.", "session_id": jarvis._session_id}

    @app.post("/profile")
    async def set_profile(req: ProfileRequest):
        jarvis.set_profile(req.profile)
        return {"profile": req.profile}

    @app.get("/memory/facts")
    async def get_facts():
        return {"facts": jarvis.memory.all_facts()}

    @app.get("/memory/lessons")
    async def get_lessons():
        return {"lessons": jarvis.memory.get_lessons(limit=50)}

    @app.get("/memory/gaps")
    async def get_gaps():
        return {"gaps": jarvis.memory.get_open_gaps()}

    @app.get("/memory/semantic")
    async def semantic_recall(q: str, n: int = 5):
        return {"results": jarvis.semantic.recall(q, n=n)}

    @app.get("/tools")
    async def list_tools():
        return {
            "tools": [
                {"name": t.name, "description": t.description, "category": t.category, "dynamic": t.dynamic}
                for t in jarvis.registry.all()
            ]
        }

    @app.post("/schedule")
    async def schedule_task(req: ScheduleRequest):
        jarvis.memory.add_scheduled_task(req.name, req.cron, req.prompt)
        return {"message": f"Task '{req.name}' scheduled ({req.cron})."}

    @app.get("/schedule")
    async def list_scheduled():
        return {"tasks": jarvis.memory.get_scheduled_tasks()}

    @app.post("/monitor")
    async def add_monitor(req: MonitorRequest):
        jarvis.memory.add_monitor_target(req.name, req.target_type, req.target, req.action)
        return {"message": f"Monitoring '{req.name}' ({req.target_type}: {req.target})."}

    @app.get("/monitor")
    async def list_monitors():
        return {"targets": jarvis.memory.get_monitor_targets()}

    @app.post("/export/markdown")
    async def export_md():
        msg = export_markdown(jarvis.memory, jarvis._session_id)
        return {"message": msg}

    @app.post("/export/pdf")
    async def export_pdf_ep():
        msg = export_pdf(jarvis.memory, jarvis._session_id)
        return {"message": msg}

    @app.get("/skills")
    async def list_skills():
        return {"skills": [s.to_dict() for s in jarvis.skills.all()]}

    @app.post("/skills/activate")
    async def activate_skill(body: dict):
        name = body.get("name", "")
        ok = jarvis.activate_skill(name)
        return {"activated": ok, "skill": name}

    @app.post("/skills/deactivate")
    async def deactivate_skill():
        jarvis.deactivate_skill()
        return {"message": "Skill deactivated."}

    @app.get("/trajectories/stats")
    async def trajectory_stats():
        return jarvis.trajectories.stats()

    @app.post("/trajectories/export")
    async def export_trajectories(body: dict = {}):
        from jarvis.research.sharegpt import export_dataset
        path = body.get("output", "jarvis_dataset.jsonl")
        msg = export_dataset(jarvis.trajectories, output_path=path)
        return {"message": msg}

    @app.post("/trajectories/export/atropos")
    async def export_atropos(body: dict = {}):
        from jarvis.research.sharegpt import export_atropos_format
        path = body.get("output", "jarvis_atropos.jsonl")
        msg = export_atropos_format(jarvis.trajectories, output_path=path)
        return {"message": msg}

    @app.get("/providers/health")
    async def provider_health():
        return await jarvis.provider_router.health_check()

    @app.get("/sandbox/backends")
    async def sandbox_backends():
        from jarvis.sandbox.router import SandboxRouter
        router = SandboxRouter()
        return {"backends": router.list_backends(), "preferred": cfg.SANDBOX_BACKEND}

    # Mount WhatsApp webhooks
    from jarvis.bots.whatsapp_bot import mount_whatsapp_webhook
    mount_whatsapp_webhook(app, jarvis)

    # ── Observability ────────────────────────────────────────────────────

    from fastapi.responses import PlainTextResponse

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return metrics.prometheus_text()

    @app.get("/metrics/json")
    async def json_metrics():
        return metrics.snapshot()

    # ── Security / key management ────────────────────────────────────────

    class KeyRequest(BaseModel):
        name: str = ""
        role: str = "user"

    @app.get("/security/keys")
    async def list_api_keys():
        return {"keys": key_store.list_keys()}

    @app.post("/security/keys")
    async def create_api_key(req: KeyRequest):
        raw = key_store.generate(name=req.name, role=req.role)
        return {"key": raw, "role": req.role, "name": req.name}

    @app.delete("/security/keys/{key_prefix}")
    async def revoke_api_key(key_prefix: str):
        # find by prefix
        for k in key_store._keys:
            if k.startswith(key_prefix):
                key_store.revoke(k)
                return {"revoked": True}
        return {"revoked": False}

    # ── ACP message bus ──────────────────────────────────────────────────

    @app.get("/acp/history")
    async def acp_history(topic: str = "", limit: int = 50):
        return {"messages": acp_bus.history(topic=topic, limit=limit)}

    @app.get("/acp/stats")
    async def acp_stats():
        return acp_bus.stats()

    @app.post("/acp/publish")
    async def acp_publish(body: dict):
        topic = body.get("topic", "")
        payload = body.get("payload")
        sender = body.get("sender", "api")
        if not topic:
            raise HTTPException(status_code=400, detail="topic required")
        await acp_bus.publish(topic, payload, sender)
        return {"published": True, "topic": topic}

    # ── Self-improvement ─────────────────────────────────────────────────

    @app.post("/self-improve/run")
    async def self_improve_run():
        from jarvis.self_improve import SelfImproveEngine
        engine = SelfImproveEngine(jarvis)
        result = await engine.run_cycle()
        return result

    @app.get("/self-improve/benchmarks")
    async def self_improve_history():
        from jarvis.self_improve import BenchmarkRunner
        runner = BenchmarkRunner(jarvis)
        return {"history": runner.load_history(), "trend": runner.trend()}

    @app.get("/self-improve/capabilities")
    async def capability_report():
        from jarvis.self_improve import CapabilityEvolver
        evolver = CapabilityEvolver(jarvis)
        return evolver.capability_report()

    # ── Advanced scheduling ──────────────────────────────────────────────

    class NLScheduleRequest(BaseModel):
        name: str
        schedule: str
        prompt: str
        priority: str = "NORMAL"
        tags: list[str] = []

    @app.post("/schedule/nl")
    async def schedule_nl(req: NLScheduleRequest):
        from jarvis.scheduling import NLScheduler, Priority
        sched = NLScheduler(jarvis)
        prio = Priority[req.priority.upper()] if req.priority.upper() in Priority.__members__ else Priority.NORMAL
        job = await sched.add(req.name, req.schedule, req.prompt, priority=prio, tags=req.tags)
        return job.to_dict()

    @app.get("/schedule/nl/parse")
    async def parse_nl_schedule(phrase: str):
        from jarvis.scheduling import NLScheduler
        cron = NLScheduler.nl_to_cron(phrase)
        return {"phrase": phrase, "cron": cron}

    # ── WebSocket streaming endpoint ─────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_chat(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                data = await ws.receive_json()
                message = data.get("message", "")
                profile = data.get("profile", "default")
                jarvis.set_profile(profile)

                async for token in jarvis.stream_chat(message):
                    await ws.send_json({"type": "token", "content": token})

                await ws.send_json({"type": "done"})
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await ws.send_json({"type": "error", "content": str(exc)})
            except Exception:
                pass

    return SecurityMiddleware(app, key_store, rate_limiter)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class JarvisScheduler:
    def __init__(self, jarvis: "Jarvis") -> None:
        self.jarvis = jarvis
        self.scheduler = AsyncIOScheduler()

    def start(self) -> None:
        self._load_tasks()
        self.scheduler.add_job(
            self._run_reflection, "interval",
            hours=cfg.REFLECTION_INTERVAL_HOURS,
            id="self_reflection", replace_existing=True,
        )
        self.scheduler.start()
        print(f"[JARVIS Scheduler] Running. Reflection every {cfg.REFLECTION_INTERVAL_HOURS}h.")

    def _load_tasks(self) -> None:
        for task in self.jarvis.memory.get_scheduled_tasks():
            try:
                self.scheduler.add_job(
                    self._run_task,
                    CronTrigger.from_crontab(task["cron"]),
                    id=task["name"], args=[task["name"], task["prompt"]], replace_existing=True,
                )
                print(f"[JARVIS Scheduler] Loaded: {task['name']} ({task['cron']})")
            except Exception as exc:
                print(f"[JARVIS Scheduler] Failed to load {task['name']}: {exc}")

    async def _run_task(self, name: str, prompt: str) -> None:
        print(f"[JARVIS Scheduler] Running: {name}")
        try:
            result = await self.jarvis.chat(prompt)
            print(f"[JARVIS Scheduler] '{name}': {result[:200]}")
            now = datetime.now(timezone.utc).isoformat()
            self.jarvis.memory.update_task_run(name, now, now)
        except Exception as exc:
            print(f"[JARVIS Scheduler] '{name}' error: {exc}")

    async def _run_reflection(self) -> None:
        print("[JARVIS] Scheduled self-reflection running...")
        result = await self.jarvis.learner.reflect(self.jarvis._session_transcript, self.jarvis.client)
        n = len(result.get("lessons", []))
        print(f"[JARVIS] Reflection done. {n} new lesson(s).")

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Voice loop
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
            if cfg.STT_ENGINE == "whisper":
                from jarvis.voice.whisper_stt import WhisperSTT
                self._stt = WhisperSTT(model_size=cfg.WHISPER_MODEL)
            else:
                from jarvis.voice.speech_to_text import STTEngine
                self._stt = STTEngine()
        return self._stt

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if not cfg.VOICE_ENABLED:
            return
        tts = self._get_tts()
        stt = self._get_stt()
        tts.speak(JARVIS_VOICE_INTRO)
        stt.start_listening(lambda t: self._on_transcript(t, loop))

    def _on_transcript(self, transcript: str, loop: asyncio.AbstractEventLoop) -> None:
        import random
        tts = self._get_tts()
        if transcript == "_wake_only_":
            tts.speak(random.choice(JARVIS_WAKE_RESPONSES))
            return
        print(f"[JARVIS Voice] Heard: {transcript}")
        future = asyncio.run_coroutine_threadsafe(self.jarvis.voice_chat(transcript), loop)
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
    loop = asyncio.get_event_loop()

    # Scheduler
    scheduler = JarvisScheduler(jarvis)
    scheduler.start()

    # Voice
    voice = VoiceLoop(jarvis)
    voice.start(loop)

    # Proactive monitor
    monitor = ProactiveMonitor(jarvis)
    monitor_task = asyncio.create_task(monitor.start(cfg.MONITOR_INTERVAL))

    # Telegram bot
    telegram_task = None
    if cfg.TELEGRAM_TOKEN:
        from jarvis.bots.telegram_bot import run_telegram_bot
        telegram_task = asyncio.create_task(run_telegram_bot(jarvis))

    # Discord bot
    discord_task = None
    if cfg.DISCORD_TOKEN:
        from jarvis.bots.discord_bot import run_discord_bot
        discord_task = asyncio.create_task(run_discord_bot(jarvis))

    # Slack bot
    slack_task = None
    if cfg.SLACK_BOT_TOKEN and cfg.SLACK_APP_TOKEN:
        from jarvis.bots.slack_bot import run_slack_bot
        slack_task = asyncio.create_task(run_slack_bot(jarvis))

    # Signal bot
    signal_task = None
    if cfg.SIGNAL_PHONE_NUMBER:
        from jarvis.bots.signal_bot import run_signal_bot
        signal_task = asyncio.create_task(run_signal_bot(jarvis))

    # IRC bot
    irc_task = None
    if cfg.IRC_SERVER:
        from jarvis.bots.irc_bot import run_irc_bot
        irc_task = asyncio.create_task(run_irc_bot(jarvis))

    # Matrix bot
    matrix_task = None
    if cfg.MATRIX_HOMESERVER and cfg.MATRIX_ACCESS_TOKEN:
        from jarvis.bots.matrix_bot import run_matrix_bot
        matrix_task = asyncio.create_task(run_matrix_bot(jarvis))

    # Mattermost bot
    mattermost_task = None
    if cfg.MATTERMOST_URL and cfg.MATTERMOST_TOKEN:
        from jarvis.bots.mattermost_bot import run_mattermost_bot
        mattermost_task = asyncio.create_task(run_mattermost_bot(jarvis))

    # Self-improvement background cycle
    self_improve_task = None
    if cfg.SELF_IMPROVE_INTERVAL_HOURS > 0:
        from jarvis.self_improve import SelfImproveEngine
        engine = SelfImproveEngine(jarvis)
        self_improve_task = asyncio.create_task(engine.start_background())

    # FastAPI (wrapped with security middleware)
    app = create_app(jarvis)
    config = uvicorn.Config(app, host=cfg.API_HOST, port=cfg.API_PORT, log_level="warning")
    server = uvicorn.Server(config)

    def _shutdown(sig, frame):
        print("\n[JARVIS] Shutdown signal received. Standing down, Sir.")
        voice.stop()
        scheduler.stop()
        monitor.stop()
        monitor_task.cancel()
        for task in [telegram_task, discord_task, slack_task, signal_task,
                     irc_task, matrix_task, mattermost_task, self_improve_task]:
            if task:
                task.cancel()
        server.should_exit = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(
        f"\n[JARVIS] All systems online.\n"
        f"  API     : http://{cfg.API_HOST}:{cfg.API_PORT}\n"
        f"  Web UI  : http://{cfg.API_HOST}:{cfg.API_PORT}/\n"
        f"  Docs    : http://{cfg.API_HOST}:{cfg.API_PORT}/docs\n"
        f"  WS      : ws://{cfg.API_HOST}:{cfg.API_PORT}/ws\n"
        f"\n{jarvis.status()}\n"
    )
    await server.serve()
