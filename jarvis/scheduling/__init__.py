"""Advanced scheduling — priority queues, retry logic, NL parsing, run history."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.core import Jarvis


class Priority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ScheduledJob:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    cron: str = ""
    prompt: str = ""
    priority: Priority = Priority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60
    enabled: bool = True
    last_run: str | None = None
    last_status: str | None = None
    run_count: int = 0
    error_count: int = 0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "cron": self.cron,
            "prompt": self.prompt,
            "priority": self.priority.name,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "last_status": self.last_status,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "tags": self.tags,
        }


@dataclass(order=True)
class _PrioritizedItem:
    priority: int  # negated so higher priority = lower heap value
    id: str = field(compare=False)
    fn: Callable[[], Awaitable[Any]] = field(compare=False)
    name: str = field(compare=False, default="")


class TaskQueue:
    """Priority queue for on-demand async tasks."""

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._history: list[dict] = []

    async def enqueue(
        self,
        fn: Callable[[], Awaitable[Any]],
        name: str = "",
        priority: Priority = Priority.NORMAL,
    ) -> str:
        task_id = uuid.uuid4().hex[:8]
        await self._queue.put(_PrioritizedItem(priority=-priority.value, id=task_id, fn=fn, name=name))
        return task_id

    async def run_worker(self) -> None:
        """Background worker — call as asyncio.create_task(queue.run_worker())."""
        while True:
            item: _PrioritizedItem = await self._queue.get()
            started = datetime.now(timezone.utc).isoformat()
            status = "success"
            try:
                await item.fn()
            except Exception as exc:
                status = f"error: {exc}"
            finally:
                self._history.append({
                    "id": item.id,
                    "name": item.name,
                    "started": started,
                    "finished": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                })
                if len(self._history) > 200:
                    self._history = self._history[-200:]
                self._queue.task_done()

    def history(self, limit: int = 50) -> list[dict]:
        return self._history[-limit:]

    @property
    def pending(self) -> int:
        return self._queue.qsize()


class NLScheduler:
    """Natural-language-aware scheduler that wraps the JARVIS memory store."""

    def __init__(self, jarvis: "Jarvis") -> None:
        self.jarvis = jarvis
        self._jobs: dict[str, ScheduledJob] = {}

    @staticmethod
    def nl_to_cron(phrase: str) -> str:
        from jarvis.tools.nlp_cron import _nl_to_cron
        return _nl_to_cron(phrase)

    async def add(
        self,
        name: str,
        schedule: str,
        prompt: str,
        priority: Priority = Priority.NORMAL,
        max_retries: int = 3,
        tags: list[str] | None = None,
    ) -> ScheduledJob:
        cron = schedule if len(schedule.split()) == 5 else self.nl_to_cron(schedule)
        job = ScheduledJob(
            name=name,
            cron=cron,
            prompt=prompt,
            priority=priority,
            max_retries=max_retries,
            tags=tags or [],
        )
        self._jobs[job.id] = job
        self.jarvis.memory.add_scheduled_task(name, cron, prompt)
        return job

    def remove(self, job_id: str) -> bool:
        return bool(self._jobs.pop(job_id, None))

    def list_jobs(self, tag: str = "") -> list[dict]:
        jobs = list(self._jobs.values())
        if tag:
            jobs = [j for j in jobs if tag in j.tags]
        return [j.to_dict() for j in sorted(jobs, key=lambda j: -j.priority)]

    async def run_now(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if not job:
            return f"Job '{job_id}' not found."
        for attempt in range(job.max_retries + 1):
            try:
                result = await self.jarvis.chat(job.prompt)
                job.last_run = datetime.now(timezone.utc).isoformat()
                job.last_status = "success"
                job.run_count += 1
                return result
            except Exception as exc:
                if attempt < job.max_retries:
                    await asyncio.sleep(job.retry_delay)
                else:
                    job.error_count += 1
                    job.last_status = f"failed after {job.max_retries} retries: {exc}"
                    return job.last_status
        return "Unreachable."


# Module-level shared task queue
task_queue = TaskQueue()
