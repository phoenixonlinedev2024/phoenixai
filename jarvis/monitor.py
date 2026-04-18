"""Proactive monitor — watch URLs and files, notify + act on changes."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.core import Jarvis


class ProactiveMonitor:
    """Periodically checks URLs/files for changes and triggers JARVIS actions."""

    def __init__(self, jarvis: "Jarvis") -> None:
        self.jarvis = jarvis
        self._running = False

    async def start(self, interval_seconds: int = 300) -> None:
        self._running = True
        print(f"[JARVIS Monitor] Started. Checking every {interval_seconds}s.")
        while self._running:
            await self._check_all()
            await asyncio.sleep(interval_seconds)

    def stop(self) -> None:
        self._running = False

    async def _check_all(self) -> None:
        targets = self.jarvis.memory.get_monitor_targets()
        tasks = [self._check_target(t) for t in targets]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_target(self, target: dict) -> None:
        name = target["name"]
        ttype = target["target_type"]
        content_hash = None

        try:
            if ttype == "url":
                content = await self._fetch_url(target["target"])
                content_hash = self._hash(content)
            elif ttype == "file":
                p = Path(target["target"])
                if not p.exists():
                    return
                content_hash = self._hash(p.read_text(encoding="utf-8", errors="replace"))
            else:
                return

            last_hash = target.get("last_hash")
            if last_hash and last_hash != content_hash:
                print(f"[JARVIS Monitor] Change detected: {name}")
                await self._trigger_action(name, target["action"], target["target"])
                _notify(f"JARVIS: Change in '{name}'", f"Action triggered: {target['action'][:60]}")

            self.jarvis.memory.update_monitor_hash(name, content_hash)

        except Exception as exc:
            print(f"[JARVIS Monitor] Error checking '{name}': {exc}")

    async def _fetch_url(self, url: str) -> str:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                return await resp.text()

    async def _trigger_action(self, name: str, action: str, target: str) -> None:
        prompt = f"Monitor '{name}' detected a change in {target}. Action to take: {action}"
        result = await self.jarvis.chat(prompt)
        print(f"[JARVIS Monitor] Action result: {result[:200]}")

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()


def _notify(title: str, message: str) -> None:
    """Send a desktop notification (cross-platform via plyer)."""
    try:
        from plyer import notification
        notification.notify(title=title, message=message, app_name="JARVIS", timeout=8)
    except Exception:
        print(f"[JARVIS Notify] {title}: {message}")
