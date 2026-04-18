"""Signal bot — JARVIS via signal-cli (free, open-source, E2E encrypted).
Requires: Java runtime + signal-cli binary.
Install: https://github.com/AsamK/signal-cli
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import TYPE_CHECKING

from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis


class SignalBot:
    def __init__(self, jarvis: "Jarvis") -> None:
        self.jarvis = jarvis
        self._running = False

    def _cli(self, *args) -> list[str]:
        return [cfg.SIGNAL_CLI_PATH, "-a", cfg.SIGNAL_PHONE_NUMBER, *args]

    async def send(self, recipient: str, message: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            *self._cli("send", "-m", message[:1000], recipient),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()

    async def run(self) -> None:
        if not cfg.SIGNAL_PHONE_NUMBER or not cfg.SIGNAL_CLI_PATH:
            print("[JARVIS Signal] SIGNAL_PHONE_NUMBER / SIGNAL_CLI_PATH not set. Skipping.")
            return

        self._running = True
        print("[JARVIS Signal] Listening for messages via signal-cli daemon...")

        proc = await asyncio.create_subprocess_exec(
            *self._cli("daemon", "--json"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        while self._running and proc.returncode is None:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
                if not line:
                    break
                data = json.loads(line.decode())
                await self._handle_event(data)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                print(f"[JARVIS Signal] Parse error: {exc}")

    async def _handle_event(self, data: dict) -> None:
        try:
            envelope = data.get("envelope", {})
            data_message = envelope.get("dataMessage", {})
            text = data_message.get("message", "")
            sender = envelope.get("source", "")
            if not text or not sender:
                return
            print(f"[JARVIS Signal] {sender}: {text}")
            reply = await self.jarvis.chat(text)
            await self.send(sender, reply)
        except Exception as exc:
            print(f"[JARVIS Signal] Handle error: {exc}")

    def stop(self) -> None:
        self._running = False


async def run_signal_bot(jarvis: "Jarvis") -> None:
    bot = SignalBot(jarvis)
    await bot.run()
