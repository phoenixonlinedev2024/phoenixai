"""IRC bot — JARVIS on IRC channels (classic, free, no registration needed)."""

from __future__ import annotations
import asyncio
import re
from typing import TYPE_CHECKING
from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis


async def run_irc_bot(jarvis: "Jarvis") -> None:
    if not cfg.IRC_SERVER:
        print("[JARVIS IRC] IRC_SERVER not set. Skipping.")
        return

    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None

    async def send(line: str) -> None:
        if writer:
            writer.write((line + "\r\n").encode())
            await writer.drain()

    async def connect() -> None:
        nonlocal reader, writer
        reader, writer = await asyncio.open_connection(cfg.IRC_SERVER, cfg.IRC_PORT)
        await send(f"NICK {cfg.IRC_NICK}")
        await send(f"USER {cfg.IRC_NICK} 0 * :JARVIS Bot")
        for ch in cfg.IRC_CHANNELS.split(","):
            ch = ch.strip()
            if ch:
                await asyncio.sleep(2)
                await send(f"JOIN {ch}")
        print(f"[JARVIS IRC] Connected to {cfg.IRC_SERVER}:{cfg.IRC_PORT}")

    await connect()

    while True:
        try:
            if not reader:
                break
            line = await reader.readline()
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            if text.startswith("PING"):
                await send(text.replace("PING", "PONG"))
                continue

            # Parse :nick!user@host PRIVMSG #channel :message
            m = re.match(r":(\S+)!\S+ PRIVMSG (\S+) :(.*)", text)
            if not m:
                continue
            nick, target, msg = m.group(1), m.group(2), m.group(3)
            if nick == cfg.IRC_NICK:
                continue

            # Respond when mentioned or in PM
            if cfg.IRC_NICK.lower() in msg.lower() or not target.startswith("#"):
                clean = re.sub(rf"\b{re.escape(cfg.IRC_NICK)}\b", "", msg, flags=re.I).strip(": ,")
                reply = await jarvis.chat(clean)
                reply_target = nick if not target.startswith("#") else target
                for line_part in reply[:400].split("\n")[:4]:
                    await send(f"PRIVMSG {reply_target} :{line_part}")

        except Exception as exc:
            print(f"[JARVIS IRC] Error: {exc}. Reconnecting in 10s...")
            await asyncio.sleep(10)
            await connect()
