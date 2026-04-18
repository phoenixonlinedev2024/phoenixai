"""Mattermost bot — JARVIS on self-hosted Mattermost (free, open-source)."""

from __future__ import annotations
import asyncio
import json
from typing import TYPE_CHECKING
from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis


async def run_mattermost_bot(jarvis: "Jarvis") -> None:
    if not cfg.MATTERMOST_URL or not cfg.MATTERMOST_TOKEN:
        print("[JARVIS Mattermost] MATTERMOST_URL / MATTERMOST_TOKEN not set. Skipping.")
        return

    ws_url = cfg.MATTERMOST_URL.replace("http", "ws") + "/api/v4/websocket"

    try:
        import websockets
        import aiohttp
    except ImportError:
        print("[JARVIS Mattermost] websockets/aiohttp not installed.")
        return

    headers = {"Authorization": f"Bearer {cfg.MATTERMOST_TOKEN}"}

    async def post_message(channel_id: str, text: str) -> None:
        async with aiohttp.ClientSession(headers=headers) as session:
            await session.post(
                f"{cfg.MATTERMOST_URL}/api/v4/posts",
                json={"channel_id": channel_id, "message": text[:4000]},
            )

    async def listen() -> None:
        async with websockets.connect(ws_url, extra_headers=headers) as ws:
            await ws.send(json.dumps({
                "seq": 1, "action": "authentication_challenge",
                "data": {"token": cfg.MATTERMOST_TOKEN},
            }))
            print("[JARVIS Mattermost] Bot online.")
            async for raw in ws:
                try:
                    event = json.loads(raw)
                    if event.get("event") != "posted":
                        continue
                    post = json.loads(event["data"]["post"])
                    msg = post.get("message", "").strip()
                    channel_id = post.get("channel_id", "")
                    user_id = post.get("user_id", "")
                    if not msg or user_id == cfg.MATTERMOST_BOT_USER_ID:
                        continue
                    if f"@{cfg.MATTERMOST_BOT_NAME}" in msg or not msg.startswith("@"):
                        clean = msg.replace(f"@{cfg.MATTERMOST_BOT_NAME}", "").strip()
                        reply = await jarvis.chat(clean)
                        await post_message(channel_id, reply)
                except Exception as exc:
                    print(f"[JARVIS Mattermost] Parse error: {exc}")

    while True:
        try:
            await listen()
        except Exception as exc:
            print(f"[JARVIS Mattermost] Disconnected: {exc}. Reconnecting in 5s...")
            await asyncio.sleep(5)
