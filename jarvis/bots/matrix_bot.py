"""Matrix bot — JARVIS on Matrix/Element (free, self-hostable, E2E encrypted)."""

from __future__ import annotations
from typing import TYPE_CHECKING
from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis


async def run_matrix_bot(jarvis: "Jarvis") -> None:
    if not cfg.MATRIX_HOMESERVER or not cfg.MATRIX_ACCESS_TOKEN:
        print("[JARVIS Matrix] MATRIX_HOMESERVER / MATRIX_ACCESS_TOKEN not set. Skipping.")
        return
    try:
        from nio import AsyncClient, MatrixRoom, RoomMessageText
    except ImportError:
        print("[JARVIS Matrix] matrix-nio not installed. Run: pip install matrix-nio")
        return

    client = AsyncClient(cfg.MATRIX_HOMESERVER, cfg.MATRIX_USER_ID)
    client.access_token = cfg.MATRIX_ACCESS_TOKEN

    async def message_callback(room: MatrixRoom, event: RoomMessageText) -> None:
        if event.sender == cfg.MATRIX_USER_ID:
            return
        body = event.body.strip()
        if not body:
            return
        # Only respond when mentioned or in DMs
        if cfg.MATRIX_USER_ID not in body and not room.is_group:
            return
        text = body.replace(cfg.MATRIX_USER_ID, "").strip()
        try:
            reply = await jarvis.chat(text)
            await client.room_send(
                room.room_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": reply[:4000]},
            )
        except Exception as exc:
            print(f"[JARVIS Matrix] Error: {exc}")

    client.add_event_callback(message_callback, RoomMessageText)
    print(f"[JARVIS Matrix] Bot online at {cfg.MATRIX_HOMESERVER}")
    await client.sync_forever(timeout=30000)
