"""Slack bot — JARVIS on Slack via slack-bolt (free)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis


async def run_slack_bot(jarvis: "Jarvis") -> None:
    if not cfg.SLACK_BOT_TOKEN or not cfg.SLACK_APP_TOKEN:
        print("[JARVIS Slack] SLACK_BOT_TOKEN / SLACK_APP_TOKEN not set. Skipping.")
        return
    try:
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    except ImportError:
        print("[JARVIS Slack] slack-bolt not installed. Run: pip install slack-bolt")
        return

    app = AsyncApp(token=cfg.SLACK_BOT_TOKEN)

    @app.event("app_mention")
    async def handle_mention(event, say):
        text = event.get("text", "")
        # Strip the mention token <@BOTID>
        import re
        text = re.sub(r"<@\w+>", "", text).strip()
        if not text:
            await say("Yes, Sir? How may I assist you?")
            return
        await say(":hourglass: Processing...")
        try:
            reply = await jarvis.chat(text)
            await say(reply[:3000])
        except Exception as exc:
            await say(f"Error: {exc}")

    @app.message("jarvis")
    async def handle_message(message, say):
        text = message.get("text", "")
        try:
            reply = await jarvis.chat(text)
            await say(reply[:3000])
        except Exception as exc:
            await say(f"Error: {exc}")

    @app.command("/jarvis")
    async def slash_jarvis(ack, respond, command):
        await ack()
        text = command.get("text", "").strip()
        if not text:
            await respond("Usage: `/jarvis <your message>`")
            return
        reply = await jarvis.chat(text)
        await respond(reply[:3000])

    handler = AsyncSocketModeHandler(app, cfg.SLACK_APP_TOKEN)
    print("[JARVIS Slack] Bot online.")
    await handler.start_async()
