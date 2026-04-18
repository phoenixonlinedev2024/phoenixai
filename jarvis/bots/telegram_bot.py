"""Telegram bot — JARVIS available on any device, for free."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis


async def run_telegram_bot(jarvis: "Jarvis") -> None:
    """Start the JARVIS Telegram bot."""
    if not cfg.TELEGRAM_TOKEN:
        print("[JARVIS Telegram] TELEGRAM_TOKEN not set. Skipping.")
        return

    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    except ImportError:
        print("[JARVIS Telegram] python-telegram-bot not installed.")
        return

    # Per-chat sessions
    _sessions: dict[int, str] = {}

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        jarvis.new_session()
        await update.message.reply_text(
            "JARVIS online. How may I assist you, Sir? "
            "Send any message to begin. Use /status for diagnostics."
        )

    async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(jarvis.status())

    async def memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        lessons = jarvis.memory.get_lessons(limit=5)
        text = "Recent lessons:\n" + "\n".join(f"• {l}" for l in lessons) if lessons else "No lessons yet."
        await update.message.reply_text(text)

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        user_text = update.message.text
        if not user_text:
            return
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        try:
            reply = await jarvis.chat(user_text)
            # Telegram max message length is 4096
            if len(reply) > 4096:
                for i in range(0, len(reply), 4096):
                    await update.message.reply_text(reply[i:i+4096])
            else:
                await update.message.reply_text(reply)
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")

    app = Application.builder().token(cfg.TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("memory", memory_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print(f"[JARVIS Telegram] Bot starting...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    print("[JARVIS Telegram] Bot online.")
