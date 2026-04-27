"""Discord bot — JARVIS on Discord servers, for free."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis


async def run_discord_bot(jarvis: "Jarvis") -> None:
    """Start the JARVIS Discord bot."""
    if not cfg.DISCORD_TOKEN:
        print("[JARVIS Discord] DISCORD_TOKEN not set. Skipping.")
        return

    try:
        import discord
        from discord.ext import commands
    except ImportError:
        print("[JARVIS Discord] discord.py not installed.")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        print(f"[JARVIS Discord] Logged in as {bot.user}")

    @bot.command(name="jarvis")
    async def jarvis_cmd(ctx, *, message: str):
        async with ctx.typing():
            try:
                reply = await jarvis.chat(message)
                # Discord limit is 2000 chars
                if len(reply) > 1990:
                    for i in range(0, len(reply), 1990):
                        await ctx.send(reply[i:i+1990])
                else:
                    await ctx.send(reply)
            except Exception as exc:
                await ctx.send(f"Error: {exc}")

    @bot.command(name="jstatus")
    async def status_cmd(ctx):
        await ctx.send(f"```\n{jarvis.status()}\n```")

    @bot.command(name="jmemory")
    async def memory_cmd(ctx):
        lessons = jarvis.memory.get_lessons(limit=5)
        text = "Recent lessons:\n" + "\n".join(f"• {lesson}" for lesson in lessons) if lessons else "No lessons yet."
        await ctx.send(text)

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return
        # Respond when mentioned
        if bot.user in message.mentions:
            text = message.content.replace(f"<@{bot.user.id}>", "").strip()
            if text:
                async with message.channel.typing():
                    reply = await jarvis.chat(text)
                    await message.channel.send(reply[:1990])
        await bot.process_commands(message)

    print("[JARVIS Discord] Connecting...")
    await bot.start(cfg.DISCORD_TOKEN)
