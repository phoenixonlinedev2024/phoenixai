"""JARVIS CLI entry point — chat, daemon, and management commands."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(
    name="jarvis",
    help="JARVIS — Just A Rather Very Intelligent System for OpenClaw",
    add_completion=False,
)
console = Console()


def _banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]JARVIS[/bold cyan] — Just A Rather Very Intelligent System\n"
        "[dim]OpenClaw Universal AI Agent | Always On | Self-Improving[/dim]",
        border_style="cyan",
    ))


def _get_jarvis():
    from jarvis.core import Jarvis
    return Jarvis()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send (interactive if omitted)"),
    voice: bool = typer.Option(False, "--voice", "-v", help="Enable voice output"),
    session: Optional[str] = typer.Option(None, "--session", help="Resume a session by ID"),
) -> None:
    """Start an interactive chat with JARVIS (or send a single message)."""
    _banner()
    jarvis = _get_jarvis()

    async def _run():
        if message:
            with console.status("[cyan]JARVIS is thinking...[/cyan]"):
                reply = await jarvis.chat(message)
            console.print(Markdown(reply))
            return

        # Interactive loop
        console.print("[dim]Type your message and press Enter. Ctrl+C to exit.[/dim]\n")
        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                if not user_input.strip():
                    continue
                if user_input.lower() in ("exit", "quit", "bye"):
                    summary = await jarvis.end_session()
                    console.print(f"[dim]{summary}[/dim]")
                    break
                with console.status("[cyan]JARVIS...[/cyan]"):
                    reply = await jarvis.chat(user_input)
                console.print(f"\n[bold cyan]JARVIS:[/bold cyan]")
                console.print(Markdown(reply))
                console.print()

                if voice:
                    from jarvis.voice.text_to_speech import TTSEngine
                    tts = TTSEngine()
                    await tts.speak_async(reply)

            except KeyboardInterrupt:
                summary = await jarvis.end_session()
                console.print(f"\n[dim]{summary}[/dim]")
                break

    asyncio.run(_run())


@app.command()
def daemon() -> None:
    """Run JARVIS as a 24/7 background service (API + voice + scheduler)."""
    _banner()
    console.print("[cyan]Starting JARVIS daemon...[/cyan]")
    jarvis = _get_jarvis()

    async def _run():
        from jarvis.daemon import run_daemon
        await run_daemon(jarvis)

    asyncio.run(_run())


@app.command()
def status() -> None:
    """Show JARVIS system status."""
    jarvis = _get_jarvis()
    console.print(Panel(jarvis.status(), title="JARVIS Status", border_style="cyan"))


@app.command()
def tools() -> None:
    """List all registered JARVIS tools."""
    jarvis = _get_jarvis()
    from rich.table import Table
    table = Table(title="JARVIS Tool Registry", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Category", style="dim")
    table.add_column("Dynamic", style="yellow")
    table.add_column("Description")
    for t in sorted(jarvis.registry.all(), key=lambda x: x.category):
        table.add_row(t.name, t.category, "yes" if t.dynamic else "no", t.description)
    console.print(table)


@app.command()
def memory() -> None:
    """Show JARVIS memory: facts and lessons learned."""
    jarvis = _get_jarvis()
    facts = jarvis.memory.all_facts()
    lessons = jarvis.memory.get_lessons(limit=20)

    if facts:
        console.print(Panel("\n".join(f"  {f['key']}: {f['value']}" for f in facts),
                            title="Stored Facts", border_style="blue"))
    else:
        console.print("[dim]No facts stored yet.[/dim]")

    if lessons:
        console.print(Panel("\n".join(f"  - {l}" for l in lessons),
                            title="Lessons Learned", border_style="green"))
    else:
        console.print("[dim]No lessons stored yet.[/dim]")


@app.command()
def voice() -> None:
    """Start voice-only mode — speak to JARVIS using your microphone."""
    _banner()
    console.print("[cyan]Voice mode active. Say 'JARVIS' to wake me.[/cyan]")
    jarvis = _get_jarvis()

    async def _run():
        from jarvis.voice.speech_to_text import STTEngine
        from jarvis.voice.text_to_speech import TTSEngine
        from jarvis.personality import JARVIS_VOICE_INTRO

        tts = TTSEngine()
        stt = STTEngine()
        loop = asyncio.get_event_loop()
        tts.speak(JARVIS_VOICE_INTRO)

        def on_transcript(text: str):
            if text == "_wake_only_":
                import random
                from jarvis.personality import JARVIS_WAKE_RESPONSES
                tts.speak(random.choice(JARVIS_WAKE_RESPONSES))
                return
            console.print(f"[green]You:[/green] {text}")
            future = asyncio.run_coroutine_threadsafe(jarvis.voice_chat(text), loop)
            try:
                reply = future.result(timeout=60)
                console.print(f"[cyan]JARVIS:[/cyan] {reply}\n")
                tts.speak(reply)
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")

        stt.start_listening(on_transcript)
        console.print("[dim]Press Ctrl+C to stop.[/dim]")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            stt.stop_listening()
            console.print("\n[dim]Voice mode stopped.[/dim]")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
