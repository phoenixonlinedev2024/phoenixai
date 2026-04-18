"""JARVIS CLI entry point."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

app = typer.Typer(
    name="jarvis",
    help="JARVIS — Just A Rather Very Intelligent System for OpenClaw",
    add_completion=False,
)
console = Console()


def _banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]J.A.R.V.I.S[/bold cyan] — Just A Rather Very Intelligent System\n"
        "[dim]OpenClaw Universal AI | Self-Improving | Always On[/dim]",
        border_style="cyan",
    ))


def _get_jarvis():
    from jarvis.core import Jarvis
    return Jarvis()


# ── chat ──────────────────────────────────────────────────────────────────

@app.command()
def chat(
    message: Optional[str] = typer.Argument(None),
    voice: bool = typer.Option(False, "--voice", "-v", help="Enable TTS output"),
    profile: str = typer.Option("default", "--profile", "-p",
                                help="Personality: default/professional/casual/terse/verbose"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream tokens in real time"),
) -> None:
    """Chat with JARVIS (interactive or one-shot)."""
    _banner()
    jarvis = _get_jarvis()
    jarvis.set_profile(profile)

    async def _run():
        tts = None
        if voice:
            from jarvis.voice.text_to_speech import TTSEngine
            tts = TTSEngine()

        if message:
            if stream:
                console.print(f"\n[bold cyan]JARVIS:[/bold cyan] ", end="")
                full = ""
                async for token in jarvis.stream_chat(message):
                    console.print(token, end="", highlight=False)
                    full += token
                console.print()
                if tts:
                    await tts.speak_async(full)
            else:
                with console.status("[cyan]JARVIS is thinking...[/cyan]"):
                    reply = await jarvis.chat(message)
                console.print(Markdown(reply))
                if tts:
                    await tts.speak_async(reply)
            return

        console.print("[dim]Interactive mode. Type 'exit' to quit.\n[/dim]")
        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                if not user_input.strip():
                    continue
                if user_input.lower() in ("exit", "quit", "bye"):
                    summary = await jarvis.end_session()
                    console.print(f"[dim]{summary}[/dim]")
                    break

                if stream:
                    console.print(f"\n[bold cyan]JARVIS:[/bold cyan] ", end="")
                    full = ""
                    async for token in jarvis.stream_chat(user_input):
                        console.print(token, end="", highlight=False)
                        full += token
                    console.print("\n")
                    if tts:
                        await tts.speak_async(full)
                else:
                    with console.status("[cyan]JARVIS...[/cyan]"):
                        reply = await jarvis.chat(user_input)
                    console.print(f"\n[bold cyan]JARVIS:[/bold cyan]")
                    console.print(Markdown(reply))
                    console.print()
                    if tts:
                        await tts.speak_async(reply)

            except KeyboardInterrupt:
                summary = await jarvis.end_session()
                console.print(f"\n[dim]{summary}[/dim]")
                break

    asyncio.run(_run())


# ── daemon ────────────────────────────────────────────────────────────────

@app.command()
def daemon() -> None:
    """Run JARVIS as a 24/7 service (API + WebSocket + Voice + Scheduler + Bots + Monitor)."""
    _banner()
    jarvis = _get_jarvis()

    async def _run():
        from jarvis.daemon import run_daemon
        await run_daemon(jarvis)

    asyncio.run(_run())


# ── voice ─────────────────────────────────────────────────────────────────

@app.command()
def voice() -> None:
    """Full voice interaction — say 'JARVIS' to wake, powered by Whisper STT."""
    _banner()
    console.print("[cyan]Voice mode active. Say 'JARVIS' to wake me.[/cyan]")
    jarvis = _get_jarvis()

    async def _run():
        from jarvis.voice.text_to_speech import TTSEngine
        from jarvis.personality import JARVIS_VOICE_INTRO, JARVIS_WAKE_RESPONSES

        tts = TTSEngine()

        if cfg.STT_ENGINE == "whisper":
            from jarvis.voice.whisper_stt import WhisperSTT
            stt = WhisperSTT(model_size=cfg.WHISPER_MODEL)
        else:
            from jarvis.voice.speech_to_text import STTEngine
            stt = STTEngine()

        loop = asyncio.get_event_loop()
        tts.speak(JARVIS_VOICE_INTRO)

        def on_transcript(text: str):
            import random
            if text == "_wake_only_":
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

    from jarvis.config import cfg
    asyncio.run(_run())


# ── status ────────────────────────────────────────────────────────────────

@app.command()
def status() -> None:
    """Show JARVIS system status."""
    jarvis = _get_jarvis()
    console.print(Panel(jarvis.status(), title="JARVIS Status", border_style="cyan"))


# ── tools ─────────────────────────────────────────────────────────────────

@app.command()
def tools() -> None:
    """List all registered tools."""
    jarvis = _get_jarvis()
    table = Table(title="JARVIS Tool Registry", border_style="cyan", show_lines=True)
    table.add_column("Name", style="bold")
    table.add_column("Category", style="dim")
    table.add_column("Dynamic", style="yellow")
    table.add_column("Description")
    for t in sorted(jarvis.registry.all(), key=lambda x: (x.category, x.name)):
        table.add_row(t.name, t.category, "✓" if t.dynamic else "", t.description[:70])
    console.print(table)


# ── memory ────────────────────────────────────────────────────────────────

@app.command()
def memory() -> None:
    """Show stored facts, lessons, and capability gaps."""
    jarvis = _get_jarvis()

    facts = jarvis.memory.all_facts()
    if facts:
        console.print(Panel("\n".join(f"  {f['key']}: {f['value']}" for f in facts),
                            title="Stored Facts", border_style="blue"))
    lessons = jarvis.memory.get_lessons(limit=20)
    if lessons:
        console.print(Panel("\n".join(f"  • {l}" for l in lessons),
                            title="Lessons Learned", border_style="green"))
    gaps = jarvis.memory.get_open_gaps()
    if gaps:
        console.print(Panel("\n".join(f"  [{g['id']}] {g['description']}" for g in gaps),
                            title="Open Capability Gaps", border_style="red"))
    console.print(f"\n[dim]{jarvis.memory.summary()}[/dim]")


# ── export ────────────────────────────────────────────────────────────────

@app.command()
def export(
    fmt: str = typer.Option("markdown", "--format", "-f", help="markdown or pdf"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
) -> None:
    """Export conversation history."""
    jarvis = _get_jarvis()
    if fmt == "pdf":
        from jarvis.export import export_pdf
        msg = export_pdf(jarvis.memory, jarvis._session_id, output)
    else:
        from jarvis.export import export_markdown
        msg = export_markdown(jarvis.memory, jarvis._session_id, output)
    console.print(msg)


# ── install-piper ─────────────────────────────────────────────────────────

@app.command()
def install_piper(
    model: str = typer.Option("en_US-lessac-medium", "--model", "-m", help="Piper voice model"),
    output_dir: str = typer.Option("./piper", "--dir", "-d"),
) -> None:
    """Download and set up Piper TTS (high-quality offline neural voice)."""
    import platform
    import subprocess
    import urllib.request

    system = platform.system().lower()
    arch = platform.machine().lower()

    release_map = {
        ("linux", "x86_64"): "piper_linux_x86_64.tar.gz",
        ("linux", "aarch64"): "piper_linux_aarch64.tar.gz",
        ("darwin", "x86_64"): "piper_macos_x86_64.tar.gz",
        ("darwin", "arm64"): "piper_macos_aarch64.tar.gz",
        ("windows", "amd64"): "piper_windows_amd64.zip",
    }

    key = (system, arch)
    if key not in release_map:
        console.print(f"[red]Unsupported platform: {system}/{arch}[/red]")
        raise typer.Exit(1)

    import tarfile
    import zipfile
    from pathlib import Path

    base_url = "https://github.com/rhasspy/piper/releases/latest/download"
    archive_name = release_map[key]
    archive_url = f"{base_url}/{archive_name}"
    model_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{model}.onnx"
    model_config_url = model_url + ".json"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / archive_name

    console.print(f"[cyan]Downloading Piper binary from {archive_url}...[/cyan]")
    try:
        urllib.request.urlretrieve(archive_url, archive_path)
    except Exception as exc:
        console.print(f"[red]Download failed: {exc}[/red]")
        raise typer.Exit(1)

    console.print("[cyan]Extracting archive...[/cyan]")
    if archive_name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(out)
    elif archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(out)
    archive_path.unlink(missing_ok=True)

    console.print(f"[cyan]Downloading voice model {model}...[/cyan]")
    try:
        urllib.request.urlretrieve(model_url, out / f"{model}.onnx")
        urllib.request.urlretrieve(model_config_url, out / f"{model}.onnx.json")
    except Exception as exc:
        console.print(f"[yellow]Model download warning: {exc}[/yellow]")

    binary = out / "piper" / ("piper.exe" if system == "windows" else "piper")
    console.print(f"[green]Piper installed to {out}[/green]")
    console.print(f"[dim]Add to .env:[/dim]")
    console.print(f"  PIPER_BINARY={binary}")
    console.print(f"  PIPER_MODEL={out / f'{model}.onnx'}")


# ── benchmark ────────────────────────────────────────────────────────────────

@app.command()
def benchmark(
    trend: bool = typer.Option(False, "--trend", "-t", help="Show pass-rate trend instead of running"),
) -> None:
    """Run JARVIS self-improvement benchmarks and report pass rate."""
    jarvis = _get_jarvis()

    async def _run():
        from jarvis.self_improve import BenchmarkRunner
        runner = BenchmarkRunner(jarvis)
        if trend:
            t = runner.trend()
            console.print(Panel(
                "\n".join(f"  {k}: {v}" for k, v in t.items()),
                title="Benchmark Trend", border_style="cyan",
            ))
            return

        with console.status("[cyan]Running benchmark suite...[/cyan]"):
            result = await runner.run_suite()

        table = Table(title="Benchmark Results", border_style="cyan", show_lines=True)
        table.add_column("Case", style="bold")
        table.add_column("Pass", style="green")
        table.add_column("Score")
        table.add_column("Latency")
        table.add_column("Error", style="red")
        for r in result["results"]:
            table.add_row(
                r["case_id"],
                "✓" if r["passed"] else "✗",
                f"{r['score']:.0%}",
                f"{r['latency_s']:.1f}s",
                r.get("error") or "",
            )
        console.print(table)
        console.print(
            f"\n[bold]Pass rate: {result['pass_rate']:.0%}[/bold]  "
            f"({result['passed']}/{result['total']} cases)  "
            f"avg latency: {result['avg_latency_s']:.1f}s"
        )

    asyncio.run(_run())


# ── keys ─────────────────────────────────────────────────────────────────────

@app.command()
def keys(
    create: bool = typer.Option(False, "--create", "-c", help="Generate a new API key"),
    name: str = typer.Option("", "--name", "-n"),
    role: str = typer.Option("user", "--role", "-r", help="user or admin"),
    revoke: Optional[str] = typer.Option(None, "--revoke", help="Key prefix to revoke"),
) -> None:
    """Manage JARVIS API keys (security module)."""
    from jarvis.security import key_store

    if revoke:
        ok = any(key_store.revoke(k) for k in list(key_store._keys) if k.startswith(revoke))
        console.print(f"[{'green' if ok else 'red'}]{'Revoked.' if ok else 'Key not found.'}[/]")
        return

    if create:
        raw = key_store.generate(name=name, role=role)
        console.print(Panel(
            f"[bold cyan]{raw}[/bold cyan]\n\n"
            f"Role: [yellow]{role}[/yellow]  Name: {name or '—'}\n\n"
            "[dim]Pass as header: X-Api-Key: <key>[/dim]",
            title="New API Key", border_style="green",
        ))
        return

    # List
    all_keys = key_store.list_keys()
    if not all_keys:
        console.print("[dim]No API keys provisioned. Use --create to generate one.[/dim]")
        return
    table = Table(title="API Keys", border_style="cyan")
    table.add_column("Prefix")
    table.add_column("Role", style="yellow")
    table.add_column("Name")
    table.add_column("Calls", justify="right")
    table.add_column("Last used")
    for k in all_keys:
        table.add_row(k["key_prefix"], k["role"], k["name"], str(k["calls"]), k["last_used"] or "never")
    console.print(table)


if __name__ == "__main__":
    app()
