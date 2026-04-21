"""Tests for jarvis.bots.* — each bot's config-missing, import-missing, and
webhook behaviour. Keeps heavy SDK imports patched out so tests run without
network deps."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Common fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()


@pytest.fixture()
def fake_jarvis():
    j = MagicMock()
    j.chat = AsyncMock(return_value="Yes, Sir.")
    j.status = MagicMock(return_value="JARVIS running")
    j.new_session = MagicMock()
    j.memory.get_lessons = MagicMock(return_value=["lesson one", "lesson two"])
    return j


# ── Telegram ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_telegram_skips_without_token(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.telegram_bot import run_telegram_bot
    monkeypatch.setattr(cfg, "TELEGRAM_TOKEN", "")
    await run_telegram_bot(fake_jarvis)
    assert "TELEGRAM_TOKEN not set" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_telegram_skips_when_sdk_missing(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.telegram_bot import run_telegram_bot
    monkeypatch.setattr(cfg, "TELEGRAM_TOKEN", "tok")
    # Force `from telegram import Update` to raise ImportError
    with patch.dict(sys.modules, {"telegram": None, "telegram.ext": None}):
        await run_telegram_bot(fake_jarvis)
    assert "python-telegram-bot not installed" in capsys.readouterr().out


# ── Discord ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discord_skips_without_token(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.discord_bot import run_discord_bot
    monkeypatch.setattr(cfg, "DISCORD_TOKEN", "")
    await run_discord_bot(fake_jarvis)
    assert "DISCORD_TOKEN not set" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_discord_skips_when_sdk_missing(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.discord_bot import run_discord_bot
    monkeypatch.setattr(cfg, "DISCORD_TOKEN", "tok")
    with patch.dict(sys.modules, {"discord": None, "discord.ext": None}):
        await run_discord_bot(fake_jarvis)
    assert "discord.py not installed" in capsys.readouterr().out


# ── Slack ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_slack_skips_without_tokens(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.slack_bot import run_slack_bot
    monkeypatch.setattr(cfg, "SLACK_BOT_TOKEN", "")
    monkeypatch.setattr(cfg, "SLACK_APP_TOKEN", "")
    await run_slack_bot(fake_jarvis)
    assert "SLACK_BOT_TOKEN" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_slack_skips_when_sdk_missing(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.slack_bot import run_slack_bot
    monkeypatch.setattr(cfg, "SLACK_BOT_TOKEN", "bot")
    monkeypatch.setattr(cfg, "SLACK_APP_TOKEN", "app")
    with patch.dict(sys.modules, {"slack_bolt": None, "slack_bolt.async_app": None,
                                   "slack_bolt.adapter": None,
                                   "slack_bolt.adapter.socket_mode": None,
                                   "slack_bolt.adapter.socket_mode.async_handler": None}):
        await run_slack_bot(fake_jarvis)
    assert "slack-bolt not installed" in capsys.readouterr().out


# ── Matrix ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_matrix_skips_without_config(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.matrix_bot import run_matrix_bot
    monkeypatch.setattr(cfg, "MATRIX_HOMESERVER", "")
    monkeypatch.setattr(cfg, "MATRIX_ACCESS_TOKEN", "")
    await run_matrix_bot(fake_jarvis)
    assert "MATRIX_HOMESERVER" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_matrix_skips_when_sdk_missing(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.matrix_bot import run_matrix_bot
    monkeypatch.setattr(cfg, "MATRIX_HOMESERVER", "https://matrix.org")
    monkeypatch.setattr(cfg, "MATRIX_ACCESS_TOKEN", "tok")
    with patch.dict(sys.modules, {"nio": None}):
        await run_matrix_bot(fake_jarvis)
    assert "matrix-nio not installed" in capsys.readouterr().out


# ── Mattermost ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mattermost_skips_without_config(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.mattermost_bot import run_mattermost_bot
    monkeypatch.setattr(cfg, "MATTERMOST_URL", "")
    monkeypatch.setattr(cfg, "MATTERMOST_TOKEN", "")
    await run_mattermost_bot(fake_jarvis)
    assert "MATTERMOST_URL" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_mattermost_skips_when_sdk_missing(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.mattermost_bot import run_mattermost_bot
    monkeypatch.setattr(cfg, "MATTERMOST_URL", "http://example.com")
    monkeypatch.setattr(cfg, "MATTERMOST_TOKEN", "tok")
    with patch.dict(sys.modules, {"websockets": None, "aiohttp": None}):
        await run_mattermost_bot(fake_jarvis)
    assert "websockets/aiohttp not installed" in capsys.readouterr().out


# ── IRC ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_irc_skips_without_server(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.irc_bot import run_irc_bot
    monkeypatch.setattr(cfg, "IRC_SERVER", "")
    await run_irc_bot(fake_jarvis)
    assert "IRC_SERVER not set" in capsys.readouterr().out


# ── Signal ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_signal_skips_without_config(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.signal_bot import run_signal_bot
    monkeypatch.setattr(cfg, "SIGNAL_PHONE_NUMBER", "")
    monkeypatch.setattr(cfg, "SIGNAL_CLI_PATH", "")
    await run_signal_bot(fake_jarvis)
    assert "SIGNAL_PHONE_NUMBER" in capsys.readouterr().out


def test_signal_cli_builds_command(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    from jarvis.bots.signal_bot import SignalBot
    monkeypatch.setattr(cfg, "SIGNAL_CLI_PATH", "/usr/bin/signal-cli")
    monkeypatch.setattr(cfg, "SIGNAL_PHONE_NUMBER", "+15551234")
    bot = SignalBot(fake_jarvis)
    cmd = bot._cli("send", "-m", "hi", "+15555678")
    assert cmd[0] == "/usr/bin/signal-cli"
    assert "+15551234" in cmd
    assert "send" in cmd


def test_signal_stop_sets_flag(fake_jarvis):
    from jarvis.bots.signal_bot import SignalBot
    bot = SignalBot(fake_jarvis)
    bot._running = True
    bot.stop()
    assert bot._running is False


@pytest.mark.asyncio
async def test_signal_handle_event_extracts_and_replies(monkeypatch, fake_jarvis):
    from jarvis.bots.signal_bot import SignalBot
    bot = SignalBot(fake_jarvis)
    sent = []

    async def fake_send(recipient, msg):
        sent.append((recipient, msg))

    bot.send = fake_send
    await bot._handle_event({
        "envelope": {
            "source": "+15550000",
            "dataMessage": {"message": "ping"},
        }
    })
    fake_jarvis.chat.assert_awaited_once_with("ping")
    assert sent == [("+15550000", "Yes, Sir.")]


@pytest.mark.asyncio
async def test_signal_handle_event_empty_skipped(fake_jarvis):
    from jarvis.bots.signal_bot import SignalBot
    bot = SignalBot(fake_jarvis)
    await bot._handle_event({"envelope": {}})
    fake_jarvis.chat.assert_not_awaited()


# ── WhatsApp webhooks ─────────────────────────────────────────────────────────

def _build_app_with_whatsapp(fake_jarvis):
    from fastapi import FastAPI
    from jarvis.bots.whatsapp_bot import mount_whatsapp_webhook
    app = FastAPI()
    mount_whatsapp_webhook(app, fake_jarvis)
    return app


def test_whatsapp_inbound_returns_503_without_twilio(monkeypatch, fake_jarvis):
    from fastapi.testclient import TestClient
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TWILIO_ACCOUNT_SID", "")
    app = _build_app_with_whatsapp(fake_jarvis)
    with TestClient(app) as client:
        resp = client.post("/whatsapp/webhook", data={"Body": "hi", "From": "whatsapp:+1"})
    assert resp.status_code == 503
    assert "Twilio not configured" in resp.text


def test_whatsapp_inbound_dispatches_to_jarvis(monkeypatch, fake_jarvis):
    from fastapi.testclient import TestClient
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TWILIO_ACCOUNT_SID", "SID")
    monkeypatch.setattr(cfg, "TWILIO_AUTH_TOKEN", "tok")
    monkeypatch.setattr(cfg, "TWILIO_WHATSAPP_FROM", "+15551000")
    with patch("jarvis.bots.whatsapp_bot._send_twilio_whatsapp") as send:
        app = _build_app_with_whatsapp(fake_jarvis)
        with TestClient(app) as client:
            resp = client.post("/whatsapp/webhook",
                               data={"Body": "hello", "From": "whatsapp:+15550002"})
    assert resp.status_code == 200
    fake_jarvis.chat.assert_awaited_once_with("hello")
    send.assert_called_once()


def test_whatsapp_inbound_empty_body_noop(monkeypatch, fake_jarvis):
    from fastapi.testclient import TestClient
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TWILIO_ACCOUNT_SID", "SID")
    app = _build_app_with_whatsapp(fake_jarvis)
    with TestClient(app) as client:
        resp = client.post("/whatsapp/webhook", data={"Body": "", "From": "whatsapp:+1"})
    assert resp.status_code == 200
    fake_jarvis.chat.assert_not_awaited()


def test_whatsapp_inbound_swallows_exception(monkeypatch, fake_jarvis):
    from fastapi.testclient import TestClient
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TWILIO_ACCOUNT_SID", "SID")
    fake_jarvis.chat = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("jarvis.bots.whatsapp_bot._send_twilio_whatsapp"):
        app = _build_app_with_whatsapp(fake_jarvis)
        with TestClient(app) as client:
            resp = client.post("/whatsapp/webhook",
                               data={"Body": "hi", "From": "whatsapp:+1"})
    assert resp.status_code == 200


def test_whatsapp_meta_verify_ok(monkeypatch, fake_jarvis):
    from fastapi.testclient import TestClient
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "WHATSAPP_VERIFY_TOKEN", "secret")
    app = _build_app_with_whatsapp(fake_jarvis)
    with TestClient(app) as client:
        resp = client.get("/whatsapp/webhook", params={
            "hub.verify_token": "secret",
            "hub.challenge": "ping-me-back",
        })
    assert resp.status_code == 200
    assert resp.text == "ping-me-back"


def test_whatsapp_meta_verify_forbidden(monkeypatch, fake_jarvis):
    from fastapi.testclient import TestClient
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "WHATSAPP_VERIFY_TOKEN", "secret")
    app = _build_app_with_whatsapp(fake_jarvis)
    with TestClient(app) as client:
        resp = client.get("/whatsapp/webhook", params={"hub.verify_token": "wrong"})
    assert resp.status_code == 403


def test_whatsapp_meta_incoming_dispatches(monkeypatch, fake_jarvis):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with patch("jarvis.bots.whatsapp_bot._send_meta_whatsapp") as send:
        app = _build_app_with_whatsapp(fake_jarvis)
        with TestClient(app) as client:
            resp = client.post("/whatsapp/meta", json={
                "entry": [{
                    "changes": [{
                        "value": {
                            "messages": [{
                                "from": "+15550009",
                                "text": {"body": "hello meta"},
                            }]
                        }
                    }]
                }]
            })
    assert resp.status_code == 200
    fake_jarvis.chat.assert_awaited_once_with("hello meta")
    send.assert_called_once()


def test_whatsapp_meta_no_messages_returns_ok(fake_jarvis):
    from fastapi.testclient import TestClient
    app = _build_app_with_whatsapp(fake_jarvis)
    with TestClient(app) as client:
        resp = client.post("/whatsapp/meta", json={"entry": [{"changes": [{"value": {}}]}]})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    fake_jarvis.chat.assert_not_awaited()


def test_whatsapp_meta_swallows_parse_error(fake_jarvis):
    from fastapi.testclient import TestClient
    app = _build_app_with_whatsapp(fake_jarvis)
    with TestClient(app) as client:
        # Send a payload shape that triggers KeyError/IndexError in the parser
        resp = client.post("/whatsapp/meta", json={"unexpected": True})
    assert resp.status_code == 200


def test_send_twilio_whatsapp_swallows_errors(monkeypatch, capsys):
    from jarvis.bots.whatsapp_bot import _send_twilio_whatsapp
    # Twilio is not installed in CI — function should catch ImportError and log.
    with patch.dict(sys.modules, {"twilio": None, "twilio.rest": None}):
        _send_twilio_whatsapp("whatsapp:+1", "hi")
    assert "Send error" in capsys.readouterr().out


def test_send_meta_whatsapp_swallows_errors(monkeypatch, capsys):
    from jarvis.bots.whatsapp_bot import _send_meta_whatsapp
    fake_requests = MagicMock()
    fake_requests.post = MagicMock(side_effect=RuntimeError("network dead"))
    with patch.dict(sys.modules, {"requests": fake_requests}):
        _send_meta_whatsapp("+1", "hi")
    assert "Send error" in capsys.readouterr().out


# ── IRC handler tests ─────────────────────────────────────────────────────────

def _make_irc_reader(*lines: bytes):
    """Return a mock reader whose readline() yields the given byte lines then CancelledError."""
    it = iter(lines)

    async def readline():
        try:
            return next(it)
        except StopIteration:
            raise asyncio.CancelledError()

    r = MagicMock()
    r.readline = readline
    return r


def _make_irc_writer():
    w = MagicMock()
    w.write = MagicMock()
    w.drain = AsyncMock()
    return w


@pytest.mark.asyncio
async def test_irc_responds_to_ping(monkeypatch, fake_jarvis, capsys):
    from jarvis.config import cfg
    from jarvis.bots.irc_bot import run_irc_bot

    monkeypatch.setattr(cfg, "IRC_SERVER", "irc.example.com")
    monkeypatch.setattr(cfg, "IRC_PORT", 6667)
    monkeypatch.setattr(cfg, "IRC_NICK", "jarvis")
    monkeypatch.setattr(cfg, "IRC_CHANNELS", "#test")

    reader = _make_irc_reader(b"PING :server.example.com\r\n")
    writer = _make_irc_writer()

    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(asyncio.CancelledError):
            await run_irc_bot(fake_jarvis)

    written = b"".join(call.args[0] for call in writer.write.call_args_list)
    assert b"PONG" in written


@pytest.mark.asyncio
async def test_irc_responds_to_privmsg_mention(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    from jarvis.bots.irc_bot import run_irc_bot

    monkeypatch.setattr(cfg, "IRC_SERVER", "irc.example.com")
    monkeypatch.setattr(cfg, "IRC_PORT", 6667)
    monkeypatch.setattr(cfg, "IRC_NICK", "jarvis")
    monkeypatch.setattr(cfg, "IRC_CHANNELS", "#test")

    reader = _make_irc_reader(b":user!u@h PRIVMSG #test :jarvis hello\r\n")
    writer = _make_irc_writer()

    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(asyncio.CancelledError):
            await run_irc_bot(fake_jarvis)

    fake_jarvis.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_irc_responds_to_pm(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    from jarvis.bots.irc_bot import run_irc_bot

    monkeypatch.setattr(cfg, "IRC_SERVER", "irc.example.com")
    monkeypatch.setattr(cfg, "IRC_PORT", 6667)
    monkeypatch.setattr(cfg, "IRC_NICK", "jarvis")
    monkeypatch.setattr(cfg, "IRC_CHANNELS", "")

    reader = _make_irc_reader(b":user!u@h PRIVMSG jarvis :direct pm\r\n")
    writer = _make_irc_writer()

    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(asyncio.CancelledError):
            await run_irc_bot(fake_jarvis)

    fake_jarvis.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_irc_ignores_own_messages(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    from jarvis.bots.irc_bot import run_irc_bot

    monkeypatch.setattr(cfg, "IRC_SERVER", "irc.example.com")
    monkeypatch.setattr(cfg, "IRC_PORT", 6667)
    monkeypatch.setattr(cfg, "IRC_NICK", "jarvis")
    monkeypatch.setattr(cfg, "IRC_CHANNELS", "")

    reader = _make_irc_reader(b":jarvis!j@h PRIVMSG #test :own message\r\n")
    writer = _make_irc_writer()

    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(asyncio.CancelledError):
            await run_irc_bot(fake_jarvis)

    fake_jarvis.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_irc_ignores_unmatched_privmsg(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    from jarvis.bots.irc_bot import run_irc_bot

    monkeypatch.setattr(cfg, "IRC_SERVER", "irc.example.com")
    monkeypatch.setattr(cfg, "IRC_PORT", 6667)
    monkeypatch.setattr(cfg, "IRC_NICK", "jarvis")
    monkeypatch.setattr(cfg, "IRC_CHANNELS", "")

    reader = _make_irc_reader(b":user!u@h PRIVMSG #test :no mention here\r\n")
    writer = _make_irc_writer()

    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(asyncio.CancelledError):
            await run_irc_bot(fake_jarvis)

    fake_jarvis.chat.assert_not_awaited()


# ── Discord handler tests ─────────────────────────────────────────────────────

def _setup_discord_bot(monkeypatch, fake_jarvis, capsys_fixture=None):
    """Run run_discord_bot with mocked SDK, return captured events and commands."""
    import asyncio as _asyncio
    from jarvis.config import cfg
    from jarvis.bots.discord_bot import run_discord_bot

    monkeypatch.setattr(cfg, "DISCORD_TOKEN", "tok")

    events = {}
    cmds = {}

    fake_bot = MagicMock()
    fake_bot.start = AsyncMock()
    fake_bot.process_commands = AsyncMock()
    fake_bot.user = MagicMock()
    fake_bot.user.id = 999

    def event_dec(fn):
        events[fn.__name__] = fn
        return fn

    def command_dec(**kwargs):
        name = kwargs.get("name", "")
        def inner(fn):
            cmds[name] = fn
            return fn
        return inner

    fake_bot.event = event_dec
    fake_bot.command = command_dec

    fake_discord = MagicMock()
    fake_commands = MagicMock()
    fake_commands.Bot.return_value = fake_bot
    # MagicMock hasattr always returns True, so `from discord.ext import commands`
    # resolves via attribute access on fake_ext — set it explicitly.
    fake_ext = MagicMock()
    fake_ext.commands = fake_commands

    return events, cmds, fake_bot, fake_discord, fake_ext, fake_commands, run_discord_bot


@pytest.mark.asyncio
async def test_discord_on_ready_prints_login(monkeypatch, fake_jarvis, capsys):
    events, cmds, fake_bot, fd, fe, fc, run_fn = _setup_discord_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, {"discord": fd, "discord.ext": fe, "discord.ext.commands": fc}):
        await run_fn(fake_jarvis)
    await events["on_ready"]()
    assert "Logged in" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_discord_jarvis_cmd_sends_reply(monkeypatch, fake_jarvis):
    events, cmds, fake_bot, fd, fe, fc, run_fn = _setup_discord_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, {"discord": fd, "discord.ext": fe, "discord.ext.commands": fc}):
        await run_fn(fake_jarvis)
    ctx = MagicMock()
    ctx.send = AsyncMock()
    await cmds["jarvis"](ctx, message="hello")
    ctx.send.assert_called_once()
    fake_jarvis.chat.assert_awaited_with("hello")


@pytest.mark.asyncio
async def test_discord_jarvis_cmd_long_reply_chunked(monkeypatch, fake_jarvis):
    fake_jarvis.chat = AsyncMock(return_value="x" * 2500)
    events, cmds, fake_bot, fd, fe, fc, run_fn = _setup_discord_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, {"discord": fd, "discord.ext": fe, "discord.ext.commands": fc}):
        await run_fn(fake_jarvis)
    ctx = MagicMock()
    ctx.send = AsyncMock()
    await cmds["jarvis"](ctx, message="write a lot")
    assert ctx.send.call_count > 1


@pytest.mark.asyncio
async def test_discord_jarvis_cmd_exception_sends_error(monkeypatch, fake_jarvis):
    fake_jarvis.chat = AsyncMock(side_effect=RuntimeError("boom"))
    events, cmds, fake_bot, fd, fe, fc, run_fn = _setup_discord_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, {"discord": fd, "discord.ext": fe, "discord.ext.commands": fc}):
        await run_fn(fake_jarvis)
    ctx = MagicMock()
    ctx.send = AsyncMock()
    await cmds["jarvis"](ctx, message="crash")
    assert "Error" in ctx.send.call_args[0][0]


@pytest.mark.asyncio
async def test_discord_status_cmd(monkeypatch, fake_jarvis):
    events, cmds, fake_bot, fd, fe, fc, run_fn = _setup_discord_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, {"discord": fd, "discord.ext": fe, "discord.ext.commands": fc}):
        await run_fn(fake_jarvis)
    ctx = MagicMock()
    ctx.send = AsyncMock()
    await cmds["jstatus"](ctx)
    ctx.send.assert_called_once()


@pytest.mark.asyncio
async def test_discord_memory_cmd(monkeypatch, fake_jarvis):
    events, cmds, fake_bot, fd, fe, fc, run_fn = _setup_discord_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, {"discord": fd, "discord.ext": fe, "discord.ext.commands": fc}):
        await run_fn(fake_jarvis)
    ctx = MagicMock()
    ctx.send = AsyncMock()
    await cmds["jmemory"](ctx)
    ctx.send.assert_called_once()
    assert "lesson" in ctx.send.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_discord_on_message_mention(monkeypatch, fake_jarvis):
    events, cmds, fake_bot, fd, fe, fc, run_fn = _setup_discord_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, {"discord": fd, "discord.ext": fe, "discord.ext.commands": fc}):
        await run_fn(fake_jarvis)

    msg = MagicMock()
    msg.author = MagicMock()  # not bot.user
    msg.author.__eq__ = MagicMock(return_value=False)
    msg.mentions = [fake_bot.user]
    msg.content = f"<@999> what's the weather?"
    msg.channel.typing = MagicMock()
    msg.channel.send = AsyncMock()
    await events["on_message"](msg)
    fake_jarvis.chat.assert_awaited()


@pytest.mark.asyncio
async def test_discord_on_message_own_message_ignored(monkeypatch, fake_jarvis):
    events, cmds, fake_bot, fd, fe, fc, run_fn = _setup_discord_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, {"discord": fd, "discord.ext": fe, "discord.ext.commands": fc}):
        await run_fn(fake_jarvis)

    msg = MagicMock()
    msg.author = fake_bot.user  # same object → equality True
    await events["on_message"](msg)
    fake_jarvis.chat.assert_not_awaited()


# ── Slack handler tests ───────────────────────────────────────────────────────

def _setup_slack_bot(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    from jarvis.bots.slack_bot import run_slack_bot

    monkeypatch.setattr(cfg, "SLACK_BOT_TOKEN", "bot-tok")
    monkeypatch.setattr(cfg, "SLACK_APP_TOKEN", "app-tok")

    slack_handlers = {}

    def fake_event(event_type):
        def dec(fn):
            slack_handlers[f"event:{event_type}"] = fn
            return fn
        return dec

    def fake_message(pattern):
        def dec(fn):
            slack_handlers[f"message:{pattern}"] = fn
            return fn
        return dec

    def fake_command(cmd):
        def dec(fn):
            slack_handlers[f"command:{cmd}"] = fn
            return fn
        return dec

    fake_app = MagicMock()
    fake_app.event = fake_event
    fake_app.message = fake_message
    fake_app.command = fake_command

    fake_AsyncApp = MagicMock(return_value=fake_app)
    fake_handler = MagicMock()
    fake_handler.start_async = AsyncMock()
    fake_AsyncSocketModeHandler = MagicMock(return_value=fake_handler)

    fake_bolt = MagicMock()
    fake_bolt_async = MagicMock()
    fake_bolt_async.AsyncApp = fake_AsyncApp
    fake_socket = MagicMock()
    fake_socket_handler = MagicMock()
    fake_socket_handler.AsyncSocketModeHandler = fake_AsyncSocketModeHandler

    return slack_handlers, run_slack_bot, {
        "slack_bolt": fake_bolt,
        "slack_bolt.async_app": fake_bolt_async,
        "slack_bolt.adapter": MagicMock(),
        "slack_bolt.adapter.socket_mode": fake_socket,
        "slack_bolt.adapter.socket_mode.async_handler": fake_socket_handler,
    }


@pytest.mark.asyncio
async def test_slack_handle_mention_with_text(monkeypatch, fake_jarvis):
    handlers, run_fn, mods = _setup_slack_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    say = AsyncMock()
    await handlers["event:app_mention"]({"text": "<@UBOT> hello"}, say)
    fake_jarvis.chat.assert_awaited()
    say.assert_called()


@pytest.mark.asyncio
async def test_slack_handle_mention_empty_text(monkeypatch, fake_jarvis):
    handlers, run_fn, mods = _setup_slack_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    say = AsyncMock()
    await handlers["event:app_mention"]({"text": "<@UBOT>"}, say)
    fake_jarvis.chat.assert_not_awaited()
    say.assert_called_once()


@pytest.mark.asyncio
async def test_slack_handle_message_keyword(monkeypatch, fake_jarvis):
    handlers, run_fn, mods = _setup_slack_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    say = AsyncMock()
    await handlers["message:jarvis"]({"text": "hey jarvis what's up"}, say)
    fake_jarvis.chat.assert_awaited()
    say.assert_called()


@pytest.mark.asyncio
async def test_slack_slash_command_with_text(monkeypatch, fake_jarvis):
    handlers, run_fn, mods = _setup_slack_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    ack = AsyncMock()
    respond = AsyncMock()
    await handlers["command:/jarvis"](ack, respond, {"text": "run diagnostics"})
    ack.assert_awaited_once()
    fake_jarvis.chat.assert_awaited()
    respond.assert_awaited()


@pytest.mark.asyncio
async def test_slack_slash_command_empty_text(monkeypatch, fake_jarvis):
    handlers, run_fn, mods = _setup_slack_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    ack = AsyncMock()
    respond = AsyncMock()
    await handlers["command:/jarvis"](ack, respond, {"text": ""})
    ack.assert_awaited_once()
    fake_jarvis.chat.assert_not_awaited()
    respond.assert_awaited_once()


# ── Telegram handler tests ────────────────────────────────────────────────────

def _setup_telegram_bot(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    from jarvis.bots.telegram_bot import run_telegram_bot

    monkeypatch.setattr(cfg, "TELEGRAM_TOKEN", "tok")

    registered_cmds = {}
    registered_msg = []

    def fake_cmd_handler(cmd, callback, **kw):
        registered_cmds[cmd] = callback
        return MagicMock()

    def fake_msg_handler(filters_expr, callback, **kw):
        registered_msg.append(callback)
        return MagicMock()

    fake_app = MagicMock()
    fake_app.initialize = AsyncMock()
    fake_app.start = AsyncMock()
    fake_app.updater = MagicMock()
    fake_app.updater.start_polling = AsyncMock()
    fake_app.add_handler = MagicMock()

    fake_builder = MagicMock()
    fake_builder.token.return_value.build.return_value = fake_app

    fake_Application = MagicMock()
    fake_Application.builder.return_value = fake_builder

    fake_telegram = MagicMock()
    fake_ext = MagicMock()
    fake_ext.Application = fake_Application
    fake_ext.CommandHandler = MagicMock(side_effect=fake_cmd_handler)
    fake_ext.MessageHandler = MagicMock(side_effect=fake_msg_handler)
    fake_ext.filters = MagicMock()
    fake_ext.ContextTypes = MagicMock()

    return registered_cmds, registered_msg, run_telegram_bot, {
        "telegram": fake_telegram,
        "telegram.ext": fake_ext,
    }


@pytest.mark.asyncio
async def test_telegram_start_handler(monkeypatch, fake_jarvis):
    cmds, msgs, run_fn, mods = _setup_telegram_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await cmds["start"](update, MagicMock())
    fake_jarvis.new_session.assert_called_once()
    update.message.reply_text.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_status_cmd(monkeypatch, fake_jarvis):
    cmds, msgs, run_fn, mods = _setup_telegram_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await cmds["status"](update, MagicMock())
    update.message.reply_text.assert_called_once_with("JARVIS running")


@pytest.mark.asyncio
async def test_telegram_memory_cmd_with_lessons(monkeypatch, fake_jarvis):
    cmds, msgs, run_fn, mods = _setup_telegram_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await cmds["memory"](update, MagicMock())
    text = update.message.reply_text.call_args[0][0]
    assert "lesson" in text.lower()


@pytest.mark.asyncio
async def test_telegram_memory_cmd_no_lessons(monkeypatch, fake_jarvis):
    fake_jarvis.memory.get_lessons = MagicMock(return_value=[])
    cmds, msgs, run_fn, mods = _setup_telegram_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await cmds["memory"](update, MagicMock())
    assert "No lessons" in update.message.reply_text.call_args[0][0]


@pytest.mark.asyncio
async def test_telegram_handle_message_normal(monkeypatch, fake_jarvis):
    cmds, msgs, run_fn, mods = _setup_telegram_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    update = MagicMock()
    update.message.text = "hello JARVIS"
    update.effective_chat.id = 1
    update.message.reply_text = AsyncMock()
    ctx = MagicMock()
    ctx.bot.send_chat_action = AsyncMock()
    await msgs[0](update, ctx)
    fake_jarvis.chat.assert_awaited_with("hello JARVIS")
    update.message.reply_text.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_handle_message_empty_text(monkeypatch, fake_jarvis):
    cmds, msgs, run_fn, mods = _setup_telegram_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    update = MagicMock()
    update.message.text = None
    await msgs[0](update, MagicMock())
    fake_jarvis.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_handle_message_error(monkeypatch, fake_jarvis):
    fake_jarvis.chat = AsyncMock(side_effect=RuntimeError("boom"))
    cmds, msgs, run_fn, mods = _setup_telegram_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)
    update = MagicMock()
    update.message.text = "crash"
    update.effective_chat.id = 1
    update.message.reply_text = AsyncMock()
    ctx = MagicMock()
    ctx.bot.send_chat_action = AsyncMock()
    await msgs[0](update, ctx)
    assert "Error" in update.message.reply_text.call_args[0][0]


# ── Matrix handler tests ──────────────────────────────────────────────────────

def _setup_matrix_bot(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    from jarvis.bots.matrix_bot import run_matrix_bot

    monkeypatch.setattr(cfg, "MATRIX_HOMESERVER", "https://matrix.org")
    monkeypatch.setattr(cfg, "MATRIX_ACCESS_TOKEN", "tok")
    monkeypatch.setattr(cfg, "MATRIX_USER_ID", "@jarvis:matrix.org")

    callbacks = []

    fake_client = MagicMock()
    fake_client.sync_forever = AsyncMock()
    fake_client.room_send = AsyncMock()
    fake_client.add_event_callback = MagicMock(side_effect=lambda fn, _cls: callbacks.append(fn))

    fake_AsyncClient = MagicMock(return_value=fake_client)
    fake_nio = MagicMock()
    fake_nio.AsyncClient = fake_AsyncClient

    return callbacks, fake_client, run_matrix_bot, {"nio": fake_nio}


@pytest.mark.asyncio
async def test_matrix_callback_responds_to_mention(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    callbacks, fake_client, run_fn, mods = _setup_matrix_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)

    room = MagicMock()
    room.room_id = "!abc:matrix.org"
    room.is_group = False

    event = MagicMock()
    event.sender = "@user:matrix.org"
    event.body = "@jarvis:matrix.org hello"

    await callbacks[0](room, event)
    fake_jarvis.chat.assert_awaited_once()
    fake_client.room_send.assert_called_once()


@pytest.mark.asyncio
async def test_matrix_callback_ignores_own_sender(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    callbacks, fake_client, run_fn, mods = _setup_matrix_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)

    room = MagicMock()
    event = MagicMock()
    event.sender = "@jarvis:matrix.org"
    event.body = "hello"

    await callbacks[0](room, event)
    fake_jarvis.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_matrix_callback_ignores_empty_body(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    callbacks, fake_client, run_fn, mods = _setup_matrix_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)

    room = MagicMock()
    event = MagicMock()
    event.sender = "@user:matrix.org"
    event.body = ""

    await callbacks[0](room, event)
    fake_jarvis.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_matrix_callback_ignores_public_room_without_mention(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    callbacks, fake_client, run_fn, mods = _setup_matrix_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)

    room = MagicMock()
    room.is_group = False  # public/multi-person room → only respond when mentioned

    event = MagicMock()
    event.sender = "@user:matrix.org"
    event.body = "no mention here"

    await callbacks[0](room, event)
    fake_jarvis.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_matrix_callback_chat_error_does_not_raise(monkeypatch, fake_jarvis):
    from jarvis.config import cfg
    fake_jarvis.chat = AsyncMock(side_effect=RuntimeError("api down"))
    callbacks, fake_client, run_fn, mods = _setup_matrix_bot(monkeypatch, fake_jarvis)
    with patch.dict(sys.modules, mods):
        await run_fn(fake_jarvis)

    room = MagicMock()
    room.room_id = "!x:m.org"
    room.is_group = False
    event = MagicMock()
    event.sender = "@user:matrix.org"
    event.body = "@jarvis:matrix.org oops"
    await callbacks[0](room, event)  # must not raise


# ── Mattermost handler tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mattermost_listen_handles_posted_event(monkeypatch, fake_jarvis):
    import json
    from jarvis.config import cfg
    from jarvis.bots.mattermost_bot import run_mattermost_bot

    monkeypatch.setattr(cfg, "MATTERMOST_URL", "http://mm.example.com")
    monkeypatch.setattr(cfg, "MATTERMOST_TOKEN", "tok")
    monkeypatch.setattr(cfg, "MATTERMOST_BOT_USER_ID", "bot_uid")
    monkeypatch.setattr(cfg, "MATTERMOST_BOT_NAME", "jarvis")

    post = {"message": "@jarvis hello", "channel_id": "chan1", "user_id": "user1"}
    event_data = {"event": "posted", "data": {"post": json.dumps(post)}}

    messages = [json.dumps(event_data)]
    it = iter(messages)

    async def fake_aiter(ws):
        for raw in messages:
            yield raw
        raise asyncio.CancelledError()

    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()
    fake_ws.__aiter__ = fake_aiter

    fake_ws_ctx = MagicMock()
    fake_ws_ctx.__aenter__ = AsyncMock(return_value=fake_ws)
    fake_ws_ctx.__aexit__ = AsyncMock(return_value=False)

    fake_websockets = MagicMock()
    fake_websockets.connect = MagicMock(return_value=fake_ws_ctx)

    fake_session = MagicMock()
    fake_session.post = AsyncMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(return_value=fake_session)

    with patch.dict(sys.modules, {"websockets": fake_websockets, "aiohttp": fake_aiohttp}), \
         patch("asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError())):
        with pytest.raises(asyncio.CancelledError):
            await run_mattermost_bot(fake_jarvis)

    fake_jarvis.chat.assert_awaited_once()


# ── Signal bot run() and send() (lines 27-58, 71-72) ────────────────────────

@pytest.mark.asyncio
async def test_signal_send_calls_subprocess(monkeypatch, fake_jarvis):
    """Lines 27-32: send() creates a subprocess and waits for it."""
    from jarvis.config import cfg
    from jarvis.bots.signal_bot import SignalBot

    monkeypatch.setattr(cfg, "SIGNAL_CLI_PATH", "/usr/bin/signal-cli")
    monkeypatch.setattr(cfg, "SIGNAL_PHONE_NUMBER", "+15551234")

    fake_proc = MagicMock()
    fake_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=fake_proc)):
        bot = SignalBot(fake_jarvis)
        await bot.send("+15559999", "hello there")

    fake_proc.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_signal_run_reads_lines_and_handles_events(monkeypatch, fake_jarvis):
    """Lines 39-54: run() starts daemon proc and processes JSON lines."""
    import json
    from jarvis.config import cfg
    from jarvis.bots.signal_bot import SignalBot

    monkeypatch.setattr(cfg, "SIGNAL_CLI_PATH", "/usr/bin/signal-cli")
    monkeypatch.setattr(cfg, "SIGNAL_PHONE_NUMBER", "+15551234")

    event = {"envelope": {"source": "+15550001", "dataMessage": {"message": "hi bot"}}}
    lines = [json.dumps(event).encode(), b""]
    line_idx = [0]

    async def fake_readline():
        val = lines[line_idx[0]]
        line_idx[0] += 1
        return val

    async def fake_wait_for(coro, timeout):
        return await coro

    fake_proc = MagicMock()
    fake_proc.returncode = None
    fake_proc.stdout.readline = fake_readline

    async def fake_send(recipient, msg):
        pass

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=fake_proc)), \
         patch("asyncio.wait_for", fake_wait_for):
        bot = SignalBot(fake_jarvis)
        bot.send = fake_send
        bot._running = True
        await bot.run()

    fake_jarvis.chat.assert_awaited_once_with("hi bot")


@pytest.mark.asyncio
async def test_signal_run_handles_timeout(monkeypatch, fake_jarvis):
    """Lines 55-56: TimeoutError from wait_for is caught and loop continues."""
    from jarvis.config import cfg
    from jarvis.bots.signal_bot import SignalBot

    monkeypatch.setattr(cfg, "SIGNAL_CLI_PATH", "/usr/bin/signal-cli")
    monkeypatch.setattr(cfg, "SIGNAL_PHONE_NUMBER", "+15551234")

    call_count = [0]

    async def fake_readline():
        return b""

    async def fake_wait_for(coro, timeout):
        call_count[0] += 1
        coro.close()
        if call_count[0] == 1:
            raise asyncio.TimeoutError()
        return b""  # empty → break loop on second call

    fake_proc = MagicMock()
    fake_proc.returncode = None
    fake_proc.stdout.readline = fake_readline

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=fake_proc)), \
         patch("asyncio.wait_for", fake_wait_for):
        bot = SignalBot(fake_jarvis)
        bot._running = True
        await bot.run()

    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_signal_handle_event_exception_is_caught(monkeypatch, fake_jarvis, capsys):
    """Lines 71-72: exception inside _handle_event is caught and printed."""
    from jarvis.bots.signal_bot import SignalBot

    fake_jarvis.chat = AsyncMock(side_effect=RuntimeError("api error"))
    bot = SignalBot(fake_jarvis)
    await bot._handle_event({
        "envelope": {"source": "+15550000", "dataMessage": {"message": "crash"}}
    })
    out = capsys.readouterr().out
    assert "Handle error" in out
