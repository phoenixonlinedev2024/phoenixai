"""Tests for jarvis.bots.* — each bot's config-missing, import-missing, and
webhook behaviour. Keeps heavy SDK imports patched out so tests run without
network deps."""

from __future__ import annotations

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
