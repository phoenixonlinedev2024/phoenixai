"""WhatsApp bot — JARVIS via Twilio WhatsApp Sandbox (free for testing)
or Meta Cloud API (free tier for production).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import PlainTextResponse

from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis
    from fastapi import FastAPI


def mount_whatsapp_webhook(app: "FastAPI", jarvis: "Jarvis") -> None:
    """Mount WhatsApp webhook routes onto an existing FastAPI app."""

    @app.post("/whatsapp/webhook")
    async def whatsapp_incoming(request: Request):
        """Twilio WhatsApp webhook — receives inbound messages."""
        if not cfg.TWILIO_ACCOUNT_SID:
            return PlainTextResponse("Twilio not configured.", status_code=503)
        try:
            form = await request.form()
            body = form.get("Body", "").strip()
            from_ = form.get("From", "")
            if not body:
                return PlainTextResponse("", status_code=200)

            reply = await jarvis.chat(body)
            _send_twilio_whatsapp(from_, reply[:1500])
            return PlainTextResponse("", status_code=200)
        except Exception as exc:
            print(f"[JARVIS WhatsApp] Error: {exc}")
            return PlainTextResponse("", status_code=200)

    @app.get("/whatsapp/webhook")
    async def whatsapp_verify(request: Request):
        """Meta Cloud API webhook verification."""
        params = request.query_params
        if params.get("hub.verify_token") == cfg.WHATSAPP_VERIFY_TOKEN:
            return PlainTextResponse(params.get("hub.challenge", ""))
        return PlainTextResponse("Forbidden", status_code=403)

    @app.post("/whatsapp/meta")
    async def whatsapp_meta(request: Request):
        """Meta WhatsApp Cloud API incoming messages."""
        try:
            data = await request.json()
            entry = data.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            messages = value.get("messages", [])
            if not messages:
                return {"status": "ok"}
            msg = messages[0]
            from_ = msg.get("from")
            text = msg.get("text", {}).get("body", "")
            if not text:
                return {"status": "ok"}
            reply = await jarvis.chat(text)
            _send_meta_whatsapp(from_, reply[:4096])
            return {"status": "ok"}
        except Exception as exc:
            print(f"[JARVIS WhatsApp Meta] Error: {exc}")
            return {"status": "ok"}


def _send_twilio_whatsapp(to: str, message: str) -> None:
    try:
        from twilio.rest import Client
        client = Client(cfg.TWILIO_ACCOUNT_SID, cfg.TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=f"whatsapp:{cfg.TWILIO_WHATSAPP_FROM}",
            to=to,
        )
    except Exception as exc:
        print(f"[JARVIS WhatsApp] Send error: {exc}")


def _send_meta_whatsapp(to: str, message: str) -> None:
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {cfg.WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": message},
        }
        requests.post(
            f"https://graph.facebook.com/v18.0/{cfg.WHATSAPP_PHONE_ID}/messages",
            headers=headers,
            json=payload,
            timeout=15,
        )
    except Exception as exc:
        print(f"[JARVIS WhatsApp Meta] Send error: {exc}")
