"""Email tools — IMAP read and SMTP send using stdlib only (free)."""

from __future__ import annotations

import email
import imaplib
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _send_email(to: str, subject: str, body: str, cc: str = "") -> str:
    if not cfg.EMAIL_ADDRESS or not cfg.EMAIL_PASSWORD:
        return "Email credentials not configured. Set EMAIL_ADDRESS and EMAIL_PASSWORD in .env"
    try:
        msg = MIMEMultipart()
        msg["From"] = cfg.EMAIL_ADDRESS
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc
        msg.attach(MIMEText(body, "plain"))
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg.SMTP_HOST, cfg.SMTP_PORT, context=context) as server:
            server.login(cfg.EMAIL_ADDRESS, cfg.EMAIL_PASSWORD)
            recipients = [to] + ([cc] if cc else [])
            server.sendmail(cfg.EMAIL_ADDRESS, recipients, msg.as_string())
        return f"Email sent to {to} with subject '{subject}'."
    except Exception as exc:
        return f"Email send error: {exc}"


def _read_emails(folder: str = "INBOX", limit: int = 10, unread_only: bool = True) -> str:
    if not cfg.EMAIL_ADDRESS or not cfg.EMAIL_PASSWORD:
        return "Email credentials not configured. Set EMAIL_ADDRESS and EMAIL_PASSWORD in .env"
    try:
        context = ssl.create_default_context()
        with imaplib.IMAP4_SSL(cfg.IMAP_HOST, context=context) as mail:
            mail.login(cfg.EMAIL_ADDRESS, cfg.EMAIL_PASSWORD)
            mail.select(folder)
            criteria = "UNSEEN" if unread_only else "ALL"
            _, data = mail.search(None, criteria)
            ids = data[0].split()
            ids = ids[-limit:]  # most recent
            results = []
            for eid in reversed(ids):
                _, msg_data = mail.fetch(eid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)
                sender = msg.get("From", "Unknown")
                subject = msg.get("Subject", "(no subject)")
                date = msg.get("Date", "")
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                            break
                else:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
                results.append(f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{body[:500]}\n---")
            return "\n".join(results) if results else "No emails found."
    except Exception as exc:
        return f"Email read error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool

    registry.register(Tool(
        name="send_email",
        description="Send an email via SMTP. Requires EMAIL_ADDRESS and EMAIL_PASSWORD in .env.",
        input_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "cc": {"type": "string", "description": "CC address (optional)"},
            },
            "required": ["to", "subject", "body"],
        },
        fn=_send_email,
        category="api",
    ))

    registry.register(Tool(
        name="read_emails",
        description="Read emails from IMAP inbox. Returns sender, subject, and first 500 chars of body.",
        input_schema={
            "type": "object",
            "properties": {
                "folder": {"type": "string", "default": "INBOX"},
                "limit": {"type": "integer", "default": 10},
                "unread_only": {"type": "boolean", "default": True},
            },
        },
        fn=_read_emails,
        category="api",
    ))
