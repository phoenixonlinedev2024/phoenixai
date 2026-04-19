"""Tests for jarvis.tools.email_tool and jarvis.tools.storage_tools."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.tools.email_tool import _read_emails, _send_email  # noqa: E402
from jarvis.tools.storage_tools import (  # noqa: E402
    _s3_delete,
    _s3_download,
    _s3_list,
    _s3_upload,
)


# ── _send_email ───────────────────────────────────────────────────────────────

def test_send_email_no_credentials(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "EMAIL_ADDRESS", "")
    monkeypatch.setattr(cfg, "EMAIL_PASSWORD", "")
    out = _send_email("alice@example.com", "hi", "hello")
    assert "not configured" in out.lower()


def test_send_email_success(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setattr(cfg, "EMAIL_PASSWORD", "secret")

    smtp_instance = MagicMock()
    smtp_ctx = MagicMock()
    smtp_ctx.__enter__ = MagicMock(return_value=smtp_instance)
    smtp_ctx.__exit__ = MagicMock(return_value=False)
    with patch("jarvis.tools.email_tool.smtplib.SMTP_SSL", return_value=smtp_ctx):
        out = _send_email("alice@example.com", "Greeting", "hello world")

    assert "sent to alice@example.com" in out.lower()
    smtp_instance.login.assert_called_once_with("me@example.com", "secret")
    smtp_instance.sendmail.assert_called_once()


def test_send_email_with_cc(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setattr(cfg, "EMAIL_PASSWORD", "secret")

    smtp_instance = MagicMock()
    smtp_ctx = MagicMock()
    smtp_ctx.__enter__ = MagicMock(return_value=smtp_instance)
    smtp_ctx.__exit__ = MagicMock(return_value=False)
    with patch("jarvis.tools.email_tool.smtplib.SMTP_SSL", return_value=smtp_ctx):
        _send_email("alice@example.com", "s", "b", cc="bob@example.com")

    recipients = smtp_instance.sendmail.call_args[0][1]
    assert "alice@example.com" in recipients
    assert "bob@example.com" in recipients


def test_send_email_exception(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setattr(cfg, "EMAIL_PASSWORD", "secret")

    with patch("jarvis.tools.email_tool.smtplib.SMTP_SSL",
               side_effect=RuntimeError("connection refused")):
        out = _send_email("alice@example.com", "s", "b")

    assert "send error" in out.lower()


# ── _read_emails ──────────────────────────────────────────────────────────────

def test_read_emails_no_credentials(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "EMAIL_ADDRESS", "")
    monkeypatch.setattr(cfg, "EMAIL_PASSWORD", "")
    out = _read_emails()
    assert "not configured" in out.lower()


def test_read_emails_no_messages(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setattr(cfg, "EMAIL_PASSWORD", "secret")

    mail = MagicMock()
    mail.search = MagicMock(return_value=("OK", [b""]))
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mail)
    ctx.__exit__ = MagicMock(return_value=False)
    with patch("jarvis.tools.email_tool.imaplib.IMAP4_SSL", return_value=ctx):
        out = _read_emails()

    assert "no emails" in out.lower()


def test_read_emails_returns_messages(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setattr(cfg, "EMAIL_PASSWORD", "secret")

    raw_msg = (
        b"From: alice@example.com\r\n"
        b"To: me@example.com\r\n"
        b"Subject: Hello\r\n"
        b"Date: Mon, 1 Jan 2024 10:00:00 +0000\r\n"
        b"Content-Type: text/plain\r\n\r\n"
        b"This is the email body."
    )

    mail = MagicMock()
    mail.search = MagicMock(return_value=("OK", [b"1"]))
    mail.fetch = MagicMock(return_value=("OK", [(b"1 (RFC822)", raw_msg)]))
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mail)
    ctx.__exit__ = MagicMock(return_value=False)
    with patch("jarvis.tools.email_tool.imaplib.IMAP4_SSL", return_value=ctx):
        out = _read_emails(folder="INBOX", limit=5, unread_only=False)

    assert "alice@example.com" in out
    assert "Hello" in out
    assert "This is the email body." in out


def test_read_emails_exception(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setattr(cfg, "EMAIL_PASSWORD", "secret")

    with patch("jarvis.tools.email_tool.imaplib.IMAP4_SSL",
               side_effect=RuntimeError("conn error")):
        out = _read_emails()

    assert "read error" in out.lower()


def test_read_emails_unread_only_criteria(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setattr(cfg, "EMAIL_PASSWORD", "secret")

    mail = MagicMock()
    mail.search = MagicMock(return_value=("OK", [b""]))
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mail)
    ctx.__exit__ = MagicMock(return_value=False)
    with patch("jarvis.tools.email_tool.imaplib.IMAP4_SSL", return_value=ctx):
        _read_emails(unread_only=True)

    _, criteria = mail.search.call_args[0]
    assert criteria == "UNSEEN"


# ── email register_tools ──────────────────────────────────────────────────────

def test_email_register_tools_populates_registry():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.email_tool import register_tools
    registry = build_registry()
    register_tools(registry)
    assert registry.get("send_email") is not None
    assert registry.get("read_emails") is not None


# ── storage_tools ─────────────────────────────────────────────────────────────

def _mock_boto3_client():
    """Return (patch_target, mock_client)."""
    mock_client = MagicMock()
    return mock_client


def test_s3_list_returns_keys():
    mock_client = MagicMock()
    mock_client.list_objects_v2 = MagicMock(return_value={
        "Contents": [
            {"Key": "foo.txt", "Size": 100},
            {"Key": "bar.txt", "Size": 250},
        ]
    })
    with patch("jarvis.tools.storage_tools._get_s3", return_value=mock_client):
        out = _s3_list("my-bucket")
    assert "foo.txt" in out
    assert "bar.txt" in out
    assert "100" in out


def test_s3_list_empty_bucket():
    mock_client = MagicMock()
    mock_client.list_objects_v2 = MagicMock(return_value={})
    with patch("jarvis.tools.storage_tools._get_s3", return_value=mock_client):
        out = _s3_list("empty-bucket")
    assert "Empty" in out


def test_s3_list_with_prefix():
    mock_client = MagicMock()
    mock_client.list_objects_v2 = MagicMock(return_value={"Contents": []})
    with patch("jarvis.tools.storage_tools._get_s3", return_value=mock_client):
        _s3_list("bkt", prefix="dir/")
    kwargs = mock_client.list_objects_v2.call_args.kwargs
    assert kwargs["Prefix"] == "dir/"


def test_s3_list_error():
    with patch("jarvis.tools.storage_tools._get_s3",
               side_effect=RuntimeError("auth failed")):
        out = _s3_list("bkt")
    assert "list error" in out.lower()


def test_s3_upload_success():
    mock_client = MagicMock()
    with patch("jarvis.tools.storage_tools._get_s3", return_value=mock_client):
        out = _s3_upload("/tmp/file.txt", "my-bucket")
    assert "Uploaded" in out
    assert "s3://my-bucket/file.txt" in out
    mock_client.upload_file.assert_called_once_with("/tmp/file.txt", "my-bucket", "file.txt")


def test_s3_upload_explicit_key():
    mock_client = MagicMock()
    with patch("jarvis.tools.storage_tools._get_s3", return_value=mock_client):
        out = _s3_upload("/tmp/file.txt", "bkt", key="custom/path.txt")
    assert "custom/path.txt" in out
    mock_client.upload_file.assert_called_once_with("/tmp/file.txt", "bkt", "custom/path.txt")


def test_s3_upload_error():
    with patch("jarvis.tools.storage_tools._get_s3",
               side_effect=RuntimeError("no creds")):
        out = _s3_upload("/tmp/a.txt", "bkt")
    assert "upload error" in out.lower()


def test_s3_download_default_dest():
    mock_client = MagicMock()
    with patch("jarvis.tools.storage_tools._get_s3", return_value=mock_client):
        out = _s3_download("bkt", "dir/file.txt")
    assert "Downloaded" in out
    assert "/tmp/file.txt" in out


def test_s3_download_custom_dest():
    mock_client = MagicMock()
    with patch("jarvis.tools.storage_tools._get_s3", return_value=mock_client):
        out = _s3_download("bkt", "file.txt", local_path="/home/user/out.txt")
    mock_client.download_file.assert_called_once_with("bkt", "file.txt", "/home/user/out.txt")
    assert "/home/user/out.txt" in out


def test_s3_download_error():
    with patch("jarvis.tools.storage_tools._get_s3",
               side_effect=RuntimeError("not found")):
        out = _s3_download("bkt", "k")
    assert "download error" in out.lower()


def test_s3_delete_success():
    mock_client = MagicMock()
    with patch("jarvis.tools.storage_tools._get_s3", return_value=mock_client):
        out = _s3_delete("bkt", "key.txt")
    assert "Deleted" in out
    mock_client.delete_object.assert_called_once_with(Bucket="bkt", Key="key.txt")


def test_s3_delete_error():
    with patch("jarvis.tools.storage_tools._get_s3",
               side_effect=RuntimeError("access denied")):
        out = _s3_delete("bkt", "k")
    assert "delete error" in out.lower()


def test_get_s3_without_boto3():
    """_get_s3 should raise a RuntimeError when boto3 isn't installed."""
    import jarvis.tools.storage_tools as st
    with patch.dict(sys.modules, {"boto3": None}):
        with pytest.raises(RuntimeError) as exc:
            st._get_s3()
    assert "boto3" in str(exc.value).lower()


def test_storage_register_tools_populates_registry():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.storage_tools import register_tools
    registry = build_registry()
    register_tools(registry)
    assert registry.get("s3_list") is not None
    assert registry.get("s3_upload") is not None
    assert registry.get("s3_download") is not None
    assert registry.get("s3_delete") is not None
