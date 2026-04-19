"""Tests for jarvis.tools.package_installer and subagent_tools."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.tools.package_installer import (  # noqa: E402
    _check_package,
    _install_package,
    _list_installed,
)
import jarvis.tools.subagent_tools as subagent_tools  # noqa: E402


# ── _install_package ──────────────────────────────────────────────────────────

def test_install_package_success():
    mock_result = MagicMock(returncode=0, stdout="Successfully installed foo-1.0", stderr="")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _install_package("foo")
    assert "Successfully installed" in out


def test_install_package_failure():
    mock_result = MagicMock(returncode=1, stdout="", stderr="ERROR: No matching distribution")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _install_package("nonexistent_zzz_pkg")
    assert "failed" in out.lower() or "error" in out.lower()


def test_install_package_timeout():
    import subprocess
    with patch("jarvis.tools.package_installer.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=120)):
        out = _install_package("slow_pkg")
    assert "timed out" in out.lower()


def test_install_package_upgrade_flag():
    mock_result = MagicMock(returncode=0, stdout="", stderr="")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result) as mock_run:
        _install_package("foo", upgrade=True)
    args = mock_run.call_args[0][0]
    assert "--upgrade" in args


# ── _list_installed ───────────────────────────────────────────────────────────

def test_list_installed_returns_string():
    mock_result = MagicMock(returncode=0, stdout="pip                 23.0\nsetuptools          67.0\n")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _list_installed()
    assert "pip" in out or "setuptools" in out


def test_list_installed_with_filter():
    mock_result = MagicMock(returncode=0,
                             stdout="requests            2.31.0\npip                 23.0\n")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _list_installed(filter_str="requests")
    assert "requests" in out
    assert "pip" not in out


def test_list_installed_empty_filter_returns_all():
    mock_result = MagicMock(returncode=0, stdout="a   1.0\nb   2.0\n")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _list_installed(filter_str="")
    assert "a" in out and "b" in out


def test_list_installed_handles_error():
    with patch("jarvis.tools.package_installer.subprocess.run",
               side_effect=RuntimeError("pipe broken")):
        out = _list_installed()
    assert "error" in out.lower()


# ── _check_package ────────────────────────────────────────────────────────────

def test_check_package_installed():
    mock_result = MagicMock(returncode=0, stdout="Name: requests\nVersion: 2.31.0\n")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _check_package("requests")
    assert "requests" in out


def test_check_package_not_installed():
    mock_result = MagicMock(returncode=1, stdout="")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _check_package("totally_fake_pkg_xyz")
    assert "not installed" in out


def test_check_package_error():
    with patch("jarvis.tools.package_installer.subprocess.run",
               side_effect=RuntimeError("oops")):
        out = _check_package("foo")
    assert "error" in out.lower()


# ── subagent_tools ────────────────────────────────────────────────────────────

def test_delegate_no_pool():
    subagent_tools._pool = None
    out = subagent_tools._delegate("do something")
    assert "not initialised" in out.lower()


def test_delegate_parallel_no_pool():
    subagent_tools._pool = None
    out = subagent_tools._delegate_parallel([{"goal": "task1"}])
    assert "not initialised" in out.lower()


def test_delegate_with_pool_calls_dispatch_one():
    mock_pool = MagicMock()
    mock_pool.dispatch_one = MagicMock(return_value="task done")
    subagent_tools._pool = mock_pool

    import asyncio

    with patch.object(asyncio, "run", return_value="task done"):
        out = subagent_tools._delegate("write hello world")

    assert out == "task done"
    subagent_tools._pool = None  # reset


def test_delegate_parallel_with_pool():
    mock_pool = MagicMock()
    fake_results = {"abc123": "result A", "def456": "result B"}

    subagent_tools._pool = mock_pool
    import asyncio
    with patch.object(asyncio, "run", return_value=fake_results):
        out = subagent_tools._delegate_parallel([
            {"goal": "task A"},
            {"goal": "task B"},
        ])

    parsed = json.loads(out)
    assert "abc123" in parsed
    subagent_tools._pool = None  # reset


def test_register_tools_populates_registry():
    subagent_tools._pool = None
    from jarvis.tools.registry import build_registry
    registry = build_registry()
    subagent_tools.register_tools(registry)
    assert registry.get("delegate_task") is not None
    assert registry.get("delegate_parallel") is not None
