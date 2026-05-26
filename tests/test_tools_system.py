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


def test_delegate_parallel_pool_exception():
    """Pool raises — _delegate_parallel returns error string."""
    import asyncio
    mock_pool = MagicMock()
    subagent_tools._pool = mock_pool
    with patch.object(asyncio, "run", side_effect=RuntimeError("pool crashed")):
        out = subagent_tools._delegate_parallel([{"goal": "x"}])
    assert "error" in out.lower()
    subagent_tools._pool = None


def test_delegate_exception_returns_error_string():
    """_delegate returns error string when pool raises."""
    import asyncio
    mock_pool = MagicMock()
    subagent_tools._pool = mock_pool
    with patch.object(asyncio, "run", side_effect=RuntimeError("dispatch exploded")):
        out = subagent_tools._delegate("do something")
    assert "error" in out.lower()
    subagent_tools._pool = None


def test_register_tools_with_jarvis_creates_pool():
    """register_tools with a jarvis argument initialises _pool."""
    subagent_tools._pool = None
    fake_jarvis = MagicMock()
    from jarvis.tools.registry import build_registry
    registry = build_registry()
    with patch("jarvis.agents.subagent.SubagentPool") as mock_cls:
        mock_cls.return_value = MagicMock()
        subagent_tools.register_tools(registry, jarvis=fake_jarvis)
    assert subagent_tools._pool is not None
    subagent_tools._pool = None


# ── package_installer edge cases ─────────────────────────────────────────────

def test_install_package_generic_exception():
    """Exception other than TimeoutExpired is caught and reported."""
    with patch("jarvis.tools.package_installer.subprocess.run",
               side_effect=OSError("no such process")):
        out = _install_package("foo")
    assert "error" in out.lower()


def test_install_package_success_includes_package_name():
    mock_result = MagicMock(returncode=0, stdout="Successfully installed testpkg-1.2.3", stderr="")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _install_package("testpkg")
    assert "testpkg" in out


def test_list_installed_caps_at_50_lines():
    many_lines = "\n".join(f"pkg{i:03d}    1.0.0" for i in range(100))
    mock_result = MagicMock(returncode=0, stdout=many_lines)
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _list_installed()
    assert out.count("\n") <= 49  # 50 lines means at most 49 newlines


def test_check_package_timeout():
    import subprocess
    with patch("jarvis.tools.package_installer.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=15)):
        out = _check_package("slowpkg")
    assert "error" in out.lower()


def test_package_installer_register_tools_populates_registry():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.package_installer import register_tools
    registry = build_registry()
    register_tools(registry)
    assert registry.get("install_package") is not None
    assert registry.get("list_packages") is not None
    assert registry.get("check_package") is not None


# ── subagent_tools: _get_pool behaviour ──────────────────────────────────────

def test_get_pool_returns_none_when_no_jarvis():
    subagent_tools._pool = None
    pool = subagent_tools._get_pool()
    assert pool is None


def test_get_pool_returns_existing_pool():
    mock_pool = MagicMock()
    subagent_tools._pool = mock_pool
    pool = subagent_tools._get_pool()
    assert pool is mock_pool
    subagent_tools._pool = None


# ── package_installer: _list_installed with filter no match ──────────────────

def test_list_installed_filter_no_match():
    mock_result = MagicMock(returncode=0, stdout="pip     23.0\nsetuptools  67.0\n")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _list_installed(filter_str="totally_not_here_xyz")
    assert out == "No matching packages."


def test_install_package_no_upgrade_flag_by_default():
    mock_result = MagicMock(returncode=0, stdout="", stderr="")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result) as mock_run:
        _install_package("foo")
    args = mock_run.call_args[0][0]
    assert "--upgrade" not in args


def test_check_package_generic_exception():
    with patch("jarvis.tools.package_installer.subprocess.run",
               side_effect=OSError("permission denied")):
        out = _check_package("secretpkg")
    assert "error" in out.lower()


def test_install_package_failure_contains_stderr():
    mock_result = MagicMock(returncode=1, stdout="", stderr="No matching distribution found")
    with patch("jarvis.tools.package_installer.subprocess.run", return_value=mock_result):
        out = _install_package("bad_pkg_xyz")
    assert "No matching distribution" in out or "failed" in out.lower()
