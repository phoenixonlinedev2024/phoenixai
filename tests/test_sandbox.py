"""Tests for jarvis.sandbox — ExecResult, LocalSandbox, SandboxRouter."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal", "docker"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.sandbox.base import ExecResult, SandboxBackend  # noqa: E402
from jarvis.sandbox.local import LocalSandbox  # noqa: E402


# ── ExecResult ────────────────────────────────────────────────────────────────

def test_exec_result_ok_when_exit_zero():
    r = ExecResult(stdout="hello", exit_code=0)
    assert r.ok is True


def test_exec_result_not_ok_when_nonzero():
    r = ExecResult(exit_code=1)
    assert r.ok is False


def test_exec_result_not_ok_when_timed_out():
    r = ExecResult(exit_code=0, timed_out=True)
    assert r.ok is False


def test_exec_result_str_includes_stdout():
    r = ExecResult(stdout="output here", exit_code=0)
    assert "output here" in str(r)


def test_exec_result_str_includes_stderr():
    r = ExecResult(stderr="error here", exit_code=1)
    assert "STDERR" in str(r)
    assert "error here" in str(r)


def test_exec_result_str_includes_timeout():
    r = ExecResult(timed_out=True, exit_code=-1)
    assert "timed out" in str(r)


def test_exec_result_str_includes_exit_code():
    r = ExecResult(exit_code=42)
    assert "42" in str(r)


# ── LocalSandbox ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_local_sandbox_run_echo():
    sb = LocalSandbox()
    result = await sb.run("echo hello_world")
    assert result.ok
    assert "hello_world" in result.stdout


@pytest.mark.asyncio
async def test_local_sandbox_run_exit_code():
    sb = LocalSandbox()
    result = await sb.run("exit 42", timeout=5)
    assert result.exit_code == 42


@pytest.mark.asyncio
async def test_local_sandbox_run_timeout():
    sb = LocalSandbox()
    result = await sb.run("sleep 10", timeout=1)
    assert result.timed_out is True
    assert result.ok is False


@pytest.mark.asyncio
async def test_local_sandbox_run_code_python():
    sb = LocalSandbox()
    result = await sb.run_code("print('jarvis_test_output')", language="python")
    assert result.ok
    assert "jarvis_test_output" in result.stdout


@pytest.mark.asyncio
async def test_local_sandbox_run_code_unsupported_language():
    sb = LocalSandbox()
    result = await sb.run_code("val x = 1", language="scala")
    assert result.exit_code == 1
    assert "not supported" in result.stderr.lower()


@pytest.mark.asyncio
async def test_local_sandbox_write_and_read_file(tmp_path):
    sb = LocalSandbox()
    path = str(tmp_path / "test.txt")
    await sb.write_file(path, "JARVIS was here")
    content = await sb.read_file(path)
    assert content == "JARVIS was here"


@pytest.mark.asyncio
async def test_local_sandbox_write_creates_parents(tmp_path):
    sb = LocalSandbox()
    path = str(tmp_path / "deep" / "nested" / "file.txt")
    await sb.write_file(path, "deep content")
    content = await sb.read_file(path)
    assert content == "deep content"


@pytest.mark.asyncio
async def test_local_sandbox_health_check():
    sb = LocalSandbox()
    assert await sb.health_check() is True


def test_local_sandbox_repr():
    sb = LocalSandbox()
    assert "LocalSandbox" in repr(sb)


# ── SandboxRouter ─────────────────────────────────────────────────────────────

@pytest.fixture()
def router(monkeypatch):
    monkeypatch.setattr("jarvis.sandbox.router.cfg.SANDBOX_BACKEND", "local")
    monkeypatch.setattr("jarvis.sandbox.router.cfg.SSH_HOST", "")
    from jarvis.sandbox.router import SandboxRouter
    return SandboxRouter()


@pytest.mark.asyncio
async def test_router_get_backend_local(router):
    backend = await router.get_backend("local")
    assert backend.name == "local"


@pytest.mark.asyncio
async def test_router_run_delegates(router):
    result = await router.run("echo delegated")
    assert "delegated" in result.stdout


@pytest.mark.asyncio
async def test_router_run_code_python(router):
    result = await router.run_code("print(1+1)", language="python")
    assert "2" in result.stdout


def test_router_list_backends_includes_local(router):
    backends = router.list_backends()
    assert "local" in backends


@pytest.mark.asyncio
async def test_router_falls_back_when_preferred_unhealthy(monkeypatch):
    monkeypatch.setattr("jarvis.sandbox.router.cfg.SANDBOX_BACKEND", "docker")
    monkeypatch.setattr("jarvis.sandbox.router.cfg.SSH_HOST", "")
    from jarvis.sandbox.router import SandboxRouter
    from jarvis.sandbox.local import LocalSandbox

    r = SandboxRouter()
    # Make all non-local backends report unhealthy
    bad = MagicMock()
    bad.health_check = AsyncMock(return_value=False)
    for name in list(r._backends.keys()):
        if name != "local":
            r._backends[name] = bad

    backend = await r.get_backend("docker")
    assert isinstance(backend, LocalSandbox)
