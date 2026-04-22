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


# ── DockerSandbox (client mocked) ─────────────────────────────────────────────

def test_docker_get_client_raises_without_sdk():
    from jarvis.sandbox.docker_sandbox import DockerSandbox
    sb = DockerSandbox()
    with patch.dict(sys.modules, {"docker": None}):
        sb._client = None
        with pytest.raises(RuntimeError, match="docker SDK not installed"):
            sb._get_client()


@pytest.mark.asyncio
async def test_docker_health_check_true(monkeypatch):
    from jarvis.sandbox.docker_sandbox import DockerSandbox
    sb = DockerSandbox()
    fake_client = MagicMock()
    fake_client.ping = MagicMock(return_value=True)
    sb._client = fake_client
    assert await sb.health_check() is True


@pytest.mark.asyncio
async def test_docker_health_check_false_on_error(monkeypatch):
    from jarvis.sandbox.docker_sandbox import DockerSandbox
    sb = DockerSandbox()
    fake_client = MagicMock()
    fake_client.ping = MagicMock(side_effect=RuntimeError("daemon down"))
    sb._client = fake_client
    assert await sb.health_check() is False


@pytest.mark.asyncio
async def test_docker_run_code_unsupported_language():
    from jarvis.sandbox.docker_sandbox import DockerSandbox
    sb = DockerSandbox()
    result = await sb.run_code("val x = 1", language="scala")
    assert result.exit_code == 1
    assert "not supported" in result.stderr.lower()


def test_docker_run_sync_wraps_output():
    from jarvis.sandbox.docker_sandbox import DockerSandbox
    sb = DockerSandbox()
    fake_client = MagicMock()
    fake_client.containers.run = MagicMock(return_value=b"hello from docker")
    sb._client = fake_client
    result = sb._run_sync(["echo", "hi"], timeout=5, workdir="/tmp", volumes=None)
    assert "hello from docker" in result.stdout
    assert result.exit_code == 0


def test_docker_run_sync_handles_exception():
    from jarvis.sandbox.docker_sandbox import DockerSandbox
    sb = DockerSandbox()
    fake_client = MagicMock()
    fake_client.containers.run = MagicMock(side_effect=RuntimeError("container failed"))
    sb._client = fake_client
    result = sb._run_sync(["bad"], timeout=5, workdir="/tmp", volumes=None)
    assert result.exit_code == 1
    assert "Docker error" in result.stderr


@pytest.mark.asyncio
async def test_docker_write_and_read_file(tmp_path):
    from jarvis.sandbox.docker_sandbox import DockerSandbox
    sb = DockerSandbox()
    path = str(tmp_path / "out.txt")
    await sb.write_file(path, "docker test")
    assert await sb.read_file(path) == "docker test"


# ── SSHSandbox (paramiko mocked) ──────────────────────────────────────────────

def test_ssh_connect_raises_without_paramiko():
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="localhost")
    with patch.dict(sys.modules, {"paramiko": None}):
        with pytest.raises(RuntimeError, match="paramiko not installed"):
            sb._connect()


def test_ssh_exec_returns_exec_result():
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="localhost", username="user")

    fake_conn = MagicMock()
    fake_stdout = MagicMock()
    fake_stdout.read.return_value = b"output"
    fake_stdout.channel.recv_exit_status.return_value = 0
    fake_stderr = MagicMock()
    fake_stderr.read.return_value = b""
    fake_conn.exec_command.return_value = (None, fake_stdout, fake_stderr)

    result = sb._exec("echo hello", timeout=5, conn=fake_conn)
    assert result.stdout == "output"
    assert result.exit_code == 0


def test_ssh_exec_handles_exception():
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="localhost", username="user")

    fake_conn = MagicMock()
    fake_conn.exec_command.side_effect = RuntimeError("channel closed")

    result = sb._exec("cmd", timeout=5, conn=fake_conn)
    assert result.exit_code == 1
    assert "channel closed" in result.stderr


@pytest.mark.asyncio
async def test_ssh_health_check_false_when_connect_fails(monkeypatch):
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="localhost")
    monkeypatch.setattr(sb, "_connect", lambda: (_ for _ in ()).throw(RuntimeError("refused")))
    assert await sb.health_check() is False


@pytest.mark.asyncio
async def test_ssh_health_check_true(monkeypatch):
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="localhost")
    fake_conn = MagicMock()
    monkeypatch.setattr(sb, "_connect", lambda: fake_conn)
    result = await sb.health_check()
    assert result is True
    fake_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_ssh_run_delegates_to_exec(monkeypatch):
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    from jarvis.sandbox.base import ExecResult
    sb = SSHSandbox(host="localhost", username="user")
    expected = ExecResult(stdout="ran", exit_code=0)
    monkeypatch.setattr(sb, "_exec", lambda cmd, timeout, conn=None: expected)
    result = await sb.run("echo hi", timeout=10)
    assert result.stdout == "ran"


@pytest.mark.asyncio
async def test_ssh_write_and_read_file(monkeypatch, tmp_path):
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="localhost", username="user")

    written = {}

    def fake_connect():
        fake_conn = MagicMock()
        fake_sftp = MagicMock()
        fake_file = MagicMock()
        fake_file.__enter__ = MagicMock(return_value=fake_file)
        fake_file.__exit__ = MagicMock(return_value=False)
        fake_file.write = MagicMock(side_effect=lambda data: written.update({"content": data}))
        fake_file.read = MagicMock(return_value=b"file content")
        fake_sftp.file = MagicMock(return_value=fake_file)
        fake_conn.open_sftp = MagicMock(return_value=fake_sftp)
        return fake_conn

    monkeypatch.setattr(sb, "_connect", fake_connect)
    await sb.write_file("/tmp/test.txt", "hello")
    assert written.get("content") == "hello"


@pytest.mark.asyncio
async def test_ssh_run_code_python(monkeypatch):
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    from jarvis.sandbox.base import ExecResult
    sb = SSHSandbox(host="localhost", username="user")
    expected = ExecResult(stdout="print result", exit_code=0)

    def fake_connect():
        fake_conn = MagicMock()
        fake_sftp = MagicMock()
        fake_file = MagicMock()
        fake_file.__enter__ = MagicMock(return_value=fake_file)
        fake_file.__exit__ = MagicMock(return_value=False)
        fake_sftp.file = MagicMock(return_value=fake_file)
        fake_conn.open_sftp = MagicMock(return_value=fake_sftp)
        return fake_conn

    monkeypatch.setattr(sb, "_connect", fake_connect)
    monkeypatch.setattr(sb, "_exec", lambda cmd, timeout, conn=None: expected)
    result = await sb.run_code("print('hi')", language="python", timeout=10)
    assert result.stdout == "print result"


# ── ModalSandbox ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_modal_health_check_true_when_modal_installed():
    from jarvis.sandbox.modal_sandbox import ModalSandbox
    sb = ModalSandbox()
    # modal is in sys.modules as a MagicMock (injected at top)
    result = await sb.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_modal_health_check_false_when_import_fails(monkeypatch):
    from jarvis.sandbox.modal_sandbox import ModalSandbox
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "modal":
            raise ImportError("no module named modal")
        return real_import(name, *args, **kwargs)

    sb = ModalSandbox()
    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = await sb.health_check()
    assert result is False
    monkeypatch.setattr(builtins, "__import__", real_import)


@pytest.mark.asyncio
async def test_modal_run_code_unsupported_language():
    from jarvis.sandbox.modal_sandbox import ModalSandbox
    sb = ModalSandbox()
    result = await sb.run_code("print('hi')", language="bash")
    assert result.exit_code == 1
    assert "Python only" in result.stderr


@pytest.mark.asyncio
async def test_modal_run_modal_error_returns_exec_result():
    from jarvis.sandbox.modal_sandbox import ModalSandbox
    sb = ModalSandbox()
    # _run_modal raises because mock modal doesn't have .App etc.
    result = sb._run_modal("echo hi", 30)
    assert isinstance(result.stderr, str)


@pytest.mark.asyncio
async def test_modal_write_and_read_file(tmp_path):
    from jarvis.sandbox.modal_sandbox import ModalSandbox
    sb = ModalSandbox()
    path = str(tmp_path / "modal_test.txt")
    await sb.write_file(path, "modal content")
    content = await sb.read_file(path)
    assert content == "modal content"


# ── SingularitySandbox ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_singularity_health_check_false_when_not_available():
    from jarvis.sandbox.modal_sandbox import SingularitySandbox
    sb = SingularitySandbox()
    # singularity binary doesn't exist in test env
    result = await sb.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_singularity_run_code_delegates_to_run(monkeypatch):
    from jarvis.sandbox.modal_sandbox import SingularitySandbox
    from jarvis.sandbox.base import ExecResult
    sb = SingularitySandbox()
    fake_result = ExecResult(stdout="code output", exit_code=0)

    async def fake_run(cmd, timeout=30):
        return fake_result

    monkeypatch.setattr(sb, "run", fake_run)
    result = await sb.run_code("print('hi')", language="python")
    assert result.stdout == "code output"


@pytest.mark.asyncio
async def test_singularity_write_and_read_file(tmp_path):
    from jarvis.sandbox.modal_sandbox import SingularitySandbox
    sb = SingularitySandbox()
    path = str(tmp_path / "sing_test.txt")
    await sb.write_file(path, "singularity data")
    content = await sb.read_file(path)
    assert content == "singularity data"


# ── SSHSandbox _connect with paramiko mocked ──────────────────────────────────

def test_ssh_connect_success_no_auth(monkeypatch):
    """Lines 34-42: _connect succeeds when paramiko is available, no key/password."""
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="host.example.com", username="user", key_path="", password="")

    fake_client = MagicMock()
    fake_paramiko = MagicMock()
    fake_paramiko.SSHClient.return_value = fake_client
    fake_paramiko.AutoAddPolicy = MagicMock()

    with patch.dict(sys.modules, {"paramiko": fake_paramiko}):
        result = sb._connect()

    assert result is fake_client
    fake_client.connect.assert_called_once()
    call_kwargs = fake_client.connect.call_args[1]
    assert "key_filename" not in call_kwargs
    assert "password" not in call_kwargs


def test_ssh_connect_uses_key_path(monkeypatch):
    """Line 38: _connect adds key_filename when key_path is set."""
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="h", username="u", key_path="/home/user/.ssh/id_rsa")

    fake_client = MagicMock()
    fake_paramiko = MagicMock()
    fake_paramiko.SSHClient.return_value = fake_client

    with patch.dict(sys.modules, {"paramiko": fake_paramiko}):
        sb._connect()

    kw = fake_client.connect.call_args[1]
    assert kw["key_filename"] == "/home/user/.ssh/id_rsa"


def test_ssh_connect_uses_password(monkeypatch):
    """Line 40: _connect adds password when password set and no key_path."""
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="h", username="u", key_path="", password="secret123")

    fake_client = MagicMock()
    fake_paramiko = MagicMock()
    fake_paramiko.SSHClient.return_value = fake_client

    with patch.dict(sys.modules, {"paramiko": fake_paramiko}):
        sb._connect()

    kw = fake_client.connect.call_args[1]
    assert kw["password"] == "secret123"
    assert "key_filename" not in kw


def test_ssh_exec_closes_own_connection(monkeypatch):
    """Lines 79, 90: _exec opens and closes its own connection when conn=None."""
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="h", username="u")

    fake_conn = MagicMock()
    fake_stdout = MagicMock()
    fake_stdout.read.return_value = b"result"
    fake_stdout.channel.recv_exit_status.return_value = 0
    fake_stderr = MagicMock()
    fake_stderr.read.return_value = b""
    fake_conn.exec_command.return_value = (None, fake_stdout, fake_stderr)

    monkeypatch.setattr(sb, "_connect", lambda: fake_conn)
    result = sb._exec("echo hi", timeout=5)  # conn=None → open+close

    assert result.exit_code == 0
    fake_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_ssh_read_file(monkeypatch):
    """Lines 103-111: read_file downloads via SFTP and returns content."""
    from jarvis.sandbox.ssh_sandbox import SSHSandbox
    sb = SSHSandbox(host="h", username="u")

    def fake_connect():
        fake_conn = MagicMock()
        fake_sftp = MagicMock()
        fake_file = MagicMock()
        fake_file.__enter__ = MagicMock(return_value=fake_file)
        fake_file.__exit__ = MagicMock(return_value=False)
        fake_file.read = MagicMock(return_value=b"remote file content")
        fake_sftp.file = MagicMock(return_value=fake_file)
        fake_conn.open_sftp = MagicMock(return_value=fake_sftp)
        return fake_conn

    monkeypatch.setattr(sb, "_connect", fake_connect)
    content = await sb.read_file("/remote/path.txt")
    assert content == "remote file content"


# ── ModalSandbox run() / run_code() executor delegation ──────────────────────

@pytest.mark.asyncio
async def test_modal_run_delegates_to_run_modal(monkeypatch):
    """Line 23: run() delegates to _run_modal via executor."""
    from jarvis.sandbox.modal_sandbox import ModalSandbox
    from jarvis.sandbox.base import ExecResult
    sb = ModalSandbox()
    expected = ExecResult(stdout="done", exit_code=0)
    monkeypatch.setattr(sb, "_run_modal", lambda cmd, timeout: expected)
    result = await sb.run("echo hi", timeout=10)
    assert result.stdout == "done"


@pytest.mark.asyncio
async def test_modal_run_code_delegates_to_run_modal_python(monkeypatch):
    """Line 30: run_code() delegates to _run_modal_python via executor."""
    from jarvis.sandbox.modal_sandbox import ModalSandbox
    from jarvis.sandbox.base import ExecResult
    sb = ModalSandbox()
    expected = ExecResult(stdout="42\n", exit_code=0)
    monkeypatch.setattr(sb, "_run_modal_python", lambda code, timeout: expected)
    result = await sb.run_code("print(42)", language="python")
    assert result.stdout == "42\n"


# ── ModalSandbox _run_modal success path (lines 42-50) ───────────────────────

def test_modal_run_modal_success(monkeypatch):
    """Lines 42-50: _run_modal returns ExecResult on modal API success."""
    from jarvis.sandbox.modal_sandbox import ModalSandbox

    fake_function_obj = MagicMock()
    fake_function_obj.remote = MagicMock(
        return_value={"stdout": "hello", "stderr": "", "exit_code": 0}
    )

    def fake_app_function(**kwargs):
        def decorator(fn):
            return fake_function_obj
        return decorator

    fake_app = MagicMock()
    fake_app.function = fake_app_function

    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=None)
    fake_ctx.__exit__ = MagicMock(return_value=False)

    fake_modal = MagicMock()
    fake_modal.App.lookup.return_value = fake_app
    fake_modal.Image.debian_slim.return_value.pip_install.return_value = MagicMock()
    fake_modal.runner.deploy_app = MagicMock(return_value=fake_ctx)

    sb = ModalSandbox()
    with patch.dict(sys.modules, {"modal": fake_modal}):
        result = sb._run_modal("echo hello", 30)

    assert result.exit_code == 0
    assert result.stdout == "hello"


# ── ModalSandbox _run_modal_python success path (lines 53-82) ────────────────

def test_modal_run_modal_python_success(monkeypatch):
    """Lines 53-82: _run_modal_python returns ExecResult on modal API success."""
    from jarvis.sandbox.modal_sandbox import ModalSandbox

    fake_exec_obj = MagicMock()
    fake_exec_obj.remote = MagicMock(
        return_value={"stdout": "42\n", "stderr": "", "exit_code": 0}
    )

    def fake_app_function(**kwargs):
        def decorator(fn):
            return fake_exec_obj
        return decorator

    fake_app = MagicMock()
    fake_app.function = fake_app_function

    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=None)
    fake_ctx.__exit__ = MagicMock(return_value=False)

    fake_modal = MagicMock()
    fake_modal.App.lookup.return_value = fake_app
    fake_modal.Image.debian_slim.return_value = MagicMock()
    fake_modal.runner.deploy_app = MagicMock(return_value=fake_ctx)

    sb = ModalSandbox()
    with patch.dict(sys.modules, {"modal": fake_modal}):
        result = sb._run_modal_python("print(42)", 30)

    assert result.exit_code == 0
    assert result.stdout == "42\n"


# ── SingularitySandbox health_check success (lines 107-108) ──────────────────

@pytest.mark.asyncio
async def test_singularity_health_check_true(monkeypatch):
    """Lines 107-108: health_check returns True when singularity exits 0."""
    from jarvis.sandbox.modal_sandbox import SingularitySandbox
    sb = SingularitySandbox()
    fake_proc = MagicMock()
    fake_proc.wait = AsyncMock(return_value=None)
    fake_proc.returncode = 0
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=fake_proc)):
        result = await sb.health_check()
    assert result is True


# ── SingularitySandbox run() (lines 113-128) ─────────────────────────────────

@pytest.mark.asyncio
async def test_singularity_run_success(monkeypatch):
    """Lines 113-124: run() executes command and returns ExecResult."""
    from jarvis.sandbox.modal_sandbox import SingularitySandbox
    sb = SingularitySandbox()
    fake_proc = MagicMock()
    fake_proc.communicate = AsyncMock(return_value=(b"output\n", b""))
    fake_proc.returncode = 0

    async def fake_wait_for(coro, timeout):
        return await coro

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=fake_proc)), \
         patch("asyncio.wait_for", fake_wait_for):
        result = await sb.run("echo hello")
    assert result.exit_code == 0
    assert "output" in result.stdout


@pytest.mark.asyncio
async def test_singularity_run_timeout(monkeypatch):
    """Lines 125-126: TimeoutError returns timed_out ExecResult."""
    from jarvis.sandbox.modal_sandbox import SingularitySandbox
    import asyncio as _asyncio
    sb = SingularitySandbox()
    fake_proc = MagicMock()
    fake_proc.communicate = AsyncMock(return_value=(b"", b""))

    async def fake_wait_for_timeout(coro, timeout):
        coro.close()
        raise _asyncio.TimeoutError()

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=fake_proc)), \
         patch("asyncio.wait_for", fake_wait_for_timeout):
        result = await sb.run("sleep 100", timeout=1)
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_singularity_run_exception(monkeypatch):
    """Lines 127-128: generic exception returns error ExecResult."""
    from jarvis.sandbox.modal_sandbox import SingularitySandbox
    sb = SingularitySandbox()

    async def raise_not_found(*args, **kwargs):
        raise FileNotFoundError("no singularity")

    with patch("asyncio.create_subprocess_exec", raise_not_found):
        result = await sb.run("echo hi")
    assert result.exit_code == 1
    assert "singularity" in result.stderr


# ── SandboxRouter _build_backends and get_backend fallback ───────────────────

def test_router_build_backends_includes_modal_and_singularity(monkeypatch):
    """Lines 42-51: _build_backends registers modal and singularity when imports succeed."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SSH_HOST", "")

    from jarvis.sandbox.router import SandboxRouter
    router = SandboxRouter()
    assert "modal" in router._backends
    assert "singularity" in router._backends


def test_router_build_backends_with_ssh_host(monkeypatch):
    """Lines 37-39: _build_backends registers SSH when SSH_HOST is configured."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SSH_HOST", "remote.example.com")

    from jarvis.sandbox.router import SandboxRouter
    router = SandboxRouter()
    assert "ssh" in router._backends


@pytest.mark.asyncio
async def test_router_get_backend_falls_back_to_local_when_all_fail(monkeypatch):
    """Line 63: returns self._local when no backend health check passes."""
    from jarvis.sandbox.router import SandboxRouter
    router = SandboxRouter()

    # Make all backends fail health checks
    for b in router._backends.values():
        b.health_check = AsyncMock(return_value=False)
    router._local.health_check = AsyncMock(return_value=False)

    result = await router.get_backend("nonexistent")
    assert result is router._local
