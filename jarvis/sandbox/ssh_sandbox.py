"""SSH sandbox — execute code on a remote host via paramiko."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path

from jarvis.sandbox.base import ExecResult, SandboxBackend
from jarvis.config import cfg


class SSHSandbox(SandboxBackend):
    name = "ssh"

    def __init__(
        self,
        host: str | None = None,
        port: int = 22,
        username: str | None = None,
        key_path: str | None = None,
        password: str | None = None,
    ) -> None:
        self.host = host or cfg.SSH_HOST
        self.port = port or cfg.SSH_PORT
        self.username = username or cfg.SSH_USER
        self.key_path = key_path or cfg.SSH_KEY_PATH
        self.password = password or cfg.SSH_PASSWORD

    def _connect(self):
        try:
            import paramiko
        except ImportError:
            raise RuntimeError("paramiko not installed. Run: pip install paramiko")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = dict(hostname=self.host, port=self.port, username=self.username, timeout=10)
        if self.key_path:
            kwargs["key_filename"] = self.key_path
        elif self.password:
            kwargs["password"] = self.password
        client.connect(**kwargs)
        return client

    async def health_check(self) -> bool:
        try:
            conn = await asyncio.get_event_loop().run_in_executor(None, self._connect)
            conn.close()
            return True
        except Exception:
            return False

    async def run(self, command: str, timeout: int = 30, cwd: str = "/tmp") -> ExecResult:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._exec, f"cd {cwd} && {command}", timeout
        )

    async def run_code(self, code: str, language: str = "python", timeout: int = 30) -> ExecResult:
        ext_map = {"python": ".py", "javascript": ".js", "bash": ".sh"}
        run_map = {"python": "python3", "javascript": "node", "bash": "bash"}
        ext = ext_map.get(language, ".sh")
        runner = run_map.get(language, "bash")
        remote_path = f"/tmp/jarvis_{uuid.uuid4().hex[:8]}{ext}"

        def _upload_and_run():
            conn = self._connect()
            sftp = conn.open_sftp()
            with sftp.file(remote_path, "w") as f:
                f.write(code)
            sftp.close()
            result = self._exec(f"{runner} {remote_path}; rm -f {remote_path}", timeout, conn)
            conn.close()
            return result

        return await asyncio.get_event_loop().run_in_executor(None, _upload_and_run)

    def _exec(self, command: str, timeout: int, conn=None) -> ExecResult:
        close_conn = conn is None
        if conn is None:
            conn = self._connect()
        try:
            _, stdout, stderr = conn.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            code = stdout.channel.recv_exit_status()
            return ExecResult(stdout=out, stderr=err, exit_code=code)
        except Exception as exc:
            return ExecResult(stderr=str(exc), exit_code=1)
        finally:
            if close_conn:
                conn.close()

    async def write_file(self, path: str, content: str) -> None:
        def _upload():
            conn = self._connect()
            sftp = conn.open_sftp()
            with sftp.file(path, "w") as f:
                f.write(content)
            sftp.close()
            conn.close()
        await asyncio.get_event_loop().run_in_executor(None, _upload)

    async def read_file(self, path: str) -> str:
        def _download():
            conn = self._connect()
            sftp = conn.open_sftp()
            with sftp.file(path, "r") as f:
                data = f.read().decode("utf-8", errors="replace")
            sftp.close()
            conn.close()
            return data
        return await asyncio.get_event_loop().run_in_executor(None, _download)
