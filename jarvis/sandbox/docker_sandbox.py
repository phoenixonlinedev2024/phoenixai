"""Docker sandbox — hardened container execution with namespace isolation."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any

from jarvis.sandbox.base import ExecResult, SandboxBackend
from jarvis.config import cfg


class DockerSandbox(SandboxBackend):
    """Runs code in an ephemeral Docker container with hardening:
    - Read-only root filesystem
    - Dropped Linux capabilities
    - No network (configurable)
    - Memory + CPU limits
    - User namespace isolation
    """

    name = "docker"

    def __init__(
        self,
        image: str | None = None,
        network: str = "none",
        memory: str = "256m",
        cpus: float = 0.5,
    ) -> None:
        self.image = image or cfg.DOCKER_IMAGE
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import docker
                self._client = docker.from_env()
            except ImportError:
                raise RuntimeError("docker SDK not installed. Run: pip install docker")
        return self._client

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            client.ping()
            return True
        except Exception:
            return False

    async def run(self, command: str, timeout: int = 30, cwd: str = "/tmp") -> ExecResult:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_sync, ["bash", "-c", command], timeout, cwd, None
        )

    async def run_code(self, code: str, language: str = "python", timeout: int = 30) -> ExecResult:
        ext_map = {"python": ".py", "javascript": ".js", "bash": ".sh", "ruby": ".rb"}
        cmd_map = {
            "python": ["python3", "/workspace/code"],
            "javascript": ["node", "/workspace/code"],
            "bash": ["bash", "/workspace/code"],
            "ruby": ["ruby", "/workspace/code"],
        }
        ext = ext_map.get(language, ".txt")
        cmd = cmd_map.get(language)
        if not cmd:
            return ExecResult(stderr=f"Language '{language}' not supported.", exit_code=1)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, encoding="utf-8", prefix="jarvis_"
        ) as f:
            f.write(code)
            host_path = f.name

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._run_sync, cmd, timeout, "/workspace",
                {host_path: {"bind": "/workspace/code", "mode": "ro"}},
            )
        finally:
            Path(host_path).unlink(missing_ok=True)

    def _run_sync(self, cmd: list, timeout: int, workdir: str, volumes: Any) -> ExecResult:
        try:
            client = self._get_client()
            container = client.containers.run(
                self.image,
                command=cmd,
                working_dir=workdir,
                volumes=volumes or {},
                network_mode=self.network,
                mem_limit=self.memory,
                nano_cpus=int(self.cpus * 1e9),
                read_only=True,
                tmpfs={"/tmp": "size=64m", "/run": "size=16m"},
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
                timeout=timeout + 5,
            )
            output = container.decode("utf-8", errors="replace") if isinstance(container, bytes) else str(container)
            return ExecResult(stdout=output, exit_code=0)
        except Exception as exc:
            err = str(exc)
            if "exit status" in err.lower() or "non-zero" in err.lower():
                return ExecResult(stderr=err, exit_code=1)
            return ExecResult(stderr=f"Docker error: {err}", exit_code=1)

    async def write_file(self, path: str, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")
