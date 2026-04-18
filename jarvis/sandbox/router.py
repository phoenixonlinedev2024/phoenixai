"""Sandbox router — selects the best available backend automatically."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from jarvis.config import cfg
from jarvis.sandbox.base import ExecResult, SandboxBackend
from jarvis.sandbox.local import LocalSandbox

if TYPE_CHECKING:
    pass


class SandboxRouter:
    """Tries backends in priority order, falls back to local if all fail."""

    def __init__(self) -> None:
        self._preferred = cfg.SANDBOX_BACKEND
        self._backends: dict[str, SandboxBackend] = {}
        self._local = LocalSandbox()
        self._build_backends()

    def _build_backends(self) -> None:
        from jarvis.sandbox.local import LocalSandbox
        self._backends["local"] = LocalSandbox()

        try:
            from jarvis.sandbox.docker_sandbox import DockerSandbox
            self._backends["docker"] = DockerSandbox()
        except Exception:
            pass

        try:
            from jarvis.sandbox.ssh_sandbox import SSHSandbox
            if cfg.SSH_HOST:
                self._backends["ssh"] = SSHSandbox()
        except Exception:
            pass

        try:
            from jarvis.sandbox.modal_sandbox import ModalSandbox
            self._backends["modal"] = ModalSandbox()
        except Exception:
            pass

        try:
            from jarvis.sandbox.modal_sandbox import SingularitySandbox
            self._backends["singularity"] = SingularitySandbox()
        except Exception:
            pass

    async def get_backend(self, name: str | None = None) -> SandboxBackend:
        name = name or self._preferred
        backend = self._backends.get(name)
        if backend and await backend.health_check():
            return backend
        # Fallback chain
        for fallback in ("docker", "ssh", "modal", "local"):
            b = self._backends.get(fallback)
            if b and await b.health_check():
                return b
        return self._local

    async def run(self, command: str, backend: str | None = None, timeout: int = 30) -> ExecResult:
        b = await self.get_backend(backend)
        return await b.run(command, timeout=timeout)

    async def run_code(self, code: str, language: str = "python",
                       backend: str | None = None, timeout: int = 30) -> ExecResult:
        b = await self.get_backend(backend)
        return await b.run_code(code, language=language, timeout=timeout)

    def list_backends(self) -> list[str]:
        return list(self._backends.keys())
