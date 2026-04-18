"""Base sandbox interface — all execution backends implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def __str__(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout.strip())
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr.strip()}")
        if self.timed_out:
            parts.append("(timed out)")
        parts.append(f"Exit: {self.exit_code}")
        return "\n".join(parts)


class SandboxBackend(ABC):
    name: str = "base"

    @abstractmethod
    async def run(self, command: str, timeout: int = 30, cwd: str = "/tmp") -> ExecResult:
        """Execute a shell command and return its result."""

    @abstractmethod
    async def run_code(self, code: str, language: str = "python", timeout: int = 30) -> ExecResult:
        """Execute a code snippet in the given language."""

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write a file inside the sandbox."""

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read a file from inside the sandbox."""

    async def health_check(self) -> bool:
        """Return True if this backend is available."""
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} sandbox>"
