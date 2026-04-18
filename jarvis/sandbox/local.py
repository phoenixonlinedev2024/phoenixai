"""Local sandbox — subprocess execution (default, no extra deps)."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from jarvis.sandbox.base import ExecResult, SandboxBackend

_LANG_CMD = {
    "python": [sys.executable],
    "javascript": ["node"],
    "bash": ["bash"],
    "sh": ["bash"],
    "ruby": ["ruby"],
    "go": ["go", "run"],
    "rust": None,  # handled specially
}

_LANG_EXT = {
    "python": ".py", "javascript": ".js", "bash": ".sh", "sh": ".sh",
    "ruby": ".rb", "go": ".go", "rust": ".rs",
}


class LocalSandbox(SandboxBackend):
    name = "local"

    async def run(self, command: str, timeout: int = 30, cwd: str = "/tmp") -> ExecResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return ExecResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(timed_out=True, exit_code=-1)
        except Exception as exc:
            return ExecResult(stderr=str(exc), exit_code=1)

    async def run_code(self, code: str, language: str = "python", timeout: int = 30) -> ExecResult:
        ext = _LANG_EXT.get(language, ".txt")
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name

        cmd_base = _LANG_CMD.get(language)
        if cmd_base is None:
            return ExecResult(stderr=f"Language '{language}' not supported in local sandbox.", exit_code=1)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_base, tmp,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return ExecResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            return ExecResult(timed_out=True, exit_code=-1)
        except Exception as exc:
            return ExecResult(stderr=str(exc), exit_code=1)
        finally:
            Path(tmp).unlink(missing_ok=True)

    async def write_file(self, path: str, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")
