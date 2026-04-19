"""Modal sandbox — serverless execution (free tier: 30GB-hrs/month).
Install: pip install modal && modal setup
"""

from __future__ import annotations

import asyncio

from jarvis.sandbox.base import ExecResult, SandboxBackend


class ModalSandbox(SandboxBackend):
    name = "modal"

    async def health_check(self) -> bool:
        try:
            import modal  # noqa: F401
            return True
        except ImportError:
            return False

    async def run(self, command: str, timeout: int = 60, cwd: str = "/tmp") -> ExecResult:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_modal, command, timeout
        )

    async def run_code(self, code: str, language: str = "python", timeout: int = 60) -> ExecResult:
        if language not in ("python",):
            return ExecResult(stderr=f"Modal sandbox supports Python only (got '{language}').", exit_code=1)
        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_modal_python, code, timeout
        )

    def _run_modal(self, command: str, timeout: int) -> ExecResult:
        try:
            import modal
            app = modal.App.lookup("jarvis-sandbox", create_if_missing=True)
            image = modal.Image.debian_slim().pip_install("requests")

            @app.function(image=image, timeout=timeout)
            def _shell(cmd: str) -> dict:
                import subprocess
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout - 5)
                return {"stdout": r.stdout, "stderr": r.stderr, "exit_code": r.returncode}

            with modal.runner.deploy_app(app):
                result = _shell.remote(command)
            return ExecResult(**result)
        except Exception as exc:
            return ExecResult(stderr=f"Modal error: {exc}", exit_code=1)

    def _run_modal_python(self, code: str, timeout: int) -> ExecResult:
        try:
            import modal
            app = modal.App.lookup("jarvis-sandbox-py", create_if_missing=True)
            image = modal.Image.debian_slim()

            @app.function(image=image, timeout=timeout)
            def _exec(src: str) -> dict:
                import sys
                import io
                import traceback
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                exit_code = 0
                try:
                    exec(compile(src, "<modal>", "exec"), {})
                except Exception:
                    sys.stderr.write(traceback.format_exc())
                    exit_code = 1
                finally:
                    out = sys.stdout.getvalue()
                    err = sys.stderr.getvalue()
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                return {"stdout": out, "stderr": err, "exit_code": exit_code}

            with modal.runner.deploy_app(app):
                result = _exec.remote(code)
            return ExecResult(**result)
        except Exception as exc:
            return ExecResult(stderr=f"Modal Python error: {exc}", exit_code=1)

    async def write_file(self, path: str, content: str) -> None:
        from pathlib import Path
        Path(path).write_text(content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        from pathlib import Path
        return Path(path).read_text(encoding="utf-8")


class SingularitySandbox(SandboxBackend):
    """Singularity/Apptainer — HPC-friendly container sandbox."""

    name = "singularity"

    def __init__(self, image: str = "docker://python:3.11-slim") -> None:
        self.image = image

    async def health_check(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "singularity", "--version",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception:
            return False

    async def run(self, command: str, timeout: int = 30, cwd: str = "/tmp") -> ExecResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                "singularity", "exec", self.image, "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
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

    async def run_code(self, code: str, language: str = "python", timeout: int = 30) -> ExecResult:
        return await self.run(f"python3 -c {repr(code)}", timeout=timeout)

    async def write_file(self, path: str, content: str) -> None:
        from pathlib import Path
        Path(path).write_text(content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        from pathlib import Path
        return Path(path).read_text(encoding="utf-8")
