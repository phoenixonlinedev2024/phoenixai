"""Sandbox execution tools — exposes the sandbox router as JARVIS tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry

_router = None


def _get_router():
    global _router
    if _router is None:
        from jarvis.sandbox.router import SandboxRouter
        _router = SandboxRouter()
    return _router


def _sandbox_run(command: str, backend: str = "local", timeout: int = 30) -> str:
    import asyncio
    try:
        router = _get_router()
        result = asyncio.run(router.run(command, backend=backend, timeout=timeout))
        return str(result)
    except Exception as exc:
        return f"Sandbox error: {exc}"


def _sandbox_run_code(code: str, language: str = "python",
                      backend: str = "local", timeout: int = 30) -> str:
    import asyncio
    try:
        router = _get_router()
        result = asyncio.run(router.run_code(code, language=language, backend=backend, timeout=timeout))
        return str(result)
    except Exception as exc:
        return f"Sandbox code error: {exc}"


def _list_sandbox_backends() -> str:
    router = _get_router()
    return "Available backends: " + ", ".join(router.list_backends())


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool

    registry.register(Tool(
        name="sandbox_run",
        description="Execute a shell command in a sandboxed environment. Choose backend: local, docker, ssh, modal, singularity.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "backend": {"type": "string", "default": "local",
                            "enum": ["local", "docker", "ssh", "modal", "singularity"]},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["command"],
        },
        fn=_sandbox_run,
        category="system",
    ))

    registry.register(Tool(
        name="sandbox_run_code",
        description="Execute code in a sandboxed environment with configurable backend and language.",
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "language": {"type": "string", "default": "python",
                             "enum": ["python", "javascript", "bash", "ruby", "go"]},
                "backend": {"type": "string", "default": "local",
                            "enum": ["local", "docker", "ssh", "modal", "singularity"]},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["code"],
        },
        fn=_sandbox_run_code,
        category="system",
    ))

    registry.register(Tool(
        name="list_sandbox_backends",
        description="List all available sandbox execution backends.",
        input_schema={"type": "object", "properties": {}},
        fn=_list_sandbox_backends,
        category="system",
    ))
