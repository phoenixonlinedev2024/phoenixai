"""Universal tool registry — every capability JARVIS can execute."""

from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

from jarvis.config import cfg


# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------

class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        fn: Callable[..., Any],
        category: str = "general",
        dynamic: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.fn = fn
        self.category = category
        self.dynamic = dynamic  # created at runtime by JARVIS

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def anthropic_tools(self) -> list[dict]:
        return [t.to_anthropic() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def load_dynamic_tools(self) -> None:
        """Load any tools saved to the dynamic tools directory."""
        cfg.TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        for path in cfg.TOOLS_DIR.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                if hasattr(mod, "register_tools"):
                    mod.register_tools(self)
            except Exception as exc:
                print(f"[JARVIS] Warning: failed to load dynamic tool {path.name}: {exc}")


# ---------------------------------------------------------------------------
# Built-in tool implementations
# ---------------------------------------------------------------------------

def _web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}\n")
        return "\n".join(lines)
    except Exception as exc:
        return f"Search error: {exc}"


def _web_fetch(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={"User-Agent": "JARVIS/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:8000] if len(text) > 8000 else text
    except Exception as exc:
        return f"Fetch error: {exc}"


def _read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        return f"Read error: {exc}"


def _write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except Exception as exc:
        return f"Write error: {exc}"


def _list_directory(path: str = ".") -> str:
    try:
        p = Path(path)
        items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = [f"{'[DIR] ' if i.is_dir() else '[FILE]'} {i.name}" for i in items]
        return "\n".join(lines) or "(empty)"
    except Exception as exc:
        return f"List error: {exc}"


def _run_shell(command: str, cwd: str | None = None, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"STDOUT:\n{out}")
        if err:
            parts.append(f"STDERR:\n{err}")
        parts.append(f"Exit code: {result.returncode}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except Exception as exc:
        return f"Shell error: {exc}"


def _execute_python(code: str) -> str:
    """Execute Python in a subprocess for isolation."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"Error:\n{err}")
        return "\n".join(parts) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Execution timed out."
    except Exception as exc:
        return f"Execution error: {exc}"


def _api_call(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: dict | None = None,
) -> str:
    try:
        import requests
        resp = requests.request(
            method.upper(),
            url,
            headers=headers or {},
            json=body,
            timeout=20,
        )
        try:
            return json.dumps(resp.json(), indent=2)
        except Exception:
            return resp.text[:4000]
    except Exception as exc:
        return f"API call error: {exc}"


def _get_system_info() -> str:
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return (
            f"CPU: {cpu}%\n"
            f"Memory: {mem.percent}% used ({mem.used // 1024**2} MB / {mem.total // 1024**2} MB)\n"
            f"Disk: {disk.percent}% used ({disk.used // 1024**3} GB / {disk.total // 1024**3} GB)"
        )
    except Exception as exc:
        return f"System info error: {exc}"


def _write_and_run_code(language: str, code: str, filename: str | None = None) -> str:
    """Write code to a temp file and execute it."""
    import tempfile
    suffix_map = {
        "python": ".py", "javascript": ".js", "bash": ".sh",
        "typescript": ".ts", "ruby": ".rb", "go": ".go",
    }
    suffix = suffix_map.get(language.lower(), ".txt")
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name

        runner_map = {
            ".py": [sys.executable, tmp],
            ".js": ["node", tmp],
            ".sh": ["bash", tmp],
            ".rb": ["ruby", tmp],
            ".go": ["go", "run", tmp],
        }
        cmd = runner_map.get(suffix)
        if not cmd:
            return f"No runner configured for {language}."

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"Error:\n{err}")
        return "\n".join(parts) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Execution timed out."
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Register all built-in tools
# ---------------------------------------------------------------------------

def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(Tool(
        name="web_search",
        description="Search the web for any information using DuckDuckGo.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5},
            },
            "required": ["query"],
        },
        fn=_web_search,
        category="web",
    ))

    reg.register(Tool(
        name="web_fetch",
        description="Fetch and extract readable text from any URL.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        },
        fn=_web_fetch,
        category="web",
    ))

    reg.register(Tool(
        name="read_file",
        description="Read the contents of any file on the filesystem.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
        fn=_read_file,
        category="files",
    ))

    reg.register(Tool(
        name="write_file",
        description="Write or overwrite content to any file. Creates parent directories as needed.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
        fn=_write_file,
        category="files",
    ))

    reg.register(Tool(
        name="list_directory",
        description="List files and directories at a given path.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current)", "default": "."},
            },
        },
        fn=_list_directory,
        category="files",
    ))

    reg.register(Tool(
        name="run_shell",
        description="Execute any shell command. Returns stdout, stderr, and exit code.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "cwd": {"type": "string", "description": "Working directory (optional)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)", "default": 30},
            },
            "required": ["command"],
        },
        fn=_run_shell,
        category="system",
    ))

    reg.register(Tool(
        name="execute_python",
        description="Execute a Python code snippet in an isolated subprocess and return output.",
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
        fn=_execute_python,
        category="code",
    ))

    reg.register(Tool(
        name="write_and_run_code",
        description="Write code in any language to a temp file and execute it. Supports python, javascript, bash, ruby, go.",
        input_schema={
            "type": "object",
            "properties": {
                "language": {"type": "string", "description": "Programming language"},
                "code": {"type": "string", "description": "Source code to run"},
                "filename": {"type": "string", "description": "Optional output filename"},
            },
            "required": ["language", "code"],
        },
        fn=_write_and_run_code,
        category="code",
    ))

    reg.register(Tool(
        name="api_call",
        description="Make any HTTP API request (REST/GraphQL). Supports GET, POST, PUT, DELETE, PATCH.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL"},
                "method": {"type": "string", "description": "HTTP method (default GET)", "default": "GET"},
                "headers": {"type": "object", "description": "HTTP headers dict"},
                "body": {"type": "object", "description": "Request body (JSON)"},
            },
            "required": ["url"],
        },
        fn=_api_call,
        category="api",
    ))

    reg.register(Tool(
        name="get_system_info",
        description="Get current CPU, memory, and disk usage of the system.",
        input_schema={
            "type": "object",
            "properties": {},
        },
        fn=_get_system_info,
        category="system",
    ))

    # Load any persisted dynamic tools
    reg.load_dynamic_tools()

    return reg
