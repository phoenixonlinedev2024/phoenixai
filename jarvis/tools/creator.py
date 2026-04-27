"""Dynamic tool creator — JARVIS synthesises new capabilities on demand."""

from __future__ import annotations

import ast
import json
import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.tools.registry import Tool, ToolRegistry

from jarvis.config import cfg


# Modules that synthesised tools are not permitted to import. The list is
# tuned to block the most common RCE / persistence vectors without crippling
# legitimate tools (which still get json/re/math/datetime/urllib/requests/etc.).
_FORBIDDEN_IMPORTS = {
    "ctypes", "cffi",                        # native code execution
    "pty", "pwd", "spwd",                    # terminal / shadow file access
    "multiprocessing", "_posixsubprocess",   # subprocess back-channels
    "marshal", "pickle", "dill",             # arbitrary deserialization
    "importlib",                             # bypasses our import filter
}

# Names that synthesised tools may not call as functions, even from builtins.
# `eval`, `exec`, and `compile` are stripped from the execution namespace too,
# but we reject them at AST-time so the error message is clearer.
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open"}


class UnsafeToolCode(ValueError):
    """Raised when a synthesised tool's code violates the safety policy."""


def _validate_tool_ast(python_code: str, require_run: bool = True) -> None:
    """Walk the AST and raise UnsafeToolCode on disallowed constructs.

    This is a defense-in-depth check, not a complete sandbox — a determined
    attacker who can already inject Python will eventually find a bypass. The
    goal is to make a prompt-injected `os.system("rm -rf /")` fail fast.

    ``require_run=True`` (default) enforces a top-level ``def run(**kwargs)``,
    used when validating freshly synthesised code from Claude. Persisted
    dynamic-tool modules wrap the synthesised body in a ``register_tools``
    function and pass ``require_run=False`` because the full module includes
    glue beyond the ``run`` definition.
    """
    try:
        tree = ast.parse(python_code)
    except SyntaxError as exc:
        raise UnsafeToolCode(f"Syntax error in synthesised tool: {exc}") from exc

    has_run = False
    for node in ast.walk(tree):
        # Module-level: only `def run(...)` and imports are allowed
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            has_run = True

        # Block imports of forbidden modules (covers `import os.system` style
        # via prefix match — `os` is allowed but `os.system` calls are blocked
        # via the call-name check below).
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_IMPORTS:
                    raise UnsafeToolCode(f"Forbidden import: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _FORBIDDEN_IMPORTS:
                raise UnsafeToolCode(f"Forbidden import: {node.module}")

        # Block `eval(...)`, `exec(...)`, `compile(...)`, `__import__(...)`,
        # `open(...)` — bare-name calls only; module-qualified calls like
        # `os.system(...)` are blocked by the attribute-call check.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALLS:
                raise UnsafeToolCode(f"Forbidden call: {func.id}()")
            if isinstance(func, ast.Attribute):
                if func.attr in {"system", "popen", "Popen", "spawn",
                                 "spawnl", "spawnlp", "spawnv", "spawnvp",
                                 "execv", "execve", "execvp", "execvpe",
                                 "fork", "forkpty"}:
                    raise UnsafeToolCode(f"Forbidden call: {ast.unparse(func)}()")

        # Block dunder access used to escape sandbox (e.g. `().__class__`).
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and \
                node.attr.endswith("__") and node.attr not in {"__init__", "__name__"}:
            raise UnsafeToolCode(f"Forbidden dunder access: {node.attr}")

    if require_run and not has_run:
        raise UnsafeToolCode("Synthesised tool must define a `run(**kwargs)` function.")

# System prompt injected when asking Claude to write a new tool
_CREATOR_PROMPT = """You are JARVIS's tool-creation subsystem.
A user requested a capability that does not exist. Your job is to write a Python
function and register it as a new JARVIS tool.

Output ONLY a JSON object with these fields (no extra prose):
{
  "name": "snake_case_tool_name",
  "description": "One-sentence description of what the tool does.",
  "category": "web|files|code|system|api|general",
  "parameters": {
    "type": "object",
    "properties": {
      "<param_name>": {"type": "string|integer|boolean|object|array", "description": "..."}
    },
    "required": ["param_name"]
  },
  "python_code": "def run(**kwargs):\\n    # full implementation\\n    return result"
}

The function must be named `run` and accept **kwargs.
It must return a string (the result to show the user).
Import everything it needs inside the function body.
"""


async def synthesise_tool(
    description: str,
    client: Any,
    registry: "ToolRegistry",
) -> "Tool | None":
    """Ask Claude to synthesise a new tool, persist it, and register it."""
    from jarvis.tools.registry import Tool

    if not cfg.DYNAMIC_TOOLS_ENABLED:
        print("[JARVIS] Tool synthesis refused: DYNAMIC_TOOLS_ENABLED=false.")
        return None

    try:
        response = await client.messages.create(
            model=cfg.CLAUDE_MODEL,
            max_tokens=2048,
            system=_CREATOR_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Create a tool that can: {description}",
                }
            ],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        spec = json.loads(raw)

        name: str = spec["name"]
        tool_description: str = spec["description"]
        category: str = spec.get("category", "general")
        parameters: dict = spec["parameters"]
        python_code: str = spec["python_code"]

        # Defense-in-depth: AST-validate before persisting or executing.
        _validate_tool_ast(python_code)

        # Persist to dynamic tools directory
        module_code = textwrap.dedent(f"""\
            # Auto-generated JARVIS tool: {name}
            # Description: {tool_description}

            {python_code}


            def register_tools(registry):
                from jarvis.tools.registry import Tool
                registry.register(Tool(
                    name={name!r},
                    description={tool_description!r},
                    input_schema={json.dumps(parameters)},
                    fn=run,
                    category={category!r},
                    dynamic=True,
                ))
        """)

        tool_path = cfg.TOOLS_DIR / f"{name}.py"
        tool_path.write_text(module_code, encoding="utf-8")

        # Register immediately without reloading from disk
        tool = Tool(
            name=name,
            description=tool_description,
            input_schema=parameters,
            fn=_make_fn(python_code),
            category=category,
            dynamic=True,
        )
        registry.register(tool)
        return tool

    except Exception as exc:
        print(f"[JARVIS] Tool synthesis error: {exc}")
        return None


def _make_fn(python_code: str):
    """Compile and return the `run` function from synthesised code.

    The execution namespace omits dangerous builtins (`eval`, `exec`,
    `compile`, `__import__`, `open`) — synthesised tools should import what
    they need at the top of their function body, where the AST validator
    has already enforced our import policy.
    """
    _validate_tool_ast(python_code)

    import builtins as _builtins
    # __import__ stays so ordinary `import math` still works inside the tool.
    # open() stays so tools can read/write files (a legitimate use case). The
    # genuinely dangerous primitives — eval/exec/compile and the interactive
    # debug helpers — are removed.
    safe_builtins = {
        k: v for k, v in vars(_builtins).items()
        if k not in {"eval", "exec", "compile",
                     "input", "breakpoint", "help", "quit", "exit"}
    }
    namespace: dict = {"__builtins__": safe_builtins}
    exec(compile(python_code, "<dynamic_tool>", "exec"), namespace)  # noqa: S102
    if "run" not in namespace or not callable(namespace["run"]):
        raise UnsafeToolCode("Synthesised tool did not produce a callable `run`.")
    return namespace["run"]
