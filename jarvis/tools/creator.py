"""Dynamic tool creator — JARVIS synthesises new capabilities on demand."""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.tools.registry import Tool, ToolRegistry

from jarvis.config import cfg

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
    """Compile and return the `run` function from synthesised code."""
    namespace: dict = {}
    exec(compile(python_code, "<dynamic_tool>", "exec"), namespace)  # noqa: S102
    return namespace["run"]
