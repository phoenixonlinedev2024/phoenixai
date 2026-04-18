"""Package installer — JARVIS installs Python packages on demand to extend itself."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _install_package(package: str, upgrade: bool = False) -> str:
    """Install a Python package via pip."""
    args = [sys.executable, "-m", "pip", "install", package]
    if upgrade:
        args.append("--upgrade")
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return f"Successfully installed {package}.\n{result.stdout.strip()}"
        return f"Install failed:\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return f"Install of {package} timed out."
    except Exception as exc:
        return f"Install error: {exc}"


def _list_installed(filter_str: str = "") -> str:
    """List installed packages, optionally filtered."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            capture_output=True, text=True, timeout=30,
        )
        lines = result.stdout.strip().splitlines()
        if filter_str:
            lines = [l for l in lines if filter_str.lower() in l.lower()]
        return "\n".join(lines[:50]) or "No matching packages."
    except Exception as exc:
        return f"List error: {exc}"


def _check_package(package: str) -> str:
    """Check if a package is installed and return its version."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip() if result.returncode == 0 else f"{package} is not installed."
    except Exception as exc:
        return f"Check error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool

    registry.register(Tool(
        name="install_package",
        description="Install any Python package via pip to extend JARVIS capabilities.",
        input_schema={
            "type": "object",
            "properties": {
                "package": {"type": "string", "description": "Package name (e.g. 'numpy' or 'numpy==1.26')"},
                "upgrade": {"type": "boolean", "default": False},
            },
            "required": ["package"],
        },
        fn=_install_package,
        category="system",
    ))

    registry.register(Tool(
        name="list_packages",
        description="List installed Python packages, optionally filtered by name.",
        input_schema={
            "type": "object",
            "properties": {
                "filter_str": {"type": "string", "description": "Filter string (optional)"},
            },
        },
        fn=_list_installed,
        category="system",
    ))

    registry.register(Tool(
        name="check_package",
        description="Check if a Python package is installed and get its version info.",
        input_schema={
            "type": "object",
            "properties": {
                "package": {"type": "string"},
            },
            "required": ["package"],
        },
        fn=_check_package,
        category="system",
    ))
