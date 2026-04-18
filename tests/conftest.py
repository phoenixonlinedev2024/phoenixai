"""Shared fixtures for the JARVIS test suite."""

import os
import tempfile
from pathlib import Path

import pytest

# Ensure no real API keys are needed during tests
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")
os.environ.setdefault("SECURITY_ENABLED", "false")


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch):
    """Redirect all JARVIS data paths to a temp directory."""
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    # Patch cfg directly so already-imported modules see the new path
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MEMORY_DB", tmp_path / "jarvis_memory.db")
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path / "tools" / "dynamic")
    monkeypatch.setattr(cfg, "LOGS_DIR", tmp_path / "logs")
    (tmp_path / "tools" / "dynamic").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def memory_store(tmp_data_dir):
    from jarvis.memory.store import MemoryStore
    return MemoryStore(db_path=tmp_data_dir / "test.db")


@pytest.fixture
def tool_registry():
    from jarvis.tools.registry import build_registry
    return build_registry()
