"""Tests for jarvis.plugins.loader — plugin discovery, loading, and hot reload."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from jarvis.plugins.loader import PluginLoader


@pytest.fixture()
def registry():
    # A plain namespace so attribute access doesn't auto-create truthy Mocks.
    return SimpleNamespace()


@pytest.fixture()
def loader(registry, tmp_path, monkeypatch):
    # Point the module-level plugins dir at a temporary directory.
    monkeypatch.setattr("jarvis.plugins.loader.PLUGINS_DIR", tmp_path)
    return PluginLoader(registry)


# ── load_all ─────────────────────────────────────────────────────────────────

def test_load_all_skips_underscore_files(loader, tmp_path):
    (tmp_path / "_private.py").write_text("def register_tools(r): r.called = True\n")
    count = loader.load_all()
    assert count == 0


def test_load_all_loads_valid_plugin(loader, tmp_path, registry, capsys):
    plugin = tmp_path / "greeter.py"
    plugin.write_text(
        "def register_tools(registry):\n"
        "    registry.called = True\n"
    )
    count = loader.load_all()
    assert count == 1
    assert registry.called is True
    out = capsys.readouterr().out
    assert "greeter.py" in out


def test_load_all_skips_plugin_without_register_tools(loader, tmp_path):
    (tmp_path / "nohook.py").write_text("x = 42\n")
    count = loader.load_all()
    assert count == 0


def test_load_all_handles_broken_plugin(loader, tmp_path, capsys):
    (tmp_path / "broken.py").write_text("this is not valid python !!!\n")
    count = loader.load_all()
    assert count == 0
    out = capsys.readouterr().out
    assert "Failed to load" in out


def test_load_all_records_mtime(loader, tmp_path):
    plugin = tmp_path / "m.py"
    plugin.write_text("def register_tools(r): pass\n")
    loader.load_all()
    assert str(plugin) in loader._mtimes


# ── Hot reload lifecycle ─────────────────────────────────────────────────────

def test_start_and_stop_hot_reload(loader):
    loader.start_hot_reload(interval=0.05)
    assert loader._running is True
    assert loader._watcher_thread is not None
    assert loader._watcher_thread.is_alive()
    loader.stop_hot_reload()
    # Give the watcher thread a moment to exit its sleep
    time.sleep(0.2)
    assert loader._running is False


def test_hot_reload_detects_new_plugin(loader, tmp_path, registry):
    # Start watcher with empty directory
    loader.start_hot_reload(interval=0.05)
    try:
        # Create a new plugin file after watcher starts
        plugin = tmp_path / "late.py"
        plugin.write_text(
            "def register_tools(registry):\n"
            "    registry.late_called = True\n"
        )
        # Poll for the watcher to detect it
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if getattr(registry, "late_called", False):
                break
            time.sleep(0.05)
    finally:
        loader.stop_hot_reload()
    assert getattr(registry, "late_called", False) is True


def test_hot_reload_reloads_modified_plugin(loader, tmp_path, registry):
    plugin = tmp_path / "mutate.py"
    plugin.write_text(
        "def register_tools(registry):\n"
        "    registry.version = 1\n"
    )
    loader.load_all()
    assert registry.version == 1

    loader.start_hot_reload(interval=0.05)
    try:
        # Force mtime change: write new content and bump mtime
        plugin.write_text(
            "def register_tools(registry):\n"
            "    registry.version = 2\n"
        )
        import os
        future = time.time() + 10
        os.utime(plugin, (future, future))

        deadline = time.time() + 10.0
        while time.time() < deadline:
            if registry.version == 2:
                break
            time.sleep(0.05)
    finally:
        loader.stop_hot_reload()
    assert registry.version == 2
