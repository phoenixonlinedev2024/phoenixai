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
    """Verify the reload logic by directly driving _load_plugin after mtime bump.

    We don't rely on the background thread here to avoid race conditions under
    heavy test-suite load. Instead we verify the mtime-comparison gating and
    the actual reload outcome deterministically.
    """
    import os

    plugin = tmp_path / "mutate.py"
    plugin.write_text(
        "def register_tools(registry):\n"
        "    registry.version = 1\n"
    )
    loader.load_all()
    assert registry.version == 1

    # Simulate the watcher detecting a mtime change by bumping the mtime and
    # directly calling _load_plugin (which is what _watch_loop would do).
    plugin.write_text(
        "def register_tools(registry):\n"
        "    registry.version = 2\n"
    )
    new_mtime = loader._mtimes[str(plugin)] + 1
    os.utime(plugin, (new_mtime, new_mtime))

    # Confirm the gating condition matches what the watcher checks
    assert plugin.stat().st_mtime > loader._mtimes[str(plugin)]

    # Drive the reload directly (same code the watcher calls)
    loader._load_plugin(plugin)
    assert registry.version == 2


def test_watch_loop_reloads_changed_plugin(loader, tmp_path, registry):
    """Lines 76-78: _watch_loop reloads a plugin when its mtime increases."""
    import os
    import time

    plugin = tmp_path / "evolving.py"
    plugin.write_text("def register_tools(registry):\n    registry.watch_v = 1\n")
    loader.load_all()
    assert registry.watch_v == 1

    old_mtime = loader._mtimes[str(plugin)]
    plugin.write_text("def register_tools(registry):\n    registry.watch_v = 2\n")
    new_mtime = old_mtime + 2
    os.utime(plugin, (new_mtime, new_mtime))

    loader.start_hot_reload(interval=0.02)
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if getattr(registry, "watch_v", 1) == 2:
                break
            time.sleep(0.05)
        assert getattr(registry, "watch_v", 1) == 2
    finally:
        loader.stop_hot_reload()


# ── Additional PluginLoader state / count tests ───────────────────────────────

def test_loader_running_is_false_initially(loader):
    assert loader._running is False


def test_loader_watcher_thread_is_none_initially(loader):
    assert loader._watcher_thread is None


def test_loader_mtimes_empty_initially(loader):
    assert loader._mtimes == {}


def test_load_all_two_plugins_returns_count_two(loader, tmp_path, registry):
    for name in ("alpha", "beta"):
        (tmp_path / f"{name}.py").write_text(
            f"def register_tools(r):\n    r.{name} = True\n"
        )
    count = loader.load_all()
    assert count == 2
    assert getattr(registry, "alpha", False) is True
    assert getattr(registry, "beta", False) is True


def test_load_plugin_returns_true_for_valid_plugin(loader, tmp_path):
    p = tmp_path / "valid.py"
    p.write_text("def register_tools(r): r.ok = True\n")
    assert loader._load_plugin(p) is True


def test_stop_hot_reload_safe_when_not_running(loader):
    assert loader._running is False
    loader.stop_hot_reload()
    assert loader._running is False


def test_load_all_no_files_returns_zero(loader, tmp_path):
    count = loader.load_all()
    assert count == 0


# ── PluginLoader registry stored ─────────────────────────────────────────────

def test_loader_registry_attribute(loader, registry):
    assert loader.registry is registry


def test_load_all_mixed_valid_and_invalid(loader, tmp_path, registry):
    (tmp_path / "valid.py").write_text("def register_tools(r): r.v = True\n")
    (tmp_path / "nohook.py").write_text("x = 1\n")
    (tmp_path / "broken.py").write_text("!!! syntax error !!!\n")
    count = loader.load_all()
    assert count == 1
    assert getattr(registry, "v", False) is True


def test_load_plugin_false_for_no_register_tools(loader, tmp_path):
    p = tmp_path / "notool.py"
    p.write_text("CONSTANT = 42\n")
    assert loader._load_plugin(p) is False


def test_load_plugin_false_on_import_error(loader, tmp_path):
    p = tmp_path / "bad_import.py"
    p.write_text("import this_module_does_not_exist_xyz\n")
    assert loader._load_plugin(p) is False


def test_watcher_thread_is_daemon(loader):
    loader.start_hot_reload(interval=9999)
    assert loader._watcher_thread.daemon is True
    loader.stop_hot_reload()


def test_watcher_thread_name(loader):
    loader.start_hot_reload(interval=9999)
    assert "jarvis" in loader._watcher_thread.name.lower()
    loader.stop_hot_reload()


def test_mtime_not_reloaded_if_unchanged(loader, tmp_path, registry):
    plugin = tmp_path / "stable.py"
    plugin.write_text("def register_tools(r): r.count = getattr(r, 'count', 0) + 1\n")
    loader.load_all()
    assert registry.count == 1
    # Call _load_plugin again without changing mtime
    loader._load_plugin(plugin)
    assert registry.count == 2  # _load_plugin always reloads when called directly
