"""Plugin loader with hot reload — discovers and loads tools from plugins/ directory."""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry

PLUGINS_DIR = Path(__file__).parent
_LOADER_FILE = Path(__file__).name


class PluginLoader:
    """Loads plugins at startup and hot-reloads when files change."""

    def __init__(self, registry: "ToolRegistry") -> None:
        self.registry = registry
        self._mtimes: dict[str, float] = {}
        self._watcher_thread: threading.Thread | None = None
        self._running = False

    def _is_plugin_file(self, path: Path) -> bool:
        """A plugin file is a non-underscore .py file other than the loader itself."""
        if path.name.startswith("_"):
            return False
        if path.name == _LOADER_FILE:
            return False
        return True

    def load_all(self) -> int:
        """Load all plugins from the plugins directory. Returns count loaded."""
        count = 0
        for path in PLUGINS_DIR.glob("*.py"):
            if not self._is_plugin_file(path):
                continue
            if self._load_plugin(path):
                count += 1
        return count

    def _load_plugin(self, path: Path) -> bool:
        # Always remember we've seen this file, so the hot-reload loop doesn't
        # repeatedly re-import files that have no register_tools hook.
        self._mtimes[str(path)] = path.stat().st_mtime
        try:
            spec = importlib.util.spec_from_file_location(f"jarvis.plugins.{path.stem}", path)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            if hasattr(mod, "register_tools"):
                mod.register_tools(self.registry)
                print(f"[JARVIS Plugins] Loaded: {path.name}")
                return True
            return False
        except Exception as exc:
            print(f"[JARVIS Plugins] Failed to load {path.name}: {exc}")
            return False

    def start_hot_reload(self, interval: float = 2.0) -> None:
        """Watch for new or modified plugin files and reload them."""
        self._running = True
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval,),
            daemon=True,
            name="jarvis-plugin-watcher",
        )
        self._watcher_thread.start()
        print(f"[JARVIS Plugins] Hot reload active (polling every {interval}s).")

    def stop_hot_reload(self) -> None:
        self._running = False

    def _watch_loop(self, interval: float) -> None:
        while self._running:
            for path in PLUGINS_DIR.glob("*.py"):
                if not self._is_plugin_file(path):
                    continue
                key = str(path)
                mtime = path.stat().st_mtime
                if key not in self._mtimes:
                    print(f"[JARVIS Plugins] New plugin detected: {path.name}")
                    self._load_plugin(path)
                elif mtime > self._mtimes[key]:
                    print(f"[JARVIS Plugins] Plugin changed, reloading: {path.name}")
                    self._load_plugin(path)
            time.sleep(interval)
