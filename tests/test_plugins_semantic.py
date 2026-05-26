"""Tests for jarvis.plugins.loader and jarvis.memory.semantic."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock
    # chromadb Settings fake
    sys.modules["chromadb"].config = MagicMock()
    sys.modules["chromadb.config"] = MagicMock()


_inject_fakes()

from jarvis.plugins.loader import PluginLoader  # noqa: E402
from jarvis.memory.semantic import SemanticMemory  # noqa: E402


# ── PluginLoader ──────────────────────────────────────────────────────────────

@pytest.fixture()
def plugin_dir(tmp_path):
    return tmp_path / "plugins"


def _make_plugin(path: Path, name: str = "test_plugin") -> Path:
    plugin = path / f"{name}.py"
    plugin.write_text("""
def register_tools(registry):
    registry.registered = True
""", encoding="utf-8")
    return plugin


def test_load_all_returns_count(plugin_dir, tmp_path):
    """PluginLoader loads plugins in a temp dir by patching PLUGINS_DIR."""
    plugin_dir.mkdir()
    _make_plugin(plugin_dir)

    # Patch PLUGINS_DIR so loader scans our temp dir
    import jarvis.plugins.loader as loader_mod
    original = loader_mod.PLUGINS_DIR
    loader_mod.PLUGINS_DIR = plugin_dir

    registry = MagicMock()
    loader = PluginLoader(registry)
    count = loader.load_all()
    loader_mod.PLUGINS_DIR = original

    assert count == 1
    assert registry.registered is True


def test_load_all_skips_dunder_files(plugin_dir):
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("")
    (plugin_dir / "_private.py").write_text("")

    import jarvis.plugins.loader as loader_mod
    original = loader_mod.PLUGINS_DIR
    loader_mod.PLUGINS_DIR = plugin_dir

    registry = MagicMock()
    loader = PluginLoader(registry)
    count = loader.load_all()
    loader_mod.PLUGINS_DIR = original

    assert count == 0


def test_load_plugin_without_register_tools_returns_false(plugin_dir):
    plugin_dir.mkdir()
    p = plugin_dir / "no_register.py"
    p.write_text("x = 1\n")

    import jarvis.plugins.loader as loader_mod
    original = loader_mod.PLUGINS_DIR
    loader_mod.PLUGINS_DIR = plugin_dir

    registry = MagicMock()
    loader = PluginLoader(registry)
    result = loader._load_plugin(p)
    loader_mod.PLUGINS_DIR = original

    assert result is False


def test_load_plugin_handles_syntax_error(plugin_dir):
    plugin_dir.mkdir()
    p = plugin_dir / "bad.py"
    p.write_text("def register_tools(r):\n  raise RuntimeError('fail')\n")

    import jarvis.plugins.loader as loader_mod
    original = loader_mod.PLUGINS_DIR
    loader_mod.PLUGINS_DIR = plugin_dir

    registry = MagicMock()
    loader = PluginLoader(registry)
    result = loader._load_plugin(p)
    loader_mod.PLUGINS_DIR = original

    assert result is False


def test_hot_reload_starts_thread():
    registry = MagicMock()
    loader = PluginLoader(registry)
    loader.start_hot_reload(interval=100.0)
    time.sleep(0.05)
    assert loader._running is True
    assert loader._watcher_thread is not None
    assert loader._watcher_thread.is_alive()
    loader.stop_hot_reload()


def test_stop_hot_reload_sets_flag():
    registry = MagicMock()
    loader = PluginLoader(registry)
    loader.start_hot_reload(interval=100.0)
    loader.stop_hot_reload()
    assert loader._running is False


# ── SemanticMemory ────────────────────────────────────────────────────────────

def _make_chroma_mock():
    """Build a chromadb mock whose collection mimics the real API."""
    collection = MagicMock()
    collection.count = MagicMock(return_value=2)
    collection.query = MagicMock(return_value={
        "documents": [["fact one", "lesson two"]],
        "metadatas": [[{"type": "fact"}, {"type": "lesson"}]],
        "distances": [[0.1, 0.2]],
    })
    client = MagicMock()
    client.get_or_create_collection = MagicMock(return_value=collection)
    return client, collection


def test_semantic_uid_is_deterministic():
    sm = SemanticMemory()
    assert sm._uid("hello") == sm._uid("hello")


def test_semantic_uid_is_different_for_different_text():
    sm = SemanticMemory()
    assert sm._uid("a") != sm._uid("b")


def test_semantic_uid_length():
    sm = SemanticMemory()
    assert len(sm._uid("test")) == 16


def test_store_calls_upsert():
    sm = SemanticMemory()
    client, col = _make_chroma_mock()
    sm._client = client
    sm._collection = col
    sm.store("some text", metadata={"type": "fact"})
    col.upsert.assert_called_once()
    args = col.upsert.call_args
    assert "some text" in args.kwargs["documents"]


def test_recall_returns_structured_results():
    sm = SemanticMemory()
    _, col = _make_chroma_mock()
    sm._collection = col
    results = sm.recall("query", n=2)
    assert len(results) == 2
    assert results[0]["text"] == "fact one"
    assert results[0]["similarity"] == 0.9
    assert results[1]["similarity"] == 0.8


def test_recall_relevant_returns_formatted_string():
    sm = SemanticMemory()
    _, col = _make_chroma_mock()
    sm._collection = col
    out = sm.recall_relevant("test query")
    assert "Semantically Relevant Memories" in out
    assert "fact one" in out


def test_recall_relevant_returns_empty_when_no_results():
    sm = SemanticMemory()
    col = MagicMock()
    col.count = MagicMock(return_value=0)
    col.query = MagicMock(return_value={"documents": [[]], "metadatas": [[]], "distances": [[]]})
    sm._collection = col
    out = sm.recall_relevant("query")
    assert out == ""


def test_store_conversation_snippet_sets_metadata():
    sm = SemanticMemory()
    col = MagicMock()
    sm._collection = col
    sm.store_conversation_snippet("user said hello", session_id="s1")
    col.upsert.assert_called_once()
    meta = col.upsert.call_args.kwargs["metadatas"][0]
    assert meta["type"] == "conversation"
    assert meta["session_id"] == "s1"


def test_store_fact_sets_metadata():
    sm = SemanticMemory()
    col = MagicMock()
    sm._collection = col
    sm.store_fact("language", "python")
    meta = col.upsert.call_args.kwargs["metadatas"][0]
    assert meta["type"] == "fact"
    assert meta["key"] == "language"


def test_count_returns_int():
    sm = SemanticMemory()
    _, col = _make_chroma_mock()
    sm._collection = col
    assert sm.count() == 2


def test_count_returns_zero_on_error():
    sm = SemanticMemory()
    col = MagicMock()
    col.count = MagicMock(side_effect=RuntimeError("db error"))
    sm._collection = col
    assert sm.count() == 0


def test_store_silently_handles_error():
    sm = SemanticMemory()
    col = MagicMock()
    col.upsert = MagicMock(side_effect=RuntimeError("write error"))
    sm._collection = col
    sm.store("text")  # should not raise


def test_recall_silently_handles_error():
    sm = SemanticMemory()
    col = MagicMock()
    col.query = MagicMock(side_effect=RuntimeError("read error"))
    col.count = MagicMock(return_value=1)
    sm._collection = col
    results = sm.recall("query")
    assert results == []


# ── SemanticMemory additional coverage ───────────────────────────────────────

def test_semantic_memory_initial_state():
    sm = SemanticMemory()
    assert sm._client is None
    assert sm._collection is None


def test_store_lesson_sets_type_metadata():
    sm = SemanticMemory()
    col = MagicMock()
    sm._collection = col
    sm.store_lesson("always validate inputs")
    meta = col.upsert.call_args.kwargs["metadatas"][0]
    assert meta["type"] == "lesson"


def test_store_fact_text_format():
    sm = SemanticMemory()
    col = MagicMock()
    sm._collection = col
    sm.store_fact("language", "Python")
    doc = col.upsert.call_args.kwargs["documents"][0]
    assert "language" in doc
    assert "Python" in doc


def test_recall_with_where_filter_passes_kwarg():
    sm = SemanticMemory()
    col = MagicMock()
    col.count = MagicMock(return_value=3)
    col.query = MagicMock(return_value={
        "documents": [["result"]],
        "metadatas": [[{"type": "fact"}]],
        "distances": [[0.1]],
    })
    sm._collection = col
    results = sm.recall("query", where={"type": "fact"})
    call_kwargs = col.query.call_args.kwargs
    assert "where" in call_kwargs
    assert call_kwargs["where"] == {"type": "fact"}
    assert len(results) == 1


def test_recall_relevant_includes_similarity():
    sm = SemanticMemory()
    col = MagicMock()
    col.count = MagicMock(return_value=1)
    col.query = MagicMock(return_value={
        "documents": [["Python tip"]],
        "metadatas": [[{"type": "lesson"}]],
        "distances": [[0.05]],
    })
    sm._collection = col
    out = sm.recall_relevant("python")
    assert "similarity" in out
    assert "Python tip" in out


def test_store_conversation_snippet_text_stored():
    sm = SemanticMemory()
    col = MagicMock()
    sm._collection = col
    sm.store_conversation_snippet("the user asked about JARVIS", session_id="s42")
    doc = col.upsert.call_args.kwargs["documents"][0]
    assert "JARVIS" in doc


# ── PluginLoader additional coverage ─────────────────────────────────────────

def test_hot_reload_start_sets_running_flag(tmp_path):
    import jarvis.plugins.loader as loader_mod
    original = loader_mod.PLUGINS_DIR
    loader_mod.PLUGINS_DIR = tmp_path

    registry = MagicMock()
    loader = PluginLoader(registry)
    loader.start_hot_reload(interval=9999)
    assert loader._running is True
    loader.stop_hot_reload()
    loader_mod.PLUGINS_DIR = original


def test_hot_reload_start_prints_message(tmp_path, capsys):
    import jarvis.plugins.loader as loader_mod
    original = loader_mod.PLUGINS_DIR
    loader_mod.PLUGINS_DIR = tmp_path

    registry = MagicMock()
    loader = PluginLoader(registry)
    loader.start_hot_reload(interval=9999)
    loader.stop_hot_reload()
    loader_mod.PLUGINS_DIR = original

    out = capsys.readouterr().out
    assert "Hot reload" in out or "polling" in out


def test_load_plugin_success_prints_loaded(tmp_path, capsys):
    plugin = tmp_path / "myplugin.py"
    plugin.write_text("def register_tools(registry):\n    pass\n")
    registry = MagicMock()
    loader = PluginLoader(registry)
    result = loader._load_plugin(plugin)
    assert result is True
    out = capsys.readouterr().out
    assert "myplugin" in out


# ── SemanticMemory._uid returns 16-char hex ───────────────────────────────────

def test_uid_is_16_char_hex():
    sm = SemanticMemory()
    uid = sm._uid("some text content")
    assert len(uid) == 16
    assert all(c in "0123456789abcdef" for c in uid)


def test_uid_is_deterministic():
    sm = SemanticMemory()
    assert sm._uid("hello") == sm._uid("hello")


def test_uid_different_text_different_uid():
    sm = SemanticMemory()
    assert sm._uid("text_a") != sm._uid("text_b")


# ── recall_relevant: empty memory returns empty string ───────────────────────

def test_recall_relevant_no_results_returns_empty_string():
    sm = SemanticMemory()
    col = MagicMock()
    col.count = MagicMock(return_value=0)
    col.query = MagicMock(return_value={"documents": [[]], "metadatas": [[]], "distances": [[]]})
    sm._collection = col
    assert sm.recall_relevant("nothing here") == ""


# ── recall_relevant: header included in output ───────────────────────────────

def test_recall_relevant_includes_header():
    sm = SemanticMemory()
    col = MagicMock()
    col.count = MagicMock(return_value=1)
    col.query = MagicMock(return_value={
        "documents": [["fact text"]],
        "metadatas": [[{"type": "fact"}]],
        "distances": [[0.1]],
    })
    sm._collection = col
    out = sm.recall_relevant("query")
    assert "Semantically Relevant" in out


# ── store_lesson text passed directly ────────────────────────────────────────

def test_store_lesson_document_text_is_lesson_itself():
    sm = SemanticMemory()
    col = MagicMock()
    sm._collection = col
    sm.store_lesson("test lesson text")
    doc = col.upsert.call_args.kwargs["documents"][0]
    assert doc == "test lesson text"


# ── store_fact document is key: value format ─────────────────────────────────

def test_store_fact_document_colon_separated():
    sm = SemanticMemory()
    col = MagicMock()
    sm._collection = col
    sm.store_fact("mykey", "myvalue")
    doc = col.upsert.call_args.kwargs["documents"][0]
    assert "mykey: myvalue" == doc


# ── PluginLoader: load_all with no .py files returns 0 ───────────────────────

def test_load_all_empty_dir_returns_0(tmp_path):
    import jarvis.plugins.loader as loader_mod
    original = loader_mod.PLUGINS_DIR
    loader_mod.PLUGINS_DIR = tmp_path
    registry = MagicMock()
    loader = PluginLoader(registry)
    count = loader.load_all()
    loader_mod.PLUGINS_DIR = original
    assert count == 0
