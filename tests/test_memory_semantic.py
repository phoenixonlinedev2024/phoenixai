"""Tests for jarvis.memory.semantic — ChromaDB-backed vector store wrapper."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


def _fake_collection(docs=None, metas=None, distances=None):
    col = MagicMock()
    col.count = MagicMock(return_value=len(docs or []))
    col.upsert = MagicMock()
    col.query = MagicMock(return_value={
        "documents": [docs or []],
        "metadatas": [metas or []],
        "distances": [distances or []],
    })
    return col


@pytest.fixture()
def mem():
    """Fresh SemanticMemory with a stubbed collection (never touches ChromaDB)."""
    # Ensure the import succeeds regardless of whether chromadb is installed.
    sys.modules.setdefault("chromadb", MagicMock())
    sys.modules.setdefault("chromadb.config", MagicMock())
    from jarvis.memory.semantic import SemanticMemory
    sm = SemanticMemory()
    sm._collection = _fake_collection()
    return sm


# ── uid hash ─────────────────────────────────────────────────────────────────

def test_uid_is_deterministic(mem):
    assert mem._uid("hello") == mem._uid("hello")
    assert len(mem._uid("anything")) == 16


def test_uid_differs_for_different_input(mem):
    assert mem._uid("a") != mem._uid("b")


# ── store ────────────────────────────────────────────────────────────────────

def test_store_upserts_text_and_metadata(mem):
    mem.store("remember this", metadata={"type": "note"})
    mem._collection.upsert.assert_called_once()
    kwargs = mem._collection.upsert.call_args.kwargs
    assert kwargs["documents"] == ["remember this"]
    assert kwargs["metadatas"] == [{"type": "note"}]


def test_store_defaults_metadata_to_empty_dict(mem):
    mem.store("no meta")
    kwargs = mem._collection.upsert.call_args.kwargs
    assert kwargs["metadatas"] == [{}]


def test_store_swallows_exceptions(mem, capsys):
    mem._collection.upsert = MagicMock(side_effect=RuntimeError("chroma down"))
    mem.store("will fail")
    assert "Store error" in capsys.readouterr().out


# ── recall ───────────────────────────────────────────────────────────────────

def test_recall_returns_ranked_list(mem):
    mem._collection = _fake_collection(
        docs=["doc1", "doc2"],
        metas=[{"type": "fact"}, {"type": "lesson"}],
        distances=[0.1, 0.3],
    )
    out = mem.recall("query", n=2)
    assert len(out) == 2
    assert out[0]["text"] == "doc1"
    assert out[0]["similarity"] == round(1 - 0.1, 3)
    assert out[1]["meta"]["type"] == "lesson"


def test_recall_passes_where_filter(mem):
    mem._collection = _fake_collection(docs=["x"], metas=[{"type": "fact"}], distances=[0.2])
    mem.recall("q", n=1, where={"type": "fact"})
    kwargs = mem._collection.query.call_args.kwargs
    assert kwargs["where"] == {"type": "fact"}


def test_recall_returns_empty_on_exception(mem, capsys):
    mem._collection.query = MagicMock(side_effect=RuntimeError("fail"))
    out = mem.recall("q")
    assert out == []
    assert "Recall error" in capsys.readouterr().out


# ── High-level helpers ───────────────────────────────────────────────────────

def test_store_fact_adds_type_and_key(mem):
    mem.store_fact("lang", "Python")
    kwargs = mem._collection.upsert.call_args.kwargs
    assert "lang: Python" in kwargs["documents"][0]
    meta = kwargs["metadatas"][0]
    assert meta["type"] == "fact"
    assert meta["key"] == "lang"


def test_store_lesson_tags_type(mem):
    mem.store_lesson("test more")
    meta = mem._collection.upsert.call_args.kwargs["metadatas"][0]
    assert meta["type"] == "lesson"


def test_store_conversation_snippet_includes_session(mem):
    mem.store_conversation_snippet("we talked about X", session_id="abc-123")
    meta = mem._collection.upsert.call_args.kwargs["metadatas"][0]
    assert meta["type"] == "conversation"
    assert meta["session_id"] == "abc-123"


# ── recall_relevant ──────────────────────────────────────────────────────────

def test_recall_relevant_empty_returns_empty_string(mem):
    mem._collection = _fake_collection()
    assert mem.recall_relevant("q") == ""


def test_recall_relevant_formats_results(mem):
    mem._collection = _fake_collection(
        docs=["doc-A", "doc-B"],
        metas=[{"type": "fact"}, {"type": "lesson"}],
        distances=[0.05, 0.20],
    )
    out = mem.recall_relevant("q", n=2)
    assert "Semantically Relevant Memories" in out
    assert "doc-A" in out
    assert "[fact]" in out
    assert "[lesson]" in out


def test_recall_relevant_meta_missing_type_falls_back(mem):
    mem._collection = _fake_collection(
        docs=["mystery"], metas=[{}], distances=[0.1],
    )
    out = mem.recall_relevant("q")
    assert "[memory]" in out


# ── count ────────────────────────────────────────────────────────────────────

def test_count_returns_zero_on_error(mem):
    mem._collection = MagicMock()
    mem._collection.count = MagicMock(side_effect=RuntimeError("nope"))
    assert mem.count() == 0


def test_count_returns_value(mem):
    mem._collection = _fake_collection(docs=["a", "b", "c"])
    assert mem.count() == 3


# ── Lazy init path ───────────────────────────────────────────────────────────

def test_get_collection_raises_when_chromadb_missing(monkeypatch):
    # Force the import to fail
    import builtins
    real_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        if name == "chromadb":
            raise ImportError("no chromadb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)

    from jarvis.memory.semantic import SemanticMemory
    sm = SemanticMemory()
    with pytest.raises(RuntimeError, match="chromadb not installed"):
        sm._get_collection()


# ── store_fact / store_lesson / store_conversation_snippet ────────────────────

def test_store_fact_passes_fact_metadata(mem):
    mem.store_fact("language", "Python")
    kwargs = mem._collection.upsert.call_args.kwargs
    assert kwargs["metadatas"][0]["type"] == "fact"
    assert kwargs["metadatas"][0]["key"] == "language"
    assert "language: Python" in kwargs["documents"][0]


def test_store_lesson_passes_lesson_metadata(mem):
    mem.store_lesson("always validate input")
    kwargs = mem._collection.upsert.call_args.kwargs
    assert kwargs["metadatas"][0]["type"] == "lesson"
    assert kwargs["documents"][0] == "always validate input"


def test_store_conversation_snippet_passes_session_id(mem):
    mem.store_conversation_snippet("user asked about Python", session_id="sess-42")
    kwargs = mem._collection.upsert.call_args.kwargs
    assert kwargs["metadatas"][0]["type"] == "conversation"
    assert kwargs["metadatas"][0]["session_id"] == "sess-42"


# ── recall similarity calculation ─────────────────────────────────────────────

def test_recall_similarity_from_distance(mem):
    """similarity = 1 - distance (cosine similarity)."""
    mem._collection = _fake_collection(
        docs=["doc"],
        metas=[{"type": "fact"}],
        distances=[0.3],
    )
    results = mem.recall("query", n=1)
    assert len(results) == 1
    assert results[0]["similarity"] == pytest.approx(0.7)


def test_recall_returns_empty_on_exception(mem):
    mem._collection.query = MagicMock(side_effect=RuntimeError("query failed"))
    results = mem.recall("anything")
    assert results == []


def test_recall_with_where_filter(mem):
    """Where filter is passed to the collection query."""
    mem._collection = _fake_collection(docs=["matched"], metas=[{"type": "fact"}], distances=[0.1])
    mem.recall("query", n=1, where={"type": "fact"})
    call_kwargs = mem._collection.query.call_args.kwargs
    assert call_kwargs.get("where") == {"type": "fact"}


# ── store error handling ──────────────────────────────────────────────────────

def test_store_error_does_not_raise(mem, capsys):
    mem._collection.upsert = MagicMock(side_effect=RuntimeError("upsert failed"))
    mem.store("data")  # should not raise
    out = capsys.readouterr().out
    assert "Store error" in out


# ── recall_relevant formatting ────────────────────────────────────────────────

def test_recall_relevant_similarity_shown_in_output(mem):
    mem._collection = _fake_collection(
        docs=["relevant fact"],
        metas=[{"type": "fact"}],
        distances=[0.15],
    )
    out = mem.recall_relevant("query")
    assert "0.85" in out  # 1 - 0.15


def test_recall_relevant_multiple_types(mem):
    mem._collection = _fake_collection(
        docs=["fact text", "lesson text", "conversation text"],
        metas=[{"type": "fact"}, {"type": "lesson"}, {"type": "conversation"}],
        distances=[0.1, 0.2, 0.3],
    )
    out = mem.recall_relevant("query", n=3)
    assert "[fact]" in out
    assert "[lesson]" in out
    assert "[conversation]" in out


# ── Additional SemanticMemory coverage ───────────────────────────────────────

def test_uid_is_hex_string(mem):
    uid = mem._uid("hello world")
    assert all(c in "0123456789abcdef" for c in uid)
    assert len(uid) == 16


def test_store_passes_document_text(mem):
    mem.store("unique text JARVIS stored here")
    docs = mem._collection.upsert.call_args.kwargs["documents"]
    assert "unique text JARVIS stored here" in docs[0]


def test_recall_clips_n_to_collection_count(mem):
    mem._collection.count = MagicMock(return_value=2)
    mem._collection.query = MagicMock(return_value={
        "documents": [["a", "b"]],
        "metadatas": [[{}, {}]],
        "distances": [[0.1, 0.2]],
    })
    mem.recall("q", n=100)
    n_results_used = mem._collection.query.call_args.kwargs["n_results"]
    assert n_results_used == 2


def test_count_on_fresh_memory_returns_zero():
    sys.modules.setdefault("chromadb", MagicMock())
    sys.modules.setdefault("chromadb.config", MagicMock())
    from jarvis.memory.semantic import SemanticMemory
    sm = SemanticMemory()
    sm._collection = None
    col = MagicMock()
    col.count = MagicMock(side_effect=RuntimeError("no collection"))

    class FakeChroma:
        def PersistentClient(self, *a, **kw):
            raise RuntimeError("no chroma")

    import jarvis.memory.semantic as sem_mod
    original = sys.modules.get("chromadb")
    sys.modules["chromadb"] = FakeChroma()
    result = sm.count()
    sys.modules["chromadb"] = original
    assert result == 0


def test_recall_relevant_meta_missing_type_uses_memory_fallback(mem):
    mem._collection = _fake_collection(
        docs=["something with no type"],
        metas=[{}],
        distances=[0.2],
    )
    out = mem.recall_relevant("test query")
    assert "[memory]" in out
