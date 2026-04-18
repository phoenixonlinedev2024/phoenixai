"""Semantic memory — ChromaDB vector store for meaning-based recall."""

from __future__ import annotations

import hashlib
from typing import Any

from jarvis.config import cfg


class SemanticMemory:
    """ChromaDB-backed vector store for facts, lessons, and conversation snippets."""

    def __init__(self) -> None:
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(cfg.DATA_DIR / "chroma"),
                settings=Settings(anonymized_telemetry=False),
            )
            self._client = client
            self._collection = client.get_or_create_collection(
                name="jarvis_memory",
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb sentence-transformers")
        return self._collection

    def _uid(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def store(self, text: str, metadata: dict | None = None) -> None:
        """Embed and store a piece of text."""
        try:
            col = self._get_collection()
            uid = self._uid(text)
            col.upsert(
                ids=[uid],
                documents=[text],
                metadatas=[metadata or {}],
            )
        except Exception as exc:
            print(f"[SemanticMemory] Store error: {exc}")

    def recall(self, query: str, n: int = 5, where: dict | None = None) -> list[dict]:
        """Return the n most semantically similar stored items."""
        try:
            col = self._get_collection()
            kwargs: dict[str, Any] = {"query_texts": [query], "n_results": min(n, col.count() or 1)}
            if where:
                kwargs["where"] = where
            results = col.query(**kwargs)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            return [
                {"text": d, "meta": m, "similarity": round(1 - dist, 3)}
                for d, m, dist in zip(docs, metas, distances)
            ]
        except Exception as exc:
            print(f"[SemanticMemory] Recall error: {exc}")
            return []

    def store_fact(self, key: str, value: str) -> None:
        self.store(f"{key}: {value}", metadata={"type": "fact", "key": key})

    def store_lesson(self, lesson: str) -> None:
        self.store(lesson, metadata={"type": "lesson"})

    def store_conversation_snippet(self, text: str, session_id: str) -> None:
        self.store(text, metadata={"type": "conversation", "session_id": session_id})

    def recall_relevant(self, query: str, n: int = 8) -> str:
        """Return a formatted context block of the most relevant memories."""
        results = self.recall(query, n=n)
        if not results:
            return ""
        lines = [f"  [{r['meta'].get('type', 'memory')}] {r['text']} (similarity: {r['similarity']})" for r in results]
        return "## Semantically Relevant Memories\n" + "\n".join(lines)

    def count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0
