"""
retrieval/vector_store.py

ChromaDB-backed vector store for the tax corpus.

Notes:
  - Token counts use word count as a proxy (intentional — fast and
    good enough for the QUBO budget constraint).
  - ChromaDB metadata values must be scalars (str/int/float/bool);
    list-valued fields are JSON-stringified at ingest time.
"""

import json
import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBED_MODEL


def _estimate_tokens(text: str) -> int:
    """Word count proxy for token count (intentional)."""
    return max(1, len(text.split()))


def _coerce_metadata(meta: dict) -> dict:
    """ChromaDB requires scalar metadata values. JSON-encode lists."""
    out = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (list, tuple, dict)):
            out[k] = json.dumps(v)
        elif v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


class VectorStore:
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection: str = CHROMA_COLLECTION,
        embed_model: str = EMBED_MODEL,
    ):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self.embed_fn,
        )

    def add_documents(self, docs: list[dict]) -> None:
        if not docs:
            return
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metas = [_coerce_metadata(d.get("metadata", {})) for d in docs]
        self.collection.upsert(ids=ids, documents=texts, metadatas=metas)

    def count(self) -> int:
        return self.collection.count()

    def search(self, query: str, k: int = 30) -> list[dict]:
        if self.collection.count() == 0:
            return []
        k = min(k, self.collection.count())
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        out: list[dict] = []
        for i, (doc_id, text, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            # ChromaDB distance → similarity score in [0, 1]
            score = max(0.0, 1.0 - float(dist))
            out.append({
                "id": doc_id,
                "text": text,
                "metadata": dict(meta) if meta else {},
                "score": score,
                "token_count": _estimate_tokens(text),
            })
        return out
