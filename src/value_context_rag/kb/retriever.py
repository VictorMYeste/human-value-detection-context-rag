"""Retrieval API for RAG."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _normalize_values(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if str(v).strip()]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts
    return [str(raw).strip()]


def validate_chunks(chunks: list[dict]) -> None:
    """Basic validation for manually curated chunks."""
    required = {"id", "source", "text"}
    for idx, chunk in enumerate(chunks):
        missing = required - set(chunk.keys())
        if missing:
            raise ValueError(f"Chunk {idx} missing keys: {sorted(missing)}")
        if "values" in chunk and chunk["values"] is not None:
            _ = _normalize_values(chunk["values"])


def load_chunks(kb_path: Path) -> list[dict]:
    if not kb_path.exists():
        raise FileNotFoundError(f"KB chunks not found: {kb_path}")
    chunks: list[dict] = []
    with kb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    validate_chunks(chunks)
    for chunk in chunks:
        if "values" in chunk or "value" in chunk:
            raw = chunk.get("values", chunk.get("value"))
            chunk["values"] = _normalize_values(raw)
            chunk.pop("value", None)
    LOGGER.info("Loaded %d KB chunks from %s", len(chunks), kb_path)
    return chunks


def _load_faiss_index(index_path: Path):
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("faiss is required to load the index") from exc

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(index_path))


def _default_embedder():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("sentence-transformers is required to embed queries") from exc

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class Retriever:
    chunks: list[dict]
    index: object
    embed_query: Callable[[str], np.ndarray]

    def retrieve(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        value: str | None = None,
        source: str | None = None,
    ) -> list[dict]:
        if not query_text:
            return []

        query_vec = self.embed_query(query_text)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        distances, indices = self.index.search(query_vec.astype("float32"), top_k)
        hits: list[dict] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[int(idx)]
            if value:
                values = chunk.get("values", [])
                if value not in values:
                    continue
            if source and chunk.get("source") != source:
                continue
            hits.append(chunk)

        LOGGER.info("Retrieved %d chunks for query", len(hits))
        return hits


def init_retriever(
    kb_path: str = "data/kb/kb_chunks.jsonl",
    index_path: str = "data/kb/kb_index.faiss",
    *,
    debug: bool = False,
) -> Retriever:
    """Load chunks and FAISS index, return a Retriever."""
    kb_path_obj = Path(kb_path)
    index_path_obj = Path(index_path)

    chunks = load_chunks(kb_path_obj)
    index = _load_faiss_index(index_path_obj)
    model = _default_embedder()

    def embed_query(text: str) -> np.ndarray:
        if debug:
            LOGGER.debug("Embedding query of length %d", len(text))
        vec = model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")

    return Retriever(chunks=chunks, index=index, embed_query=embed_query)
