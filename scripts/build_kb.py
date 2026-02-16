"""Build FAISS index for a manually curated KB."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from value_context_rag.kb.retriever import load_chunks
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KB FAISS index.")
    parser.add_argument(
        "--kb_output_dir",
        default="data/kb",
        help="Output directory for KB chunks and index.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing KB outputs if present.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _build_faiss_index(embeddings: np.ndarray):
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("faiss is required to build the index") from exc

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    silence_transformers_logging()

    output_dir = Path(args.kb_output_dir)
    chunks_path = output_dir / "kb_chunks.jsonl"
    index_path = output_dir / "kb_index.faiss"
    embeddings_path = output_dir / "kb_embeddings.npy"

    if not args.overwrite:
        if index_path.exists() or embeddings_path.exists():
            raise FileExistsError(
                f"KB outputs already exist under {output_dir}. Use --overwrite to replace."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"KB chunks file not found at {chunks_path}. Create it manually first."
        )

    chunks = load_chunks(chunks_path)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "sentence-transformers is required to build embeddings"
        ) from exc

    LOGGER.info("Embedding %d chunks", len(chunks))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    np.save(embeddings_path, embeddings)
    index = _build_faiss_index(embeddings)

    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("faiss is required to save the index") from exc

    faiss.write_index(index, str(index_path))
    LOGGER.info("Saved FAISS index to %s", index_path)

    source_counts = Counter(chunk.get("source", "unknown") for chunk in chunks)
    value_counts = Counter()
    for chunk in chunks:
        values = chunk.get("values", [])
        for val in values:
            value_counts[val] += 1
    LOGGER.info("Chunk count: %d", len(chunks))
    LOGGER.info("Chunks per source: %s", dict(source_counts))
    if value_counts:
        LOGGER.info("Chunks per value: %s", dict(value_counts))


if __name__ == "__main__":
    main()
