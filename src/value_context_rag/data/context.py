"""Context builders for sentence/window/document inputs."""

from __future__ import annotations

from typing import Literal

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)

MarkerStyle = Literal["deberta", "gemma"]

TGT_START_DEBERTA = "<TGT>"
TGT_END_DEBERTA = "</TGT>"
TGT_START_GEMMA = "<<<TARGET>>>"
TGT_END_GEMMA = "<<<END>>>"


def _ensure_index(doc_sentences: list[str], target_idx: int) -> None:
    if target_idx < 0 or target_idx >= len(doc_sentences):
        raise IndexError(
            f"target_idx out of range: {target_idx} for document of size {len(doc_sentences)}"
        )


def _wrap_target(text: str, marker_style: MarkerStyle) -> str:
    if marker_style == "gemma":
        return f"{TGT_START_GEMMA} {text} {TGT_END_GEMMA}"
    return f"{TGT_START_DEBERTA}{text}{TGT_END_DEBERTA}"


def build_sentence_context(
    doc_sentences: list[str],
    target_idx: int,
    *,
    debug: bool = False,
) -> str:
    """Return only the target sentence as context."""
    _ensure_index(doc_sentences, target_idx)
    if debug:
        LOGGER.debug(
            "Building sentence context (doc_len=%d, target_idx=%d)",
            len(doc_sentences),
            target_idx,
        )
    return str(doc_sentences[target_idx])


def build_window_context(
    doc_sentences: list[str],
    target_idx: int,
    *,
    n_prev: int = 2,
    n_next: int = 2,
    marker_style: MarkerStyle = "deberta",
    debug: bool = False,
) -> str:
    """Return a window of sentences around the target, optionally marked."""
    _ensure_index(doc_sentences, target_idx)

    start = max(0, target_idx - n_prev)
    end = min(len(doc_sentences), target_idx + n_next + 1)

    if debug:
        LOGGER.debug(
            "Building window context (doc_len=%d, target_idx=%d, window=[%d:%d])",
            len(doc_sentences),
            target_idx,
            start,
            end,
        )

    window = [str(s) for s in doc_sentences[start:end]]
    target_pos = target_idx - start
    window[target_pos] = _wrap_target(window[target_pos], marker_style)

    context = " ".join(window).strip()
    if debug:
        LOGGER.debug("Window context length=%d", len(context))

    return context


def build_doc_context(
    doc_sentences: list[str],
    target_idx: int,
    *,
    marker_style: MarkerStyle = "deberta",
    debug: bool = False,
) -> str:
    """Return a full-document context with the target sentence marked."""
    _ensure_index(doc_sentences, target_idx)

    if debug:
        LOGGER.debug(
            "Building doc context (doc_len=%d, target_idx=%d, marker_style=%s)",
            len(doc_sentences),
            target_idx,
            marker_style,
        )

    if marker_style == "gemma":
        lines = []
        for idx, sentence in enumerate(doc_sentences, start=1):
            line = f"{idx}: {sentence}"
            if idx - 1 == target_idx:
                line = _wrap_target(line, marker_style)
            lines.append(line)
        context = "\n".join(lines).strip()
    else:
        sentences = [str(s) for s in doc_sentences]
        sentences[target_idx] = _wrap_target(sentences[target_idx], marker_style)
        context = " ".join(sentences).strip()

    if debug:
        LOGGER.debug("Doc context length=%d", len(context))

    return context
