"""Early-fusion RAG model wrapper."""

from __future__ import annotations

from value_context_rag.models.deberta import build_deberta_model


def build_early_fusion_model(base_model_name: str, num_labels: int):
    """Build the early-fusion RAG model.

    Note: early fusion is handled at input construction time; the model is a
    standard encoder with a multi-label head.
    """
    _ = base_model_name
    model, tokenizer = build_deberta_model(num_labels=num_labels)
    return model, tokenizer
