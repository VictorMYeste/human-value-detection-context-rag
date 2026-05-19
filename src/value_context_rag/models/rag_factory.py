"""Factory for building RAG models by architecture."""

from __future__ import annotations

from torch import nn

from value_context_rag.models.rag_cross_attention import build_cross_attention_model
from value_context_rag.models.rag_late_fusion import build_late_fusion_model


def build_rag_model(
    config: dict, num_labels: int
) -> tuple[nn.Module, dict[str, object]]:
    """Build a late-fusion or cross-attention RAG model based on the config.

    Returns:
      (model, tokenizers) where tokenizers is a dict like {"doc": tok, "kb": tok}.
    """
    model_cfg = config.get("model", {})
    rag_cfg = config.get("rag", {})

    base_model_name = model_cfg.get("name", "microsoft/deberta-v3-base")
    kb_model_name = rag_cfg.get("kb_encoder_name", base_model_name)
    mode = rag_cfg.get("mode", "none")
    enabled = bool(rag_cfg.get("enabled", False))
    if not enabled or mode not in {"late", "cross_attention"}:
        raise ValueError(
            "build_rag_model only supports rag.mode in {'late', 'cross_attention'}"
        )

    if mode == "late":
        model, doc_tokenizer, kb_tokenizer = build_late_fusion_model(
            base_model_name=base_model_name,
            kb_model_name=kb_model_name,
            num_labels=num_labels,
        )
        return model, {"doc": doc_tokenizer, "kb": kb_tokenizer}

    if mode == "cross_attention":
        num_cross_layers = int(rag_cfg.get("num_cross_layers", 1))
        model, doc_tokenizer, kb_tokenizer = build_cross_attention_model(
            base_model_name=base_model_name,
            num_labels=num_labels,
            num_cross_layers=num_cross_layers,
        )
        return model, {"doc": doc_tokenizer, "kb": kb_tokenizer}

    raise ValueError(f"Unsupported rag.mode: {mode}")
