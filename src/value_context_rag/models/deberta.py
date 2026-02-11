"""DeBERTa-v3-base model utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


class DebertaForMultiLabel(nn.Module):
    """DeBERTa-v3-base with a multi-label classification head."""

    def __init__(self, base_model, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def build_deberta_model(num_labels: int) -> Tuple[nn.Module, object]:
    """Load DeBERTa-v3-base and return (model, tokenizer)."""
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("transformers is required for DeBERTa") from exc

    model_name = "microsoft/deberta-v3-base"
    LOGGER.info("Loading DeBERTa model %s", model_name)
    LOGGER.debug("Initializing tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    LOGGER.debug("Initializing base model for %s", model_name)
    base_model = AutoModel.from_pretrained(model_name)
    hidden_size = base_model.config.hidden_size
    LOGGER.debug("Base model hidden size: %d", hidden_size)
    model = DebertaForMultiLabel(base_model, hidden_size, num_labels)
    LOGGER.debug("Created multi-label head with %d labels", num_labels)
    return model, tokenizer


def encode_batch(
    tokenizer,
    texts: List[str],
    max_length: int = 1024,
) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of texts into model inputs."""
    LOGGER.debug("Encoding batch of %d texts (max_length=%d)", len(texts), max_length)
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if "input_ids" in encoded:
        lengths = (encoded["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        mean_len = float(lengths.float().mean().item()) if lengths.numel() else 0.0
        trunc_count = (
            int((lengths >= max_length).sum().item()) if lengths.numel() else 0
        )
        LOGGER.debug(
            "Token lengths: max=%d mean=%.1f truncated=%d/%d",
            max_len,
            mean_len,
            trunc_count,
            lengths.numel(),
        )
    return encoded
