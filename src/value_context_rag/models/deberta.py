"""DeBERTa model utilities."""

from __future__ import annotations

import torch
from torch import nn

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


class DebertaForMultiLabel(nn.Module):
    """DeBERTa-v3-base with a multi-label classification head (legacy wrapper)."""

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
        # Ensure dtype matches classifier weights (e.g., fp16 vs fp32)
        pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)
        return logits


class DebertaV3ForMultiLabelClassification(torch.nn.Module):
    """HF-native DeBERTa v3 model for multi-label classification."""

    def __init__(self, model_name: str, num_labels: int, label_names: list[str]):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("transformers is required for DeBERTa") from exc

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        config.problem_type = "multi_label_classification"
        config.id2label = {i: name for i, name in enumerate(label_names)}
        config.label2id = {name: i for i, name in enumerate(label_names)}
        self.config = config
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, output_dir: str) -> None:
        import os

        os.makedirs(output_dir, exist_ok=True)
        # Save config
        self.config.save_pretrained(output_dir)
        # Save weights
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(
        cls, output_dir: str, model_name: str, label_names: list[str]
    ) -> DebertaV3ForMultiLabelClassification:
        import os

        instance = cls(
            model_name=model_name, num_labels=len(label_names), label_names=label_names
        )
        state_dict = torch.load(
            os.path.join(output_dir, "pytorch_model.bin"), map_location="cpu"
        )
        instance.load_state_dict(state_dict)
        return instance


def build_deberta_model(
    num_labels: int,
    *,
    model_name: str = "microsoft/deberta-v3-base",
    label_names: list[str] | None = None,
) -> tuple[nn.Module, object]:
    """Load a DeBERTa checkpoint and return (model, tokenizer)."""
    try:
        from transformers import (  # type: ignore
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("transformers is required for DeBERTa") from exc

    LOGGER.info("Loading DeBERTa model %s", model_name)
    LOGGER.debug("Initializing tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    LOGGER.debug("Initializing config for %s", model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.problem_type = "multi_label_classification"
    if label_names:
        config.id2label = {i: name for i, name in enumerate(label_names)}
        config.label2id = {name: i for i, name in enumerate(label_names)}
    LOGGER.debug("Initializing model for %s", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
    LOGGER.debug("Created multi-label classifier with %d labels", num_labels)
    return model, tokenizer


def encode_batch(
    tokenizer,
    texts: list[str],
    max_length: int = 1024,
) -> dict[str, torch.Tensor]:
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
