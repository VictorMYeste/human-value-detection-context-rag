"""Cross-attention RAG model (doc tokens attend to KB tokens)."""

from __future__ import annotations

import torch
from torch import nn

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("transformers is required for cross-attention RAG") from exc


class CrossAttentionBlock(nn.Module):
    """A transformer-style block with self-attn + cross-attn + FFN."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        kb: torch.Tensor,
        doc_key_padding_mask: torch.Tensor | None,
        kb_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self-attention over doc tokens.
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=doc_key_padding_mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention: doc queries attend to KB keys/values.
        cross_out, _ = self.cross_attn(
            x, kb, kb, key_padding_mask=kb_key_padding_mask, need_weights=False
        )
        x = self.norm2(x + self.dropout(cross_out))

        # Feed-forward.
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class CrossAttentionRAGModel(nn.Module):
    """Cross-attention RAG model with a doc encoder and KB encoder."""

    def __init__(
        self,
        doc_encoder: nn.Module,
        kb_encoder: nn.Module,
        num_labels: int,
        num_cross_layers: int = 1,
    ) -> None:
        super().__init__()
        self.doc_encoder = doc_encoder
        self.kb_encoder = kb_encoder
        self.num_cross_layers = num_cross_layers

        hidden_size = int(getattr(doc_encoder.config, "hidden_size", 768))
        num_heads = int(getattr(doc_encoder.config, "num_attention_heads", 12))
        self.cross_layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_size, num_heads)
                for _ in range(num_cross_layers)
            ]
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _encode_kb(self, input_ids, attention_mask, batch_size: int):
        # Accept [batch, k, seq] or [batch*k, seq]
        if input_ids.dim() == 3:
            batch_size, top_k, seq_len = input_ids.shape
            flat_ids = input_ids.view(batch_size * top_k, seq_len)
            flat_mask = attention_mask.view(batch_size * top_k, seq_len)
        else:
            flat_ids = input_ids
            flat_mask = attention_mask
            top_k = max(1, flat_ids.size(0) // batch_size)
            seq_len = flat_ids.size(1)

        valid_rows = flat_mask.sum(dim=1) > 0
        if valid_rows.any():
            outputs = self.kb_encoder(
                input_ids=flat_ids[valid_rows],
                attention_mask=flat_mask[valid_rows],
            )
            valid_hidden = outputs.last_hidden_state
            flat_hidden = valid_hidden.new_zeros(
                (flat_ids.size(0), seq_len, valid_hidden.size(-1))
            )
            flat_hidden[valid_rows] = valid_hidden
        else:
            hidden = int(getattr(self.kb_encoder.config, "hidden_size", 768))
            flat_hidden = torch.zeros(
                (flat_ids.size(0), seq_len, hidden),
                device=flat_ids.device,
                dtype=torch.float32,
            )

        kb_hidden = flat_hidden.view(batch_size, top_k, seq_len, -1)
        kb_hidden = kb_hidden.reshape(batch_size, top_k * seq_len, -1)
        kb_mask = flat_mask.view(batch_size, top_k * seq_len)
        return kb_hidden, kb_mask

    def forward(
        self,
        doc_input_ids,
        doc_attention_mask,
        kb_input_ids,
        kb_attention_mask,
    ) -> torch.Tensor:
        """
        Args:
          doc_*: [batch, seq_len]
          kb_*: [batch, k, seq_len] or [batch*k, seq_len]
        Returns:
          logits: [batch, num_labels]
        """
        doc_outputs = self.doc_encoder(
            input_ids=doc_input_ids, attention_mask=doc_attention_mask
        )
        h_doc = doc_outputs.last_hidden_state

        kb_hidden, kb_mask = self._encode_kb(
            kb_input_ids, kb_attention_mask, h_doc.size(0)
        )

        doc_key_padding_mask = None
        if doc_attention_mask is not None:
            doc_key_padding_mask = doc_attention_mask.eq(0)
        kb_key_padding_mask = None
        if kb_mask is not None:
            kb_key_padding_mask = kb_mask.eq(0)
            # MultiheadAttention returns NaNs when all keys are masked for a sample.
            all_masked = kb_key_padding_mask.all(dim=1)
            if all_masked.any():
                kb_key_padding_mask = kb_key_padding_mask.clone()
                kb_key_padding_mask[all_masked, 0] = False

        for layer in self.cross_layers:
            h_doc = layer(h_doc, kb_hidden, doc_key_padding_mask, kb_key_padding_mask)

        pooled = h_doc[:, 0]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_cross_attention_model(
    base_model_name: str,
    num_labels: int,
    num_cross_layers: int = 1,
) -> tuple[nn.Module, AutoTokenizer, AutoTokenizer]:
    """Build the cross-attention RAG model and tokenizers."""
    doc_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    kb_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    doc_encoder = AutoModel.from_pretrained(base_model_name)
    kb_encoder = AutoModel.from_pretrained(base_model_name)
    model = CrossAttentionRAGModel(
        doc_encoder=doc_encoder,
        kb_encoder=kb_encoder,
        num_labels=num_labels,
        num_cross_layers=num_cross_layers,
    )
    return model, doc_tokenizer, kb_tokenizer
