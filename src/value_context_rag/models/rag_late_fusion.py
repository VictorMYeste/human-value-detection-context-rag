"""Late-fusion RAG model (doc encoder + KB encoder + fusion)."""

from __future__ import annotations

import torch
from torch import nn

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("transformers is required for late-fusion RAG") from exc


class LateFusionRAGModel(nn.Module):
    """Late-fusion RAG model that fuses doc and KB representations."""

    def __init__(
        self,
        doc_encoder: nn.Module,
        kb_encoder: nn.Module,
        num_labels: int,
        fusion_method: str = "concat",
    ) -> None:
        super().__init__()
        self.doc_encoder = doc_encoder
        self.kb_encoder = kb_encoder
        self.fusion_method = fusion_method

        doc_hidden = int(getattr(doc_encoder.config, "hidden_size", 768))
        kb_hidden = int(getattr(kb_encoder.config, "hidden_size", 768))

        if fusion_method == "concat":
            fused_hidden = doc_hidden + kb_hidden
            self.fusion_proj = None
        elif fusion_method in {"add", "gated"}:
            fused_hidden = doc_hidden
            self.fusion_proj = (
                nn.Linear(kb_hidden, doc_hidden) if kb_hidden != doc_hidden else None
            )
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        if fusion_method == "gated":
            self.gate = nn.Linear(doc_hidden + doc_hidden, doc_hidden)
        else:
            self.gate = None

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(fused_hidden, num_labels)

    def _encode_doc(self, input_ids, attention_mask):
        outputs = self.doc_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]

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

        valid_rows = flat_mask.sum(dim=1) > 0
        if valid_rows.any():
            outputs = self.kb_encoder(
                input_ids=flat_ids[valid_rows],
                attention_mask=flat_mask[valid_rows],
            )
            valid_kb_cls = outputs.last_hidden_state[:, 0]
            kb_cls = valid_kb_cls.new_zeros((flat_ids.size(0), valid_kb_cls.size(-1)))
            kb_cls[valid_rows] = valid_kb_cls
        else:
            hidden = int(getattr(self.kb_encoder.config, "hidden_size", 768))
            kb_cls = torch.zeros(
                flat_ids.size(0), hidden, device=flat_ids.device, dtype=torch.float32
            )

        kb_cls = kb_cls.view(batch_size, top_k, -1)
        kb_valid = flat_mask.view(batch_size, top_k, seq_len).sum(dim=-1) > 0
        return kb_cls, kb_valid

    def _aggregate_kb(
        self, kb_cls: torch.Tensor, kb_valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        if kb_valid is None:
            return kb_cls.mean(dim=1)
        weights = kb_valid.to(kb_cls.dtype).unsqueeze(-1)
        denom = kb_valid.sum(dim=1, keepdim=True).clamp(min=1).to(kb_cls.dtype)
        return (kb_cls * weights).sum(dim=1) / denom

    def _fuse(self, h_doc: torch.Tensor, h_kb: torch.Tensor) -> torch.Tensor:
        if self.fusion_method == "concat":
            return torch.cat([h_doc, h_kb], dim=-1)
        if self.fusion_proj is not None:
            h_kb = self.fusion_proj(h_kb)
        if self.fusion_method == "add":
            return h_doc + h_kb
        # gated fusion
        gate = torch.sigmoid(self.gate(torch.cat([h_doc, h_kb], dim=-1)))
        return gate * h_doc + (1.0 - gate) * h_kb

    def forward(
        self,
        doc_input_ids,
        doc_attention_mask,
        kb_input_ids,
        kb_attention_mask,
    ) -> torch.Tensor:
        """
        Args:
          doc_input_ids, doc_attention_mask: [batch, seq_len]
          kb_input_ids, kb_attention_mask: [batch, k, seq_len] or [batch*k, seq_len]
        Returns:
          logits: [batch, num_labels]
        """
        h_doc = self._encode_doc(doc_input_ids, doc_attention_mask)
        kb_cls, kb_valid = self._encode_kb(
            kb_input_ids, kb_attention_mask, h_doc.size(0)
        )
        h_kb = self._aggregate_kb(kb_cls, kb_valid)
        fused = self._fuse(h_doc, h_kb)
        fused = self.dropout(fused)
        return self.classifier(fused)


def build_late_fusion_model(
    base_model_name: str,
    kb_model_name: str,
    num_labels: int,
) -> tuple[nn.Module, AutoTokenizer, AutoTokenizer]:
    """Build the late-fusion RAG model and tokenizers."""
    doc_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    kb_tokenizer = AutoTokenizer.from_pretrained(kb_model_name)
    doc_encoder = AutoModel.from_pretrained(base_model_name)
    kb_encoder = AutoModel.from_pretrained(kb_model_name)
    model = LateFusionRAGModel(
        doc_encoder=doc_encoder,
        kb_encoder=kb_encoder,
        num_labels=num_labels,
    )
    return model, doc_tokenizer, kb_tokenizer
