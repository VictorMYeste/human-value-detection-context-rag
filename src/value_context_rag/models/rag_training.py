"""Training utilities for late-fusion and cross-attention RAG architectures."""

from __future__ import annotations

import inspect
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from value_context_rag.data.context import (
    build_doc_context,
    build_sentence_context,
    build_window_context,
)
from value_context_rag.data.dataset import get_label_names, load_split
from value_context_rag.eval.metrics import compute_f1_metrics
from value_context_rag.kb.retriever import init_retriever
from value_context_rag.models.deberta import encode_batch
from value_context_rag.models.rag_factory import build_rag_model
from value_context_rag.models.training import _build_doc_budget_context
from value_context_rag.utils.logging import get_logger
from value_context_rag.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _resolve_bf16(training_cfg: dict, device: torch.device) -> bool:
    if not bool(training_cfg.get("bf16", False)):
        return False
    if device.type != "cuda":
        LOGGER.warning("bf16 requested but CUDA is unavailable; falling back to fp32")
        return False
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        LOGGER.warning(
            "bf16 requested but CUDA bf16 is unsupported; falling back to fp32"
        )
        return False
    return True


def _build_adamw(
    params,
    *,
    learning_rate: float,
    weight_decay: float,
    training_cfg: dict,
):
    """Build AdamW with conservative defaults to avoid CUDA foreach instability."""
    kwargs: dict[str, object] = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }

    adamw_sig = inspect.signature(torch.optim.AdamW)
    if "foreach" in adamw_sig.parameters:
        kwargs["foreach"] = bool(training_cfg.get("adamw_foreach", False))
    if "fused" in adamw_sig.parameters:
        kwargs["fused"] = bool(training_cfg.get("adamw_fused", False))

    LOGGER.info(
        "AdamW settings: lr=%g wd=%g foreach=%s fused=%s",
        learning_rate,
        weight_decay,
        str(kwargs.get("foreach", "n/a")),
        str(kwargs.get("fused", "n/a")),
    )
    return torch.optim.AdamW(params, **kwargs)


def _autocast_context(use_bf16: bool):
    if not use_bf16:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _tensor_stats(t: torch.Tensor) -> str:
    if t.numel() == 0:
        return "empty"
    finite = torch.isfinite(t)
    finite_ratio = float(finite.float().mean().item())
    clean = t.detach().float().nan_to_num()
    return (
        f"shape={tuple(t.shape)} finite_ratio={finite_ratio:.4f} "
        f"min={clean.min().item():.4f} max={clean.max().item():.4f} "
        f"mean={clean.mean().item():.4f}"
    )


def _log_nan_batch_debug(
    *,
    model,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    logits: torch.Tensor | None,
    reason: str,
    use_bf16: bool,
) -> None:
    """Log diagnostics for the first NaN/Inf batch and optionally retry in fp32."""
    LOGGER.error("NaN/Inf batch debug (reason=%s)", reason)

    doc_mask = batch.get("doc_attention_mask")
    kb_mask = batch.get("kb_attention_mask")
    if doc_mask is not None:
        doc_lens = doc_mask.sum(dim=-1).detach().cpu().tolist()
        LOGGER.error(
            "doc lengths: min=%d max=%d mean=%.2f",
            int(min(doc_lens)) if doc_lens else 0,
            int(max(doc_lens)) if doc_lens else 0,
            float(np.mean(doc_lens)) if doc_lens else 0.0,
        )
    if kb_mask is not None and kb_mask.dim() == 3:
        valid_chunks = (kb_mask.sum(dim=-1) > 0).sum(dim=-1).detach().cpu().tolist()
        LOGGER.error(
            "kb valid chunks per example: min=%d max=%d mean=%.2f",
            int(min(valid_chunks)) if valid_chunks else 0,
            int(max(valid_chunks)) if valid_chunks else 0,
            float(np.mean(valid_chunks)) if valid_chunks else 0.0,
        )

    for key in ("doc_input_ids", "doc_attention_mask", "kb_input_ids", "kb_attention_mask"):
        if key in batch:
            LOGGER.error("%s: %s", key, _tensor_stats(batch[key]))
    LOGGER.error("labels: %s", _tensor_stats(labels))
    if logits is not None:
        LOGGER.error("logits: %s", _tensor_stats(logits))

    if use_bf16:
        try:
            with torch.no_grad():
                outputs_fp32 = model(**batch)
                logits_fp32 = _get_logits(outputs_fp32).float()
            fp32_ok = bool(torch.isfinite(logits_fp32).all().item())
            LOGGER.error(
                "fp32 retry finite=%s | logits_fp32: %s",
                fp32_ok,
                _tensor_stats(logits_fp32),
            )
        except Exception as exc:
            LOGGER.exception("fp32 retry failed: %s", exc)


def _first_non_finite_param(model: nn.Module) -> tuple[str, torch.Tensor] | None:
    """Return first non-finite parameter tensor, if any."""
    for name, param in model.named_parameters():
        if param is None:
            continue
        if not torch.isfinite(param.detach()).all():
            return name, param.detach()
    return None


def _first_non_finite_grad(model: nn.Module) -> tuple[str, torch.Tensor] | None:
    """Return first non-finite gradient tensor, if any."""
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if not torch.isfinite(grad.detach()).all():
            return name, grad.detach()
    return None


def _enable_gradient_checkpointing(model: nn.Module) -> list[str]:
    """Enable gradient checkpointing on wrapper and known encoder submodules."""
    enabled_on: list[str] = []
    seen: set[int] = set()

    candidates: list[tuple[str, object]] = [("model", model)]
    for attr in ("doc_encoder", "kb_encoder", "base_model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            candidates.append((attr, sub))

    for name, module in candidates:
        if not isinstance(module, nn.Module):
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)

        enable_fn = getattr(module, "gradient_checkpointing_enable", None)
        if callable(enable_fn):
            try:
                enable_fn()
                enabled_on.append(name)
            except Exception:
                continue

    return enabled_on


@dataclass
class RagExample:
    doc_text: str
    kb_texts: list[str]
    labels: torch.Tensor


class RagDataset(Dataset):
    def __init__(self, examples: list[RagExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> RagExample:
        return self.examples[idx]


def _build_document_index(df):
    docs: dict[str, list[str]] = {}
    index: dict[tuple[str, str], int] = {}

    grouped = {}
    for row in df.to_dict(orient="records"):
        grouped.setdefault(row["text_id"], []).append(row)

    for text_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda r: int(r["sent_id"]))
        sentences = [str(r["text"]) for r in ordered]
        docs[str(text_id)] = sentences
        for idx, row in enumerate(ordered):
            index[(str(row["text_id"]), str(row["sent_id"]))] = idx

    return docs, index


def _truncate_kb_chunks_to_budget(
    kb_texts: list[str],
    tokenizer,
    kb_budget_tokens: int,
) -> list[str]:
    """Truncate retrieved KB chunks so total KB tokens stay within budget."""
    if kb_budget_tokens <= 0 or not kb_texts:
        return []

    remaining = int(kb_budget_tokens)
    kept: list[str] = []
    for text in kb_texts:
        if remaining <= 0:
            break
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        clipped = token_ids[:remaining]
        if not clipped:
            break
        kept_text = tokenizer.decode(clipped, skip_special_tokens=True).strip()
        if kept_text:
            kept.append(kept_text)
        remaining -= len(clipped)
    return kept


def _log_zero_kb_rate(split_name: str, kb_texts: list[list[str]]) -> None:
    """Log percentage of examples with no usable KB chunks."""
    total = len(kb_texts)
    if total == 0:
        LOGGER.info("%s KB coverage: no examples", split_name)
        return

    zero_kb = 0
    for chunks in kb_texts:
        valid_count = sum(
            1 for chunk in chunks if isinstance(chunk, str) and chunk.strip()
        )
        if valid_count == 0:
            zero_kb += 1

    pct = 100.0 * zero_kb / total
    LOGGER.info(
        "%s KB coverage: zero-valid-KB %d/%d (%.2f%%)",
        split_name,
        zero_kb,
        total,
        pct,
    )


def build_rag_contexts(
    df,
    context_cfg: dict,
    rag_cfg: dict,
    retriever,
    *,
    doc_tokenizer=None,
    kb_tokenizer=None,
    doc_budget_tokens: int | None = None,
    kb_budget_tokens: int | None = None,
    debug: bool = False,
    collect_kb: bool = False,
) -> dict[str, list[str]]:
    context_type = context_cfg.get("type", "sentence")
    n_prev = int(context_cfg.get("n_prev", 2))
    n_next = int(context_cfg.get("n_next", 2))
    use_rag = bool(rag_cfg.get("enabled", False))
    top_k = int(rag_cfg.get("top_k", 5))

    docs, idx_map = _build_document_index(df)
    doc_texts: list[str] = []
    kb_texts: list[list[str]] = []
    kb_info: list[list[dict]] = []

    for row in df.to_dict(orient="records"):
        text_id = str(row["text_id"])
        sent_id = str(row["sent_id"])
        doc_sentences = docs[text_id]
        target_idx = idx_map[(text_id, sent_id)]

        if context_type == "sentence":
            context = build_sentence_context(doc_sentences, target_idx, debug=debug)
        elif context_type == "window":
            context = build_window_context(
                doc_sentences,
                target_idx,
                n_prev=n_prev,
                n_next=n_next,
                marker_style="deberta",
                debug=debug,
            )
        elif context_type == "doc":
            if doc_tokenizer is not None and doc_budget_tokens:
                context = _build_doc_budget_context(
                    doc_sentences,
                    target_idx,
                    tokenizer=doc_tokenizer,
                    max_tokens=int(doc_budget_tokens),
                    marker_style="deberta",
                )
            else:
                context = build_doc_context(
                    doc_sentences,
                    target_idx,
                    marker_style="deberta",
                    debug=debug,
                )
        else:
            raise ValueError(f"Unknown context type: {context_type}")

        chunks: list[dict] = []
        if use_rag and retriever is not None:
            chunks = retriever.retrieve(context, top_k=top_k)
        kb_list = [c.get("text", "") for c in chunks] if chunks else []
        if kb_tokenizer is not None and kb_budget_tokens:
            kb_list = _truncate_kb_chunks_to_budget(
                kb_list, kb_tokenizer, int(kb_budget_tokens)
            )

        doc_texts.append(context)
        kb_texts.append(kb_list)
        if collect_kb:
            kb_info.append(chunks)

    if debug and doc_texts:
        LOGGER.debug(
            "Built %d contexts (context_type=%s)", len(doc_texts), context_type
        )
    result = {"doc_texts": doc_texts, "kb_texts": kb_texts}
    if collect_kb:
        result["kb_info"] = kb_info
    return result


def build_rag_dataloader(
    doc_texts: list[str],
    kb_texts: list[list[str]],
    labels: np.ndarray,
    tokenizers: dict,
    batch_size: int,
    max_length: int,
    rag_mode: str,
    shuffle: bool = True,
) -> DataLoader:
    if rag_mode not in {"late", "cross_attention"}:
        raise ValueError(
            "build_rag_dataloader only supports rag_mode in "
            "{'late', 'cross_attention'}"
        )

    examples = [
        RagExample(
            doc_text=text,
            kb_texts=kb_texts[idx] if idx < len(kb_texts) else [],
            labels=torch.tensor(label, dtype=torch.float32),
        )
        for idx, (text, label) in enumerate(zip(doc_texts, labels, strict=False))
    ]

    def collate(batch: list[RagExample]):
        batch_labels = torch.stack([ex.labels for ex in batch])

        # Tokenize doc and KB separately for architecture-specific fusion.
        doc_texts_batch = [ex.doc_text for ex in batch]
        doc_encoded = encode_batch(
            tokenizers["doc"], doc_texts_batch, max_length=max_length
        )

        # pad KB lists to max_k
        max_k = max((len(ex.kb_texts) for ex in batch), default=1)
        if max_k == 0:
            max_k = 1
        flat_kb: list[str] = []
        missing_mask: list[bool] = []
        for ex in batch:
            kb_list = list(ex.kb_texts)
            if len(kb_list) < max_k:
                kb_list = kb_list + [""] * (max_k - len(kb_list))
                missing_mask.extend([False] * len(ex.kb_texts))
                missing_mask.extend([True] * (max_k - len(ex.kb_texts)))
            else:
                missing_mask.extend([False] * max_k)
            flat_kb.extend(kb_list)

        kb_encoded = tokenizers["kb"](
            flat_kb,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if missing_mask:
            pad_id = tokenizers["kb"].pad_token_id
            for idx, missing in enumerate(missing_mask):
                if missing:
                    kb_encoded["input_ids"][idx].fill_(pad_id)
                    kb_encoded["attention_mask"][idx].zero_()

        seq_len = kb_encoded["input_ids"].size(1)
        kb_input_ids = kb_encoded["input_ids"].view(len(batch), max_k, seq_len)
        kb_attention = kb_encoded["attention_mask"].view(len(batch), max_k, seq_len)

        return {
            "doc_input_ids": doc_encoded["input_ids"],
            "doc_attention_mask": doc_encoded["attention_mask"],
            "kb_input_ids": kb_input_ids,
            "kb_attention_mask": kb_attention,
            "labels": batch_labels,
        }

    return DataLoader(
        RagDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


def _get_logits(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def _evaluate_rag(
    model,
    dataloader,
    device,
    *,
    label_names: list[str],
    threshold: float,
    use_bf16: bool = False,
) -> dict[str, float]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast_context(use_bf16):
                outputs = model(**batch)
                logits = _get_logits(outputs)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.vstack(all_labels) if all_labels else np.zeros((0, 0))
    y_pred = np.vstack(all_preds) if all_preds else np.zeros((0, 0))
    return compute_f1_metrics(y_true, y_pred, label_names=label_names)


def run_eval_rag(
    config: dict,
    *,
    checkpoint_path: Path,
    split: str,
    output_pred_path: Path,
    output_metrics_path: Path,
    debug: bool = False,
) -> dict[str, object]:
    """Run evaluation for late/cross-attention RAG and save predictions + metrics."""
    label_names = get_label_names()
    num_labels = len(label_names)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    training_cfg = config.get("training", {})
    use_rag = bool(rag_cfg.get("enabled", False))
    rag_mode = rag_cfg.get("mode", "none") if use_rag else "none"
    max_length = int(training_cfg.get("max_length", 1024))
    kb_budget_tokens = rag_cfg.get("kb_max_tokens")
    doc_budget_tokens = context_cfg.get("doc_max_tokens")
    if doc_budget_tokens is None and kb_budget_tokens is not None:
        doc_budget_tokens = max_length - int(kb_budget_tokens)

    if rag_mode not in {"late", "cross_attention"}:
        raise ValueError(
            f"run_eval_rag only supports late/cross_attention (got {rag_mode})"
        )

    retriever = init_retriever(
        rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
        rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
        debug=debug,
    )

    model, tokenizers = build_rag_model(config, num_labels=num_labels)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = _resolve_bf16(training_cfg, device)
    model.to(device)
    if not use_bf16:
        model = model.float().to(device)
    elif use_bf16:
        LOGGER.info("Enabled bf16 autocast for RAG evaluation")

    batch_size = int(training_cfg.get("batch_size", 8))
    threshold = float(training_cfg.get("pred_threshold", 0.5))
    doc_tokenizer = tokenizers.get("doc")
    kb_tokenizer = tokenizers.get("kb", doc_tokenizer)

    df = load_split(split)
    ctx = build_rag_contexts(
        df,
        context_cfg,
        rag_cfg,
        retriever,
        doc_tokenizer=doc_tokenizer,
        kb_tokenizer=kb_tokenizer,
        doc_budget_tokens=doc_budget_tokens,
        kb_budget_tokens=kb_budget_tokens,
        debug=debug,
        collect_kb=True,
    )
    _log_zero_kb_rate(split, ctx["kb_texts"])
    labels = df[label_names].to_numpy(dtype=int)

    dataloader = build_rag_dataloader(
        ctx["doc_texts"],
        ctx["kb_texts"],
        labels,
        tokenizers,
        batch_size,
        max_length,
        rag_mode,
        shuffle=False,
    )

    model.eval()
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pred_path.open("w", encoding="utf-8") as handle:
        for start, batch in enumerate(dataloader):
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                with _autocast_context(use_bf16):
                    outputs = model(**batch)
                    logits = _get_logits(outputs)
                probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_labels.append(labels_batch.cpu().numpy())
            all_preds.append(preds)

            for idx, pred_vec in enumerate(preds):
                row_idx = start * batch_size + idx
                if row_idx >= len(df):
                    continue
                row = df.iloc[row_idx]
                gold_labels = [
                    label_names[i] for i, val in enumerate(labels[row_idx]) if val == 1
                ]
                pred_labels = [
                    label_names[i] for i, val in enumerate(pred_vec) if val == 1
                ]
                chunk_list = ctx.get("kb_info", [[]])[row_idx]
                kb_values = []
                kb_ids = []
                for chunk in chunk_list:
                    kb_ids.append(chunk.get("id"))
                    kb_values.extend(chunk.get("values", []))
                kb_values = sorted({v for v in kb_values if v})
                record = {
                    "text_id": str(row["text_id"]),
                    "sent_id": str(row["sent_id"]),
                    "gold_labels": gold_labels,
                    "pred_labels": pred_labels,
                    "kb_chunk_ids": kb_ids,
                    "kb_values": kb_values,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    y_true = np.vstack(all_labels) if all_labels else np.zeros((0, 0))
    y_pred = np.vstack(all_preds) if all_preds else np.zeros((0, 0))
    metrics = compute_f1_metrics(y_true, y_pred, label_names=label_names)
    metrics["meta"] = {
        "model_name": config.get("model", {}).get("name", "microsoft/deberta-v3-base"),
        "context_type": context_cfg.get("type", "sentence"),
        "rag_mode": rag_mode,
        "use_rag": use_rag,
        "top_k": int(rag_cfg.get("top_k", 5)),
        "seed": config.get("seed", 42),
        "split": split,
    }

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def train_and_eval_rag(
    config: dict,
    run_name: str | None = None,
    *,
    resume_path: Path | None = None,
) -> dict[str, object]:
    """Train and evaluate a RAG architecture for one config."""
    label_names = get_label_names()
    num_labels = len(label_names)

    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    training_cfg = config.get("training", {})

    set_seed(int(config.get("seed", 42)))

    use_rag = bool(rag_cfg.get("enabled", False))
    rag_mode = rag_cfg.get("mode", "none") if use_rag else "none"
    context_type = context_cfg.get("type", "sentence")

    if rag_mode not in {"late", "cross_attention"}:
        raise ValueError(
            "train_and_eval_rag only supports rag.mode in "
            "{'late', 'cross_attention'}"
        )

    if run_name is None:
        run_name = f"rag_{rag_mode}_{context_type}"

    retriever = (
        init_retriever(
            rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
            rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
            debug=bool(config.get("debug", False)),
        )
        if use_rag
        else None
    )

    train_df = load_split("training")
    val_df = load_split("validation")
    max_samples = config.get("max_samples")
    if max_samples is not None:
        train_df = train_df.head(int(max_samples))
        val_df = val_df.head(int(max_samples))
    train_labels = train_df[label_names].to_numpy(dtype=int)
    val_labels = val_df[label_names].to_numpy(dtype=int)

    model, tokenizers = build_rag_model(config, num_labels=num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = _resolve_bf16(training_cfg, device)
    if training_cfg.get("gradient_checkpointing", False):
        enabled_on = _enable_gradient_checkpointing(model)
        if enabled_on:
            LOGGER.info(
                "Enabled gradient checkpointing on: %s", ", ".join(enabled_on)
            )
        else:
            LOGGER.warning("Gradient checkpointing not supported by this model")
    model.to(device)
    if not use_bf16:
        model = model.float().to(device)
    elif use_bf16:
        LOGGER.info("Enabled bf16 autocast for RAG training/evaluation")
    bad_param = _first_non_finite_param(model)
    if bad_param is not None:
        bad_name, bad_tensor = bad_param
        raise RuntimeError(
            f"Model has non-finite parameters before training ({bad_name}: {_tensor_stats(bad_tensor)})"
        )

    batch_size = int(training_cfg.get("batch_size", 8))
    max_length = int(training_cfg.get("max_length", 1024))
    kb_budget_tokens = rag_cfg.get("kb_max_tokens")
    doc_budget_tokens = context_cfg.get("doc_max_tokens")
    if doc_budget_tokens is None and kb_budget_tokens is not None:
        doc_budget_tokens = max_length - int(kb_budget_tokens)

    doc_tokenizer = tokenizers.get("doc")
    kb_tokenizer = tokenizers.get("kb", doc_tokenizer)
    train_ctx = build_rag_contexts(
        train_df,
        context_cfg,
        rag_cfg,
        retriever,
        doc_tokenizer=doc_tokenizer,
        kb_tokenizer=kb_tokenizer,
        doc_budget_tokens=doc_budget_tokens,
        kb_budget_tokens=kb_budget_tokens,
    )
    _log_zero_kb_rate("train", train_ctx["kb_texts"])
    val_ctx = build_rag_contexts(
        val_df,
        context_cfg,
        rag_cfg,
        retriever,
        doc_tokenizer=doc_tokenizer,
        kb_tokenizer=kb_tokenizer,
        doc_budget_tokens=doc_budget_tokens,
        kb_budget_tokens=kb_budget_tokens,
    )
    _log_zero_kb_rate("validation", val_ctx["kb_texts"])

    num_epochs = int(training_cfg.get("num_epochs", 20))
    learning_rate = float(training_cfg.get("learning_rate", 1e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.15))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    patience = int(training_cfg.get("early_stopping_patience", 3))
    threshold = float(training_cfg.get("pred_threshold", 0.5))
    fail_fast_on_nan = bool(training_cfg.get("fail_fast_on_nan", False)) or bool(
        config.get("debug", False)
    )
    nan_debug_dumped = False

    train_loader = build_rag_dataloader(
        train_ctx["doc_texts"],
        train_ctx["kb_texts"],
        train_labels,
        tokenizers,
        batch_size,
        max_length,
        rag_mode,
    )
    val_loader = build_rag_dataloader(
        val_ctx["doc_texts"],
        val_ctx["kb_texts"],
        val_labels,
        tokenizers,
        batch_size,
        max_length,
        rag_mode,
        shuffle=False,
    )

    optimizer = _build_adamw(
        model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        training_cfg=training_cfg,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    save_checkpoints = bool(config.get("save_checkpoints", True))

    results_dir = Path(config.get("results_dir", "results"))
    best_path = results_dir / "checkpoints" / f"{run_name}.pt"
    last_path = results_dir / "checkpoints" / f"{run_name}_last.pt"
    if save_checkpoints:
        best_path.parent.mkdir(parents=True, exist_ok=True)

    best_macro = float("-inf")
    no_improve = 0
    start_epoch = 1
    checkpoint_every = int(training_cfg.get("checkpoint_every_epochs", 1))

    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_macro = float(checkpoint.get("best_metric", best_macro))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
        else:
            # Backward compatibility with plain model state_dict checkpoints.
            model.load_state_dict(checkpoint)
        LOGGER.info("Resumed training from %s (epoch %d)", resume_path, start_epoch)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        step = 0
        batches_used = 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast_context(use_bf16):
                outputs = model(**batch)
                logits = _get_logits(outputs)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                if not nan_debug_dumped:
                    _log_nan_batch_debug(
                        model=model,
                        batch=batch,
                        labels=labels,
                        logits=logits,
                        reason="non_finite_logits",
                        use_bf16=use_bf16,
                    )
                    nan_debug_dumped = True
                bad_param = _first_non_finite_param(model)
                if bad_param is not None:
                    bad_name, bad_tensor = bad_param
                    LOGGER.error(
                        "Model parameters became non-finite (%s: %s)",
                        bad_name,
                        _tensor_stats(bad_tensor),
                    )
                    raise RuntimeError("Non-finite model parameters detected")
                if fail_fast_on_nan:
                    raise RuntimeError(
                        "NaN/Inf logits encountered (fail_fast_on_nan enabled)"
                    )
                LOGGER.warning("NaN/Inf logits detected; skipping batch")
                continue
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                if not nan_debug_dumped:
                    _log_nan_batch_debug(
                        model=model,
                        batch=batch,
                        labels=labels,
                        logits=logits,
                        reason="non_finite_labels",
                        use_bf16=use_bf16,
                    )
                    nan_debug_dumped = True
                if fail_fast_on_nan:
                    raise RuntimeError(
                        "NaN/Inf labels encountered (fail_fast_on_nan enabled)"
                    )
                LOGGER.warning("NaN/Inf labels detected; skipping batch")
                continue
            loss = loss_fn(logits.float(), labels.float())
            if torch.isnan(loss) or torch.isinf(loss):
                if not nan_debug_dumped:
                    _log_nan_batch_debug(
                        model=model,
                        batch=batch,
                        labels=labels,
                        logits=logits,
                        reason="non_finite_loss",
                        use_bf16=use_bf16,
                    )
                    nan_debug_dumped = True
                if fail_fast_on_nan:
                    raise RuntimeError(
                        "NaN/Inf loss encountered (fail_fast_on_nan enabled)"
                    )
                LOGGER.warning("NaN/Inf loss detected; skipping batch")
                continue
            loss = loss / grad_accum_steps
            loss.backward()
            step += 1
            batches_used += 1

            if step % grad_accum_steps == 0:
                bad_grad = _first_non_finite_grad(model)
                if bad_grad is not None:
                    bad_name, bad_tensor = bad_grad
                    LOGGER.error(
                        "Non-finite gradients detected before optimizer step (%s: %s)",
                        bad_name,
                        _tensor_stats(bad_tensor),
                    )
                    optimizer.zero_grad()
                    if fail_fast_on_nan:
                        raise RuntimeError("Non-finite gradients encountered")
                    continue
                if max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        LOGGER.error(
                            "Non-finite grad_norm detected (value=%s); skipping optimizer step",
                            str(grad_norm),
                        )
                        optimizer.zero_grad()
                        if fail_fast_on_nan:
                            raise RuntimeError("Non-finite grad_norm encountered")
                        continue
                optimizer.step()
                optimizer.zero_grad()
                bad_param = _first_non_finite_param(model)
                if bad_param is not None:
                    bad_name, bad_tensor = bad_param
                    LOGGER.error(
                        "Model parameters became non-finite after optimizer step (%s: %s)",
                        bad_name,
                        _tensor_stats(bad_tensor),
                    )
                    raise RuntimeError("Non-finite model parameters after optimizer step")

            total_loss += float(loss.item()) * grad_accum_steps

        if batches_used > 0 and (step % grad_accum_steps != 0):
            do_flush_step = True
            bad_grad = _first_non_finite_grad(model)
            if bad_grad is not None:
                bad_name, bad_tensor = bad_grad
                LOGGER.error(
                    "Non-finite gradients detected before flush step (%s: %s)",
                    bad_name,
                    _tensor_stats(bad_tensor),
                )
                optimizer.zero_grad()
                if fail_fast_on_nan:
                    raise RuntimeError("Non-finite gradients encountered on flush step")
                do_flush_step = False
            if do_flush_step and max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    LOGGER.error(
                        "Non-finite grad_norm on flush step (value=%s); skipping optimizer step",
                        str(grad_norm),
                    )
                    optimizer.zero_grad()
                    if fail_fast_on_nan:
                        raise RuntimeError("Non-finite grad_norm on flush step")
                    do_flush_step = False
            if do_flush_step:
                optimizer.step()
                optimizer.zero_grad()
                bad_param = _first_non_finite_param(model)
                if bad_param is not None:
                    bad_name, bad_tensor = bad_param
                    LOGGER.error(
                        "Model parameters became non-finite after flush optimizer step (%s: %s)",
                        bad_name,
                        _tensor_stats(bad_tensor),
                    )
                    raise RuntimeError(
                        "Non-finite model parameters after flush optimizer step"
                    )

        if batches_used == 0:
            LOGGER.warning(
                "No valid batches in epoch %d; stopping training early.", epoch
            )
            break

        metrics = _evaluate_rag(
            model,
            val_loader,
            device,
            label_names=label_names,
            threshold=threshold,
            use_bf16=use_bf16,
        )
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        LOGGER.info(
            "Epoch %d/%d loss=%.4f macro_f1=%.4f",
            epoch,
            num_epochs,
            total_loss / max(1, batches_used),
            macro_f1,
        )

        if macro_f1 > best_macro:
            best_macro = macro_f1
            no_improve = 0
            if save_checkpoints:
                torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info(
                    "Early stopping after %d epochs without improvement", patience
                )
                break

        if (
            save_checkpoints
            and checkpoint_every > 0
            and (epoch % checkpoint_every == 0)
        ):
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_macro,
                },
                last_path,
            )
            LOGGER.info("Saved last checkpoint to %s", last_path)

    return {
        "macro_f1": best_macro,
        "checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "checkpoint_saved": save_checkpoints,
    }
