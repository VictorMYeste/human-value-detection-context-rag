"""Training and evaluation loops for DeBERTa."""

from __future__ import annotations

import inspect
import json
import math
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
from value_context_rag.eval.metrics import compute_f1_metrics, sweep_thresholds
from value_context_rag.kb.retriever import init_retriever
from value_context_rag.models.deberta import build_deberta_model, encode_batch
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
        LOGGER.warning("bf16 requested but CUDA bf16 is unsupported; falling back to fp32")
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


def _save_hf_bundle(
    model,
    tokenizer,
    label_names: list[str],
    output_dir: Path,
    *,
    extra_info: dict | None = None,
) -> None:
    """Save model + tokenizer artifacts in a HF-friendly folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(output_dir)
    except Exception:
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    (output_dir / "label_names.json").write_text(
        json.dumps(label_names, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception:
        LOGGER.warning("Tokenizer could not be saved to %s", output_dir)

    # Ensure SentencePiece model and special tokens map exist.
    spm_path = output_dir / "spm.model"
    if not spm_path.exists():
        spm_source = getattr(tokenizer, "vocab_file", None)
        if spm_source and Path(spm_source).exists():
            spm_path.write_bytes(Path(spm_source).read_bytes())
    stm_path = output_dir / "special_tokens_map.json"
    if not stm_path.exists():
        try:
            stm_path.write_text(
                json.dumps(tokenizer.special_tokens_map, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            LOGGER.warning("Could not write special_tokens_map.json to %s", output_dir)

    # Save training args (minimal, for reproducibility).
    training_args = extra_info or {}
    torch.save(training_args, output_dir / "training_args.bin")

    # Minimal model card
    model_name = training_args.get("model_name", "microsoft/deberta-v3-base")
    task = training_args.get("task", "multi_label_classification")
    context = training_args.get("context_type", "sentence")
    rag = training_args.get("use_rag", False)
    top_k = training_args.get("top_k", None)
    lines = [
        "# DeBERTa-v3 Multi-label Model",
        "",
        f"- Base model: `{model_name}`",
        f"- Task: `{task}`",
        f"- Labels: `{len(label_names)}`",
        f"- Context: `{context}`",
        f"- RAG: `{rag}`",
    ]
    if top_k is not None:
        lines.append(f"- RAG top_k: `{top_k}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class TextExample:
    text: str
    labels: torch.Tensor


class TextDataset(Dataset):
    def __init__(self, examples: list[TextExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TextExample:
        return self.examples[idx]


def _safe_int(value: str) -> tuple[int, bool]:
    try:
        return int(value), True
    except Exception:
        return 0, False


def _order_doc_rows(rows: list[dict]) -> list[dict]:
    with_parsed = []
    all_numeric = True
    for row in rows:
        sent_id = str(row.get("sent_id", ""))
        parsed, ok = _safe_int(sent_id)
        if not ok:
            all_numeric = False
        with_parsed.append((parsed, sent_id, row))
    if all_numeric:
        with_parsed.sort(key=lambda x: x[0])
    else:
        with_parsed.sort(key=lambda x: x[1])
    return [row for _, _, row in with_parsed]


def _build_document_index(
    df,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], int]]:
    docs: dict[str, list[str]] = {}
    index: dict[tuple[str, str], int] = {}

    grouped = {}
    for row in df.to_dict(orient="records"):
        grouped.setdefault(row["text_id"], []).append(row)

    for text_id, rows in grouped.items():
        ordered = _order_doc_rows(rows)
        sentences = [str(r["text"]) for r in ordered]
        docs[text_id] = sentences
        for idx, row in enumerate(ordered):
            index[(str(row["text_id"]), str(row["sent_id"]))] = idx

    return docs, index


def _truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


def _build_doc_budget_context(
    doc_sentences: list[str],
    target_idx: int,
    *,
    tokenizer,
    max_tokens: int,
    marker_style: str,
) -> str:
    def mark(sentence: str, is_target: bool) -> str:
        if not is_target:
            return sentence
        if marker_style == "deberta":
            return f"<TGT>{sentence}</TGT>"
        return f"<<<<TARGET>>>> {sentence} <<<<END>>>>"

    sentence_texts = [
        mark(sent, idx == target_idx) for idx, sent in enumerate(doc_sentences)
    ]
    token_counts = [
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in sentence_texts
    ]

    selected = {target_idx}
    total_tokens = token_counts[target_idx]
    offset = 1
    while True:
        added = False
        for cand in (target_idx - offset, target_idx + offset):
            if cand < 0 or cand >= len(doc_sentences):
                continue
            if cand in selected:
                continue
            if total_tokens + token_counts[cand] > max_tokens:
                continue
            selected.add(cand)
            total_tokens += token_counts[cand]
            added = True
        if not added:
            break
        offset += 1

    ordered = [idx for idx in range(len(doc_sentences)) if idx in selected]
    return " ".join(sentence_texts[idx] for idx in ordered)


def build_contexts(
    df,
    *,
    context_type: str,
    n_prev: int,
    n_next: int,
    use_rag: bool,
    top_k: int,
    retriever,
    debug: bool,
    tokenizer=None,
    doc_budget_tokens: int | None = None,
    kb_budget_tokens: int | None = None,
    collect_kb: bool = False,
) -> list[str] | tuple[list[str], list[list[dict]]]:
    docs, idx_map = _build_document_index(df)
    if debug:
        LOGGER.debug(
            "Building contexts: context=%s n_prev=%d n_next=%d use_rag=%s top_k=%d",
            context_type,
            n_prev,
            n_next,
            use_rag,
            top_k,
        )
    contexts: list[str] = []
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
            if tokenizer is not None and doc_budget_tokens:
                context = _build_doc_budget_context(
                    doc_sentences,
                    target_idx,
                    tokenizer=tokenizer,
                    max_tokens=doc_budget_tokens,
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
            if debug and chunks:
                LOGGER.debug(
                    "Retrieved %d KB chunks (ids=%s values=%s)",
                    len(chunks),
                    [c.get("id") for c in chunks],
                    [c.get("values", []) for c in chunks],
                )
            if chunks:
                kb_text = "\n\n".join(chunk["text"] for chunk in chunks)
                if tokenizer is not None and kb_budget_tokens:
                    kb_text = _truncate_to_tokens(
                        kb_text, tokenizer, kb_budget_tokens
                    )
                context = f"TEXT:\\n{context}\\n\\nKNOWLEDGE:\\n{kb_text}"
        if debug and len(contexts) < 3:
            LOGGER.debug("Sample context %d length=%d", len(contexts), len(context))

        contexts.append(context)
        if collect_kb:
            kb_info.append(chunks)

    if collect_kb:
        return contexts, kb_info
    return contexts


def _build_dataloader(
    texts: list[str],
    labels: np.ndarray,
    tokenizer,
    *,
    batch_size: int,
    shuffle: bool,
    max_length: int,
) -> DataLoader:
    examples = [
        TextExample(text=text, labels=torch.tensor(label, dtype=torch.float32))
        for text, label in zip(texts, labels, strict=False)
    ]

    def _overflow_flags(batch_texts: list[str]) -> list[bool]:
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_length=True,
            padding=False,
        )
        mapping = encoded.get("overflow_to_sample_mapping", [])
        counts = [0 for _ in batch_texts]
        for idx in mapping:
            if 0 <= idx < len(counts):
                counts[idx] += 1
        return [c > 1 for c in counts]

    def collate(batch: list[TextExample]):
        batch_texts = [ex.text for ex in batch]
        batch_labels = torch.stack([ex.labels for ex in batch])
        encoded = encode_batch(tokenizer, batch_texts, max_length=max_length)
        encoded["labels"] = batch_labels
        encoded["overflowed"] = torch.tensor(
            _overflow_flags(batch_texts), dtype=torch.bool
        )
        return encoded

    if shuffle:
        LOGGER.debug("Creating shuffled dataloader (batch_size=%d)", batch_size)
    else:
        LOGGER.debug("Creating dataloader (batch_size=%d)", batch_size)
    return DataLoader(
        TextDataset(examples),
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


def _evaluate(
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
            batch.pop("overflowed", None)
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


def save_predictions_jsonl(
    model,
    tokenizer,
    df,
    label_names: list[str],
    output_path: Path,
    *,
    context_type: str,
    n_prev: int,
    n_next: int,
    use_rag: bool,
    top_k: int,
    retriever,
    max_length: int,
    batch_size: int,
    threshold: float,
    doc_budget_tokens: int | None = None,
    kb_budget_tokens: int | None = None,
    debug: bool,
    use_bf16: bool = False,
) -> None:
    """Run inference and save predictions to JSONL."""
    model.eval()
    if debug:
        LOGGER.debug("Saving predictions to %s", output_path)
    device = next(model.parameters()).device

    contexts_result = build_contexts(
        df,
        context_type=context_type,
        n_prev=n_prev,
        n_next=n_next,
        use_rag=use_rag,
        top_k=top_k,
        retriever=retriever,
        debug=debug,
        tokenizer=tokenizer,
        doc_budget_tokens=doc_budget_tokens,
        kb_budget_tokens=kb_budget_tokens,
        collect_kb=True,
    )
    if isinstance(contexts_result, tuple):
        texts, kb_info = contexts_result
    else:
        texts, kb_info = contexts_result, [[] for _ in range(len(contexts_result))]

    label_matrix = df[label_names].to_numpy(dtype=int)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_labels = label_matrix[start : start + batch_size]
            encoded = encode_batch(tokenizer, batch_texts, max_length=max_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                with _autocast_context(use_bf16):
                    outputs = model(**encoded)
                    logits = _get_logits(outputs)
                probs = torch.sigmoid(logits.float()).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            if debug and start == 0:
                gold_rate = float(batch_labels.mean()) if batch_labels.size else 0.0
                pred_rate = float(preds.mean()) if preds.size else 0.0
                LOGGER.debug(
                    "Batch0 stats: gold_rate=%.4f pred_rate=%.4f "
                    "probs[min=%.4f max=%.4f mean=%.4f] threshold=%.2f",
                    gold_rate,
                    pred_rate,
                    float(probs.min()) if probs.size else 0.0,
                    float(probs.max()) if probs.size else 0.0,
                    float(probs.mean()) if probs.size else 0.0,
                    threshold,
                )

            for idx, pred_vec in enumerate(preds):
                row_idx = start + idx
                row = df.iloc[row_idx]
                gold_labels = [
                    label_names[i]
                    for i, val in enumerate(batch_labels[idx])
                    if val == 1
                ]
                pred_labels = [
                    label_names[i] for i, val in enumerate(pred_vec) if val == 1
                ]
                chunk_list = kb_info[row_idx] if row_idx < len(kb_info) else []
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

    LOGGER.info("Saved predictions to %s", output_path)


def run_eval(
    config: dict,
    *,
    checkpoint_path: Path,
    split: str,
    output_pred_path: Path,
    output_metrics_path: Path,
    debug: bool = False,
    tune_threshold: bool = False,
    threshold_start: float = 0.0,
    threshold_stop: float = 1.0,
    threshold_step: float = 0.01,
) -> dict[str, object]:
    """Run evaluation for a given split and save predictions + metrics."""
    label_names = get_label_names()
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    model, tokenizer = build_deberta_model(
        num_labels=len(label_names),
        model_name=model_name,
        label_names=label_names,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    if debug:
        LOGGER.debug("Loaded checkpoint %s", checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    training_cfg = config.get("training", {})
    use_bf16 = _resolve_bf16(training_cfg, device)
    if use_bf16:
        LOGGER.info("Enabled bf16 autocast for evaluation")

    context_type = context_cfg.get("type", "sentence")
    n_prev = int(context_cfg.get("n_prev", 2))
    n_next = int(context_cfg.get("n_next", 2))
    use_rag = bool(rag_cfg.get("enabled", False))
    top_k = int(rag_cfg.get("top_k", 5))
    max_length = int(training_cfg.get("max_length", 1024))
    kb_budget_tokens = rag_cfg.get("kb_max_tokens")
    doc_budget_tokens = context_cfg.get("doc_max_tokens")
    if doc_budget_tokens is None and kb_budget_tokens is not None:
        doc_budget_tokens = max_length - int(kb_budget_tokens)

    LOGGER.info(
        "Context budgets: max_length=%d doc_budget=%s kb_budget=%s",
        max_length,
        str(doc_budget_tokens) if context_type == "doc" else "n/a",
        str(kb_budget_tokens) if use_rag else "n/a",
    )

    retriever = (
        init_retriever(
            rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
            rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
            debug=debug,
        )
        if use_rag
        else None
    )

    df = load_split(split)
    if debug:
        LOGGER.debug("Loaded %s split with %d rows", split, len(df))
    contexts_result = build_contexts(
        df,
        context_type=context_type,
        n_prev=n_prev,
        n_next=n_next,
        use_rag=use_rag,
        top_k=top_k,
        retriever=retriever,
        debug=debug,
        tokenizer=tokenizer,
        doc_budget_tokens=doc_budget_tokens if context_type == "doc" else None,
        kb_budget_tokens=kb_budget_tokens if use_rag else None,
        collect_kb=True,
    )
    if isinstance(contexts_result, tuple):
        texts, kb_info = contexts_result
    else:
        texts, kb_info = contexts_result, [[] for _ in range(len(contexts_result))]
    labels = df[label_names].to_numpy(dtype=int)

    model.eval()
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    batch_size = int(training_cfg.get("batch_size", 16))
    threshold = float(training_cfg.get("pred_threshold", 0.5))

    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pred_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_labels = labels[start : start + batch_size]
            encoded = encode_batch(tokenizer, batch_texts, max_length=max_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                with _autocast_context(use_bf16):
                    outputs = model(**encoded)
                    logits = _get_logits(outputs)
                probs = torch.sigmoid(logits.float()).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_probs.append(probs)
            if debug and start == 0:
                gold_rate = float(batch_labels.mean()) if batch_labels.size else 0.0
                pred_rate = float(preds.mean()) if preds.size else 0.0
                LOGGER.debug(
                    "Eval batch0 stats: gold_rate=%.4f pred_rate=%.4f "
                    "probs[min=%.4f max=%.4f mean=%.4f] threshold=%.2f",
                    gold_rate,
                    pred_rate,
                    float(probs.min()) if probs.size else 0.0,
                    float(probs.max()) if probs.size else 0.0,
                    float(probs.mean()) if probs.size else 0.0,
                    threshold,
                )
            all_preds.append(preds)

            for idx, pred_vec in enumerate(preds):
                row_idx = start + idx
                row = df.iloc[row_idx]
                gold_labels = [
                    label_names[i]
                    for i, val in enumerate(batch_labels[idx])
                    if val == 1
                ]
                pred_labels = [
                    label_names[i] for i, val in enumerate(pred_vec) if val == 1
                ]
                chunk_list = kb_info[row_idx] if row_idx < len(kb_info) else []
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

    y_pred = np.vstack(all_preds) if all_preds else np.zeros_like(labels)
    metrics = compute_f1_metrics(labels, y_pred, label_names=label_names)
    configured_mode = rag_cfg.get("mode")
    if not use_rag:
        rag_mode = "none"
    elif configured_mode in {"early", "late", "cross_attention"}:
        rag_mode = str(configured_mode)
    else:
        # Backward compatibility: legacy *_rag configs imply early fusion.
        rag_mode = "early"

    metrics["meta"] = {
        "model_name": config.get("model", {}).get("name", "microsoft/deberta-v3-base"),
        "context_type": context_type,
        "rag_mode": rag_mode,
        "use_rag": use_rag,
        "top_k": top_k,
        "seed": config.get("seed", 42),
        "split": split,
    }
    if tune_threshold:
        y_probs = np.vstack(all_probs) if all_probs else np.zeros_like(labels)
        sweep = sweep_thresholds(
            labels,
            y_probs,
            label_names=label_names,
            start=threshold_start,
            stop=threshold_stop,
            step=threshold_step,
        )
        metrics["threshold_sweep"] = {
            "best_threshold": sweep["best_threshold"],
            "best_metrics": sweep["best_metrics"],
        }
        LOGGER.info(
            "Best threshold=%.2f (macro_f1=%.4f micro_f1=%.4f)",
            sweep["best_threshold"],
            sweep["best_metrics"]["macro_f1"],
            sweep["best_metrics"]["micro_f1"],
        )

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Saved predictions to %s", output_pred_path)
    LOGGER.info("Saved metrics to %s", output_metrics_path)
    return metrics


def train_and_eval(
    config: dict,
    *,
    run_name: str | None = None,
    resume_path: Path | None = None,
) -> tuple[float, bool]:
    """Train DeBERTa and evaluate on validation (and optional test)."""
    set_seed(config.get("seed", 42))

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 16))
    num_epochs = int(training_cfg.get("num_epochs", 3))
    learning_rate = float(training_cfg.get("learning_rate", 2e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    max_length = int(training_cfg.get("max_length", 512))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    early_patience = int(training_cfg.get("early_stopping_patience", 3))
    collapse_threshold = float(training_cfg.get("collapse_threshold", 0.01))
    collapse_min_epochs = int(training_cfg.get("collapse_min_epochs", 3))
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")

    context_cfg = config.get("context", {})
    context_type = context_cfg.get("type", "sentence")
    n_prev = int(context_cfg.get("n_prev", 2))
    n_next = int(context_cfg.get("n_next", 2))
    doc_budget_tokens = context_cfg.get("doc_max_tokens")

    rag_cfg = config.get("rag", {})
    use_rag = bool(rag_cfg.get("enabled", False))
    top_k = int(rag_cfg.get("top_k", 5))
    kb_budget_tokens = rag_cfg.get("kb_max_tokens")

    results_dir = Path(config.get("results_dir", "results"))
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading dataset splits")
    train_df = load_split("training")
    val_df = load_split("validation")
    max_samples = config.get("max_samples")
    if max_samples is not None:
        train_df = train_df.head(int(max_samples))
        val_df = val_df.head(int(max_samples))

    retriever = (
        init_retriever(
            rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
            rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
            debug=False,
        )
        if use_rag
        else None
    )

    label_names = get_label_names()
    train_labels = train_df[label_names].to_numpy(dtype=float)
    val_labels = val_df[label_names].to_numpy(dtype=float)

    model, tokenizer = build_deberta_model(
        num_labels=len(label_names),
        model_name=model_name,
        label_names=label_names,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = _resolve_bf16(training_cfg, device)
    if training_cfg.get("gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
            LOGGER.info("Enabled gradient checkpointing")
        except Exception:
            LOGGER.warning("Gradient checkpointing not supported by this model")
    model.to(device)
    if not use_bf16:
        model = model.float()
        model.to(device)
        LOGGER.info("Forced model parameters to fp32")
    elif use_bf16:
        LOGGER.info("Enabled bf16 autocast for training/evaluation")

    if doc_budget_tokens is None and kb_budget_tokens is not None:
        doc_budget_tokens = max_length - int(kb_budget_tokens)

    LOGGER.info(
        "Context budgets: max_length=%d doc_budget=%s kb_budget=%s",
        max_length,
        str(doc_budget_tokens) if context_type == "doc" else "n/a",
        str(kb_budget_tokens) if use_rag else "n/a",
    )

    LOGGER.info("Building training contexts (%s)", context_type)
    train_texts = build_contexts(
        train_df,
        context_type=context_type,
        n_prev=n_prev,
        n_next=n_next,
        use_rag=use_rag,
        top_k=top_k,
        retriever=retriever,
        debug=False,
        tokenizer=tokenizer,
        doc_budget_tokens=doc_budget_tokens if context_type == "doc" else None,
        kb_budget_tokens=kb_budget_tokens if use_rag else None,
    )
    LOGGER.info("Building validation contexts (%s)", context_type)
    val_texts = build_contexts(
        val_df,
        context_type=context_type,
        n_prev=n_prev,
        n_next=n_next,
        use_rag=use_rag,
        top_k=top_k,
        retriever=retriever,
        debug=False,
        tokenizer=tokenizer,
        doc_budget_tokens=doc_budget_tokens if context_type == "doc" else None,
        kb_budget_tokens=kb_budget_tokens if use_rag else None,
    )

    train_loader = _build_dataloader(
        train_texts,
        train_labels,
        tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_length=max_length,
    )
    val_loader = _build_dataloader(
        val_texts,
        val_labels,
        tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
    )

    # Sanity check the first batch before training to catch NaNs early.
    try:
        first_batch = next(iter(train_loader))
        first_labels = first_batch.pop("labels").to(device)
        first_batch = {k: v.to(device) for k, v in first_batch.items()}
        with torch.no_grad():
            with _autocast_context(use_bf16):
                first_outputs = model(**first_batch)
                first_logits = _get_logits(first_outputs)
        if torch.isnan(first_logits).any() or torch.isinf(first_logits).any():
            LOGGER.error("NaN/Inf logits detected in sanity check batch")
            LOGGER.debug(
                "Sanity logits stats: min=%.4f max=%.4f mean=%.4f",
                first_logits.nan_to_num().min().item(),
                first_logits.nan_to_num().max().item(),
                first_logits.nan_to_num().mean().item(),
            )
            LOGGER.debug(
                "Sanity labels stats: min=%.4f max=%.4f mean=%.4f",
                first_labels.nan_to_num().min().item(),
                first_labels.nan_to_num().max().item(),
                first_labels.nan_to_num().mean().item(),
            )
            raise RuntimeError("Sanity check failed: NaN/Inf logits in first batch")
    except StopIteration:
        LOGGER.warning("Training dataloader is empty; skipping training")
        return float("-inf")

    optimizer = _build_adamw(
        model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        training_cfg=training_cfg,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_metric = -math.inf
    suffix = run_name or "deberta_best"
    best_path = ckpt_dir / f"{suffix}.pt"
    last_path = ckpt_dir / f"{suffix}_last.pt"
    start_epoch = 1
    epochs_no_improve = 0
    save_checkpoints = bool(config.get("save_checkpoints", True))
    save_hf_model = bool(training_cfg.get("save_hf_model", True))
    threshold = float(training_cfg.get("pred_threshold", 0.5))
    LOGGER.info(
        "Training config: batch=%d lr=%g wd=%g max_len=%d accum=%d save_ckpt=%s",
        batch_size,
        learning_rate,
        weight_decay,
        max_length,
        grad_accum_steps,
        save_checkpoints,
    )

    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_metric = float(checkpoint.get("best_metric", best_metric))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        LOGGER.info("Resumed training from %s (epoch %d)", resume_path, start_epoch)

    checkpoint_every = int(training_cfg.get("checkpoint_every_epochs", 1))
    token_stats = {
        "max_length": max_length,
        "train_truncated": 0,
        "train_total": 0,
    }
    val_history: list[dict[str, float]] = []
    epochs_completed = 0

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        batches_used = 0
        LOGGER.info("Starting epoch %d/%d", epoch, num_epochs)
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            overflowed = batch.pop("overflowed", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            if overflowed is not None and epoch == start_epoch:
                token_stats["train_truncated"] += int(overflowed.sum().item())
                token_stats["train_total"] += int(overflowed.numel())
            with _autocast_context(use_bf16):
                outputs = model(**batch)
                logits = _get_logits(outputs)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                LOGGER.warning("NaN/Inf logits detected; skipping batch")
                LOGGER.debug(
                    "Logits stats: min=%.4f max=%.4f mean=%.4f",
                    logits.nan_to_num().min().item(),
                    logits.nan_to_num().max().item(),
                    logits.nan_to_num().mean().item(),
                )
                LOGGER.debug(
                    "Labels stats: min=%.4f max=%.4f mean=%.4f",
                    labels.nan_to_num().min().item(),
                    labels.nan_to_num().max().item(),
                    labels.nan_to_num().mean().item(),
                )
                continue
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                LOGGER.warning("NaN/Inf labels detected; skipping batch")
                continue
            loss = loss_fn(logits.float(), labels.float()) / max(grad_accum_steps, 1)
            if torch.isnan(loss):
                LOGGER.warning("NaN loss detected; skipping batch")
                LOGGER.debug(
                    "Logits stats: min=%.4f max=%.4f mean=%.4f",
                    logits.min().item(),
                    logits.max().item(),
                    logits.mean().item(),
                )
                LOGGER.debug(
                    "Labels stats: min=%.4f max=%.4f mean=%.4f",
                    labels.min().item(),
                    labels.max().item(),
                    labels.mean().item(),
                )
                continue
            loss.backward()
            if step % grad_accum_steps == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            batches_used += 1

        # Flush gradients if we didn't hit an exact accumulation boundary.
        if batches_used > 0 and (batches_used % grad_accum_steps != 0):
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if batches_used == 0:
            LOGGER.warning(
                "No valid batches in epoch %d (all skipped due to NaN/Inf); "
                "stopping training early.",
                epoch,
            )
            break
        avg_loss = total_loss / max(batches_used, 1)
        LOGGER.info("Epoch %d/%d - train loss %.4f", epoch, num_epochs, avg_loss)

        LOGGER.info("Evaluating on validation split")
        metrics = _evaluate(
            model,
            val_loader,
            device,
            label_names=label_names,
            threshold=threshold,
            use_bf16=use_bf16,
        )
        LOGGER.info(
            "Validation metrics - macro_f1=%.4f micro_f1=%.4f",
            metrics["macro_f1"],
            metrics["micro_f1"],
        )
        val_history.append(
            {
                "epoch": float(epoch),
                "macro_f1": float(metrics["macro_f1"]),
                "micro_f1": float(metrics["micro_f1"]),
            }
        )
        epochs_completed += 1

        if metrics["macro_f1"] > best_metric:
            best_metric = metrics["macro_f1"]
            if save_checkpoints:
                torch.save(model.state_dict(), best_path)
                LOGGER.info("Saved best checkpoint to %s", best_path)
                if save_hf_model:
                    hf_dir = results_dir / "hf_models" / suffix
                    hf_meta = {
                        "model_name": model_name,
                        "task": "multi_label_classification",
                        "context_type": context_type,
                        "use_rag": use_rag,
                        "top_k": top_k,
                        "seed": config.get("seed", 42),
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "max_length": max_length,
                        "num_epochs": num_epochs,
                        "grad_accum_steps": grad_accum_steps,
                        "pred_threshold": threshold,
                    }
                    _save_hf_bundle(
                        model, tokenizer, label_names, hf_dir, extra_info=hf_meta
                    )
                    LOGGER.info("Saved HF bundle to %s", hf_dir)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            LOGGER.debug("No improvement for %d epochs", epochs_no_improve)
            if early_patience > 0 and epochs_no_improve >= early_patience:
                LOGGER.info(
                    "Early stopping triggered after %d epochs without improvement",
                    epochs_no_improve,
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
                    "best_metric": best_metric,
                },
                last_path,
            )
            LOGGER.info("Saved last checkpoint to %s", last_path)

    if config.get("eval_test", False):
        LOGGER.info("Evaluating on test split")
        test_df = load_split("test")
        test_labels = test_df[label_names].to_numpy(dtype=float)
        test_texts = build_contexts(
            test_df,
            context_type=context_type,
            n_prev=n_prev,
            n_next=n_next,
            use_rag=use_rag,
            top_k=top_k,
            retriever=retriever,
            debug=False,
            tokenizer=tokenizer,
            doc_budget_tokens=doc_budget_tokens if context_type == "doc" else None,
            kb_budget_tokens=kb_budget_tokens if use_rag else None,
        )
        test_loader = _build_dataloader(
            test_texts,
            test_labels,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_length,
        )
        test_metrics = _evaluate(
            model,
            test_loader,
            device,
            label_names=label_names,
            threshold=threshold,
        )
        LOGGER.info(
            "Test metrics - macro_f1=%.4f micro_f1=%.4f",
            test_metrics["macro_f1"],
            test_metrics["micro_f1"],
        )

    stats_path = results_dir / "logs" / f"token_stats_{suffix}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if token_stats["train_total"] > 0:
        token_stats["train_truncated_rate"] = (
            token_stats["train_truncated"] / token_stats["train_total"]
        )
    token_stats["truncation_method"] = "overflow_to_sample_mapping"
    stats_path.write_text(
        json.dumps(token_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved token stats to %s", stats_path)
    if val_history:
        LOGGER.info("Validation summary (per epoch):")
        header = f"{'epoch':>5}  {'macro_f1':>8}  {'micro_f1':>8}"
        LOGGER.info(header)
        for row in val_history:
            LOGGER.info(
                "%5d  %8.4f  %8.4f",
                int(row["epoch"]),
                row["macro_f1"],
                row["micro_f1"],
            )
    collapsed = False
    if epochs_completed >= collapse_min_epochs and best_metric < collapse_threshold:
        collapsed = True
        LOGGER.warning(
            "Run flagged as collapsed (best_macro_f1=%.4f < %.4f after %d epochs)",
            best_metric,
            collapse_threshold,
            epochs_completed,
        )
    return float(best_metric), collapsed
