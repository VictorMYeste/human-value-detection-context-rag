"""Training and evaluation loops for DeBERTa."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from value_context_rag.kb.retriever import init_retriever
from value_context_rag.eval.metrics import compute_f1_metrics
from value_context_rag.models.deberta import build_deberta_model, encode_batch
from value_context_rag.utils.logging import get_logger
from value_context_rag.utils.seed import set_seed

LOGGER = get_logger(__name__)


@dataclass
class TextExample:
    text: str
    labels: torch.Tensor


class TextDataset(Dataset):
    def __init__(self, examples: List[TextExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TextExample:
        return self.examples[idx]


def _safe_int(value: str) -> Tuple[int, bool]:
    try:
        return int(value), True
    except Exception:
        return 0, False


def _order_doc_rows(rows: List[dict]) -> List[dict]:
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
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], int]]:
    docs: Dict[str, List[str]] = {}
    index: Dict[Tuple[str, str], int] = {}

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
    collect_kb: bool = False,
) -> List[str] | tuple[List[str], List[List[dict]]]:
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
    contexts: List[str] = []
    kb_info: List[List[dict]] = []

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
            context = build_doc_context(
                doc_sentences,
                target_idx,
                marker_style="deberta",
                debug=debug,
            )
        else:
            raise ValueError(f"Unknown context type: {context_type}")

        chunks: List[dict] = []
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
                context = f"KNOWLEDGE:\\n{kb_text}\\n\\nTEXT:\\n{context}"
        if debug and len(contexts) < 3:
            LOGGER.debug("Sample context %d length=%d", len(contexts), len(context))

        contexts.append(context)
        if collect_kb:
            kb_info.append(chunks)

    if collect_kb:
        return contexts, kb_info
    return contexts


def _build_dataloader(
    texts: List[str],
    labels: np.ndarray,
    tokenizer,
    *,
    batch_size: int,
    shuffle: bool,
    max_length: int,
) -> DataLoader:
    examples = [
        TextExample(text=text, labels=torch.tensor(label, dtype=torch.float32))
        for text, label in zip(texts, labels)
    ]

    def collate(batch: List[TextExample]):
        batch_texts = [ex.text for ex in batch]
        batch_labels = torch.stack([ex.labels for ex in batch])
        encoded = encode_batch(tokenizer, batch_texts, max_length=max_length)
        encoded["labels"] = batch_labels
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


def _evaluate(model, dataloader, device, label_names: List[str]) -> Dict[str, float]:
    model.eval()
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.vstack(all_labels) if all_labels else np.zeros((0, 0))
    y_pred = np.vstack(all_preds) if all_preds else np.zeros((0, 0))
    return compute_f1_metrics(y_true, y_pred, label_names=label_names)


def save_predictions_jsonl(
    model,
    tokenizer,
    df,
    label_names: List[str],
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
    debug: bool,
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
                logits = model(**encoded)
                probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

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
    config: Dict,
    *,
    checkpoint_path: Path,
    split: str,
    output_pred_path: Path,
    output_metrics_path: Path,
    debug: bool = False,
) -> Dict[str, object]:
    """Run evaluation for a given split and save predictions + metrics."""
    label_names = get_label_names()
    model, tokenizer = build_deberta_model(num_labels=len(label_names))
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

    context_type = context_cfg.get("type", "sentence")
    n_prev = int(context_cfg.get("n_prev", 2))
    n_next = int(context_cfg.get("n_next", 2))
    use_rag = bool(rag_cfg.get("enabled", False))
    top_k = int(rag_cfg.get("top_k", 5))

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
        collect_kb=True,
    )
    if isinstance(contexts_result, tuple):
        texts, kb_info = contexts_result
    else:
        texts, kb_info = contexts_result, [[] for _ in range(len(contexts_result))]
    labels = df[label_names].to_numpy(dtype=int)

    model.eval()
    all_preds: List[np.ndarray] = []
    batch_size = int(training_cfg.get("batch_size", 16))
    max_length = int(training_cfg.get("max_length", 1024))

    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pred_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_labels = labels[start : start + batch_size]
            encoded = encode_batch(tokenizer, batch_texts, max_length=max_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded)
                probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
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

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Saved predictions to %s", output_pred_path)
    LOGGER.info("Saved metrics to %s", output_metrics_path)
    return metrics


def train_and_eval(
    config: Dict,
    *,
    run_name: str | None = None,
    resume_path: Path | None = None,
) -> float:
    """Train DeBERTa and evaluate on validation (and optional test)."""
    set_seed(config.get("seed", 42))

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 16))
    num_epochs = int(training_cfg.get("num_epochs", 3))
    learning_rate = float(training_cfg.get("learning_rate", 2e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    max_length = int(training_cfg.get("max_length", 512))
    early_patience = int(training_cfg.get("early_stopping_patience", 3))

    context_cfg = config.get("context", {})
    context_type = context_cfg.get("type", "sentence")
    n_prev = int(context_cfg.get("n_prev", 2))
    n_next = int(context_cfg.get("n_next", 2))

    rag_cfg = config.get("rag", {})
    use_rag = bool(rag_cfg.get("enabled", False))
    top_k = int(rag_cfg.get("top_k", 5))

    results_dir = Path(config.get("results_dir", "results"))
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading dataset splits")
    train_df = load_split("train")
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

    train_texts = build_contexts(
        train_df,
        context_type=context_type,
        n_prev=n_prev,
        n_next=n_next,
        use_rag=use_rag,
        top_k=top_k,
        retriever=retriever,
        debug=False,
    )
    val_texts = build_contexts(
        val_df,
        context_type=context_type,
        n_prev=n_prev,
        n_next=n_next,
        use_rag=use_rag,
        top_k=top_k,
        retriever=retriever,
        debug=False,
    )

    model, tokenizer = build_deberta_model(num_labels=len(label_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_metric = -math.inf
    suffix = run_name or "deberta_best"
    best_path = ckpt_dir / f"{suffix}.pt"
    last_path = ckpt_dir / f"{suffix}_last.pt"
    start_epoch = 1
    epochs_no_improve = 0

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

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            if "input_ids" in batch:
                lengths = (batch["input_ids"] != 0).sum(dim=1)
                token_stats["train_truncated"] += int(
                    (lengths >= max_length).sum().item()
                )
                token_stats["train_total"] += int(lengths.numel())
            optimizer.zero_grad()
            logits = model(**batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        LOGGER.info("Epoch %d/%d - train loss %.4f", epoch, num_epochs, avg_loss)

        metrics = _evaluate(model, val_loader, device, label_names=label_names)
        LOGGER.info(
            "Validation metrics - macro_f1=%.4f micro_f1=%.4f",
            metrics["macro_f1"],
            metrics["micro_f1"],
        )

        if metrics["macro_f1"] > best_metric:
            best_metric = metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)
            LOGGER.info("Saved best checkpoint to %s", best_path)
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

        if checkpoint_every > 0 and (epoch % checkpoint_every == 0):
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
        )
        test_loader = _build_dataloader(
            test_texts,
            test_labels,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_length,
        )
        test_metrics = _evaluate(model, test_loader, device, label_names=label_names)
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
    stats_path.write_text(
        json.dumps(token_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved token stats to %s", stats_path)
    return float(best_metric)
