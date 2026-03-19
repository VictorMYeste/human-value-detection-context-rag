"""Training utilities for RAG architectures (late/cross/early)."""

from __future__ import annotations

import json
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
from value_context_rag.utils.logging import get_logger
from value_context_rag.utils.seed import set_seed

LOGGER = get_logger(__name__)


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


def build_rag_contexts(
    df,
    context_cfg: dict,
    rag_cfg: dict,
    retriever,
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


def _concat_early(doc_text: str, kb_list: list[str]) -> str:
    if not kb_list:
        return doc_text
    kb_text = "\n\n".join(kb_list)
    return f"TEXT:\n{doc_text}\n\nKNOWLEDGE:\n{kb_text}"


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
        if rag_mode in {"none", "early"}:
            if rag_mode == "early":
                texts = [_concat_early(ex.doc_text, ex.kb_texts) for ex in batch]
            else:
                texts = [ex.doc_text for ex in batch]
            encoded = encode_batch(tokenizers["doc"], texts, max_length=max_length)
            encoded["labels"] = batch_labels
            return encoded

        # late / cross_attention: tokenize doc + KB separately
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
) -> dict[str, float]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
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

    if rag_mode not in {"late", "cross_attention"}:
        raise ValueError(
            f"run_eval_rag only supports late/cross_attention (got {rag_mode})"
        )

    retriever = init_retriever(
        rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
        rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
        debug=debug,
    )

    df = load_split(split)
    ctx = build_rag_contexts(
        df, context_cfg, rag_cfg, retriever, debug=debug, collect_kb=True
    )
    labels = df[label_names].to_numpy(dtype=int)

    model, tokenizers = build_rag_model(config, num_labels=num_labels)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if bool(training_cfg.get("force_fp32", True)):
        model = model.float().to(device)

    batch_size = int(training_cfg.get("batch_size", 8))
    max_length = int(training_cfg.get("max_length", 1024))
    threshold = float(training_cfg.get("pred_threshold", 0.5))

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
    all_probs: list[np.ndarray] = []

    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pred_path.open("w", encoding="utf-8") as handle:
        for start, batch in enumerate(dataloader):
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = _get_logits(outputs)
                probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_labels.append(labels_batch.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

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


def train_and_eval_rag(config: dict, run_name: str | None = None) -> dict[str, object]:
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

    train_df = load_split("train")
    val_df = load_split("val")

    train_ctx = build_rag_contexts(train_df, context_cfg, rag_cfg, retriever)
    val_ctx = build_rag_contexts(val_df, context_cfg, rag_cfg, retriever)

    train_labels = train_df[label_names].to_numpy(dtype=int)
    val_labels = val_df[label_names].to_numpy(dtype=int)

    model, tokenizers = build_rag_model(config, num_labels=num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if bool(training_cfg.get("force_fp32", True)):
        model = model.float().to(device)

    batch_size = int(training_cfg.get("batch_size", 8))
    max_length = int(training_cfg.get("max_length", 1024))
    num_epochs = int(training_cfg.get("num_epochs", 20))
    learning_rate = float(training_cfg.get("learning_rate", 1e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.15))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    patience = int(training_cfg.get("early_stopping_patience", 3))
    threshold = float(training_cfg.get("pred_threshold", 0.5))

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
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    results_dir = Path(config.get("results_dir", "results"))
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_name}.pt"

    best_macro = float("-inf")
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        step = 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = _get_logits(outputs)
            loss = loss_fn(logits, labels)
            loss = loss / grad_accum_steps
            loss.backward()
            step += 1

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += float(loss.item()) * grad_accum_steps

        metrics = _evaluate_rag(
            model, val_loader, device, label_names=label_names, threshold=threshold
        )
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        LOGGER.info(
            "Epoch %d/%d loss=%.4f macro_f1=%.4f",
            epoch + 1,
            num_epochs,
            total_loss / max(1, len(train_loader)),
            macro_f1,
        )

        if macro_f1 > best_macro:
            best_macro = macro_f1
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info(
                    "Early stopping after %d epochs without improvement", patience
                )
                break

    return {"macro_f1": best_macro, "checkpoint": str(best_path)}
