"""Robustness evaluation utilities for KB noise experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from value_context_rag.data.context import (
    build_doc_context,
    build_sentence_context,
    build_window_context,
)
from value_context_rag.data.kb_noise import (
    drop_top_chunk,
    inject_offtopic_noise,
    limit_k,
)
from value_context_rag.eval.metrics import compute_f1_metrics
from value_context_rag.models.rag_training import build_rag_dataloader
from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _build_doc_contexts(df, context_cfg: dict, debug: bool = False) -> list[str]:
    context_type = context_cfg.get("type", "sentence")
    n_prev = int(context_cfg.get("n_prev", 2))
    n_next = int(context_cfg.get("n_next", 2))

    # Build doc index
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

    contexts: list[str] = []
    for row in df.to_dict(orient="records"):
        text_id = str(row["text_id"])
        sent_id = str(row["sent_id"])
        doc_sentences = docs[text_id]
        target_idx = index[(text_id, sent_id)]

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

        contexts.append(context)

    return contexts


def _apply_noise(
    chunks: list[dict],
    noise_config: dict,
    global_kb: list[dict],
) -> list[dict]:
    noise_type = noise_config.get("type", "none")
    if noise_type in {"none", None}:
        return list(chunks)
    if noise_type == "drop_top":
        drop_prob = float(noise_config.get("drop_prob", 0.0))
        return drop_top_chunk(chunks, drop_prob)
    if noise_type == "inject_noise":
        noise_ratio = float(noise_config.get("noise_ratio", 0.0))
        return inject_offtopic_noise(chunks, global_kb, noise_ratio)
    if noise_type == "limit_k":
        k = int(noise_config.get("k", 0))
        return limit_k(chunks, k)
    raise ValueError(f"Unknown noise type: {noise_type}")


def run_robustness_experiment(
    model,
    tokenizers,
    df,
    label_names: list[str],
    retriever,
    noise_config: dict,
    config: dict,
    *,
    output_pred_path=None,
) -> dict[str, float]:
    """Run a robustness experiment under a KB noise condition."""
    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    training_cfg = config.get("training", {})

    use_rag = bool(rag_cfg.get("enabled", False))
    rag_mode = rag_cfg.get("mode", "none") if use_rag else "none"
    top_k = int(rag_cfg.get("top_k", 5))
    batch_size = int(training_cfg.get("batch_size", 8))
    max_length = int(training_cfg.get("max_length", 1024))
    threshold = float(training_cfg.get("pred_threshold", 0.5))

    doc_texts = _build_doc_contexts(df, context_cfg, debug=False)
    kb_texts: list[list[str]] = []
    kb_info: list[list[dict]] = []

    global_kb = getattr(retriever, "chunks", []) if retriever is not None else []

    for doc_text in doc_texts:
        if use_rag and retriever is not None:
            chunks = retriever.retrieve(doc_text, top_k=top_k)
            noisy = _apply_noise(chunks, noise_config, global_kb)
            kb_list = [c.get("text", "") for c in noisy] if noisy else []
        else:
            noisy = []
            kb_list = []
        kb_texts.append(kb_list)
        kb_info.append(noisy)

    labels = df[label_names].to_numpy(dtype=int)
    dataloader = build_rag_dataloader(
        doc_texts,
        kb_texts,
        labels,
        tokenizers,
        batch_size,
        max_length,
        rag_mode,
        shuffle=False,
    )

    device = next(model.parameters()).device
    noise_type = noise_config.get("type", "none")
    noise_level = (
        noise_config.get("drop_prob")
        or noise_config.get("noise_ratio")
        or noise_config.get("k")
        or 0
    )

    model.eval()
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    pred_handle = None
    if output_pred_path is not None:
        output_path = Path(output_pred_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pred_handle = output_path.open("w", encoding="utf-8")

    try:
        with torch.no_grad():
            for start, batch in enumerate(dataloader):
                labels_batch = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = (
                    outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                )
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).long()
                all_labels.append(labels_batch.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

                if pred_handle is not None:
                    batch_preds = preds.cpu().numpy()
                    batch_labels = labels_batch.cpu().numpy()
                    for idx, pred_vec in enumerate(batch_preds):
                        row_idx = start * batch_size + idx
                        if row_idx >= len(df):
                            continue
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
                        "noise_type": noise_config.get("type", "none"),
                        "noise_level": noise_level,
                    }
                    pred_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if pred_handle is not None:
            pred_handle.close()

    y_true = np.vstack(all_labels) if all_labels else np.zeros((0, 0))
    y_pred = np.vstack(all_preds) if all_preds else np.zeros((0, 0))
    metrics = compute_f1_metrics(y_true, y_pred, label_names=label_names)

    metrics["noise_type"] = noise_type
    metrics["noise_level"] = float(noise_level) if noise_level is not None else 0.0
    return metrics
