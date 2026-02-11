"""Evaluation metrics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


def binarize_probs(probs: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """Binarize probability outputs using a fixed threshold."""
    LOGGER.debug("Binarizing probabilities with threshold=%.3f", threshold)
    return (probs >= threshold).astype(int)


def compute_global_metrics(gold: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute global micro/macro precision, recall, and F1."""
    gold = gold.astype(int)
    pred = pred.astype(int)
    LOGGER.debug("Computing global metrics (samples=%d, labels=%d)", *gold.shape)

    if gold.size == 0:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    tp = (gold & pred).sum()
    fp = ((1 - gold) & pred).sum()
    fn = (gold & (1 - pred)).sum()

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for col in range(gold.shape[1]):
        tp_c = (gold[:, col] & pred[:, col]).sum()
        fp_c = ((1 - gold[:, col]) & pred[:, col]).sum()
        fn_c = (gold[:, col] & (1 - pred[:, col])).sum()
        precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = (
            2 * precision_c * recall_c / (precision_c + recall_c)
            if (precision_c + recall_c) > 0
            else 0.0
        )
        precisions.append(float(precision_c))
        recalls.append(float(recall_c))
        f1s.append(float(f1_c))

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def compute_per_label_f1(
    gold: np.ndarray,
    pred: np.ndarray,
    *,
    label_names: List[str],
) -> Dict[str, float]:
    """Compute per-label F1 scores."""
    gold = gold.astype(int)
    pred = pred.astype(int)
    LOGGER.debug("Computing per-label F1 for %d labels", len(label_names))
    if gold.size == 0:
        return {name: 0.0 for name in label_names}

    per_label_f1: Dict[str, float] = {}
    for col, name in enumerate(label_names):
        tp_c = (gold[:, col] & pred[:, col]).sum()
        fp_c = ((1 - gold[:, col]) & pred[:, col]).sum()
        fn_c = (gold[:, col] & (1 - pred[:, col])).sum()
        precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = (
            2 * precision_c * recall_c / (precision_c + recall_c)
            if (precision_c + recall_c) > 0
            else 0.0
        )
        per_label_f1[name] = float(f1_c)
    return per_label_f1


def macro_f1_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-F1 only, for efficient paired tests."""
    metrics = compute_global_metrics(y_true, y_pred)
    return float(metrics["macro_f1"])


def compute_f1_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: List[str],
) -> Dict[str, object]:
    """Compute micro/macro F1 and per-label F1."""
    global_metrics = compute_global_metrics(y_true, y_pred)
    per_label_f1 = compute_per_label_f1(y_true, y_pred, label_names=label_names)
    macro_f1 = global_metrics["macro_f1"]
    micro_f1 = global_metrics["micro_f1"]
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "per_label_f1": per_label_f1,
    }
