"""Evaluate Gemma predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from value_context_rag.data.dataset import get_label_names
from value_context_rag.eval.metrics import compute_f1_metrics
from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gemma predictions.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSONL file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def load_predictions(
    path: Path, label_names: List[str]
) -> tuple[np.ndarray, np.ndarray]:
    gold = []
    pred = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            gold_labels = set(record.get("gold_labels", []))
            pred_labels = set(record.get("pred_labels", []))
            gold.append([1 if name in gold_labels else 0 for name in label_names])
            pred.append([1 if name in pred_labels else 0 for name in label_names])

    return np.asarray(gold, dtype=int), np.asarray(pred, dtype=int)


def infer_metrics_path(pred_path: Path) -> Path:
    logs_dir = pred_path.parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    name = pred_path.stem + "_metrics.json"
    return logs_dir / name


def evaluate_predictions(pred_path: Path, *, debug: bool = False) -> dict:
    """Load predictions, compute metrics, log, and save metrics JSON."""
    if debug:
        LOGGER.setLevel("DEBUG")
    label_names = get_label_names()
    gold, pred = load_predictions(pred_path, label_names)
    metrics = compute_f1_metrics(gold, pred, label_names=label_names)

    LOGGER.info("Macro F1: %.4f", metrics["macro_f1"])
    LOGGER.info("Micro F1: %.4f", metrics["micro_f1"])

    metrics_path = infer_metrics_path(pred_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved metrics to %s", metrics_path)
    return metrics


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    evaluate_predictions(pred_path, debug=args.debug)


if __name__ == "__main__":
    main()
