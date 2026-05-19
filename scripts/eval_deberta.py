"""Evaluate a trained DeBERTa checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from value_context_rag.models.training import run_eval
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1] if model_name else "deberta"
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", base).strip("-").lower()
    return slug or "deberta"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DeBERTa checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint. If omitted, inferred from config/context/rag/seed.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--tune_threshold",
        action="store_true",
        help="Sweep thresholds on the split to maximize macro-F1.",
    )
    parser.add_argument(
        "--threshold_start",
        type=float,
        default=0.0,
        help="Threshold sweep start.",
    )
    parser.add_argument(
        "--threshold_stop",
        type=float,
        default=1.0,
        help="Threshold sweep stop.",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.01,
        help="Threshold sweep step.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    silence_transformers_logging()

    config = load_config(args.config)
    LOGGER.debug("Loaded config keys: %s", list(config.keys()))
    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    model_slug = _model_slug(model_name)
    seed = int(config.get("seed", 42))

    context_type = context_cfg.get("type", "sentence")
    use_rag = bool(rag_cfg.get("enabled", False))
    rag_suffix = "rag" if use_rag else "no_rag"
    LOGGER.debug(
        "Eval config: context=%s rag=%s seed=%d split=%s",
        context_type,
        use_rag,
        seed,
        args.split,
    )

    results_dir = Path(config.get("results_dir", "results"))
    artifact_prefix = (
        f"deberta_{context_type}_{rag_suffix}_seed{seed}_{model_slug}"
    )
    run_name = f"{artifact_prefix}_best"
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else results_dir / "checkpoints" / f"{run_name}.pt"
    )
    LOGGER.debug("Using checkpoint path: %s", ckpt_path)

    predictions_dir = results_dir / "predictions"
    logs_dir = results_dir / "logs"
    pred_path = (
        predictions_dir
        / f"{artifact_prefix}_{args.split}.jsonl"
    )
    metrics_path = (
        logs_dir
        / f"{artifact_prefix}_{args.split}_metrics.json"
    )

    LOGGER.info("=" * 80)
    LOGGER.info(
        "Run: eval model=deberta variant=%s context=%s rag=%s seed=%d split=%s",
        model_name,
        context_type,
        use_rag,
        seed,
        args.split,
    )
    LOGGER.info("Evaluating checkpoint %s on %s split", ckpt_path, args.split)
    run_eval(
        config,
        checkpoint_path=ckpt_path,
        split=args.split,
        output_pred_path=pred_path,
        output_metrics_path=metrics_path,
        debug=args.debug,
        tune_threshold=args.tune_threshold,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
    )
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
