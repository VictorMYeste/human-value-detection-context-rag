"""Evaluate a trained DeBERTa checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from value_context_rag.models.training import run_eval
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    config = load_config(args.config)
    LOGGER.debug("Loaded config keys: %s", list(config.keys()))
    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
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
    run_name = f"deberta_{context_type}_{rag_suffix}_seed{seed}_best"
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
        / f"deberta_{context_type}_{rag_suffix}_seed{seed}_{args.split}.jsonl"
    )
    metrics_path = (
        logs_dir
        / f"deberta_{context_type}_{rag_suffix}_seed{seed}_{args.split}_metrics.json"
    )

    LOGGER.info("=" * 80)
    LOGGER.info(
        "Run: eval model=deberta context=%s rag=%s seed=%d split=%s",
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
    )
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
