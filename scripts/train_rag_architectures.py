"""Train a single RAG architecture config."""

from __future__ import annotations

import argparse
from pathlib import Path

from value_context_rag.models.rag_training import run_eval_rag, train_and_eval_rag
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging
from value_context_rag.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RAG architecture for one config."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Optional run name.")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate on test split after training.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)
        LOGGER.setLevel(logging.DEBUG)

    silence_transformers_logging()

    config = load_config(args.config)
    config["seed"] = int(args.seed)
    set_seed(int(args.seed), debug=args.debug)

    run_name = args.run_name
    best = train_and_eval_rag(config, run_name=run_name)
    LOGGER.info("Best macro_f1=%.4f", float(best.get("macro_f1", 0.0)))

    if args.eval_test:
        results_dir = Path(config.get("results_dir", "results"))
        context_type = config.get("context", {}).get("type", "sentence")
        rag_mode = config.get("rag", {}).get("mode", "none")
        rag_suffix = "rag" if config.get("rag", {}).get("enabled", False) else "no_rag"
        if run_name is None:
            run_name = f"rag_{rag_mode}_{context_type}"
        ckpt_path = results_dir / "checkpoints" / f"{run_name}.pt"
        pred_path = (
            results_dir
            / "predictions"
            / f"deberta_{context_type}_{rag_suffix}_seed{args.seed}.jsonl"
        )
        metrics_path = (
            results_dir
            / "logs"
            / f"deberta_{context_type}_{rag_suffix}_seed{args.seed}_test_metrics.json"
        )
        metrics = run_eval_rag(
            config,
            checkpoint_path=ckpt_path,
            split="test",
            output_pred_path=pred_path,
            output_metrics_path=metrics_path,
            debug=args.debug,
        )
        LOGGER.info(
            "Test metrics - macro_f1=%.4f micro_f1=%.4f",
            metrics.get("macro_f1", 0.0),
            metrics.get("micro_f1", 0.0),
        )


if __name__ == "__main__":
    main()
