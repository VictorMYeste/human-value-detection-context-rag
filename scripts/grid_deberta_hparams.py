"""Grid search for DeBERTa hyperparameters (sentence, no RAG)."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

from value_context_rag.models.training import train_and_eval
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for DeBERTa hparams.")
    parser.add_argument(
        "--config",
        default="configs/deberta_sentence.yaml",
        help="Base config to use.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit samples for quick search.",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/grid_deberta_hparams.csv",
        help="CSV path to store results.",
    )
    parser.add_argument(
        "--retry_collapsed",
        type=int,
        default=1,
        help="Retries for collapsed runs.",
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

    silence_transformers_logging()

    base_config = load_config(args.config)

    weight_decays = [0.01, 0.1, 0.15]
    batch_sizes = [8, 16]
    # max_lengths = [512, 1024]
    max_lengths = [1024]
    learning_rates = [1e-5, 2e-5, 3e-5]

    results: list[dict[str, float]] = []
    grid = list(product(learning_rates, weight_decays, batch_sizes, max_lengths))
    LOGGER.info("Running grid with %d configurations", len(grid))

    for idx, (lr, wd, batch, max_len) in enumerate(grid, start=1):
        config = dict(base_config)
        config["context"] = dict(base_config.get("context", {}))
        config["rag"] = dict(base_config.get("rag", {}))
        config["training"] = dict(base_config.get("training", {}))

        config["context"]["type"] = "sentence"
        config["rag"]["enabled"] = False

        config["training"]["learning_rate"] = float(lr)
        config["training"]["weight_decay"] = float(wd)
        config["training"]["batch_size"] = int(batch)
        config["training"]["max_length"] = int(max_len)
        config["training"]["grad_accum_steps"] = 1
        config["training"]["num_epochs"] = int(
            config["training"].get("num_epochs", 10)
        )
        config["training"]["early_stopping_patience"] = int(
            config["training"].get("early_stopping_patience", 3)
        )

        if args.max_samples is not None:
            config["max_samples"] = args.max_samples

        run_name = (
            f"grid_deberta_sentence_lr{lr}_wd{wd}_b{batch}_ml{max_len}"
        )
        LOGGER.info(
            "[%d/%d] lr=%.1e wd=%.2f batch=%d max_len=%d",
            idx,
            len(grid),
            lr,
            wd,
            batch,
            max_len,
        )
        attempts = max(args.retry_collapsed, 0) + 1
        best_macro_f1 = float("-inf")
        collapsed = True
        for attempt in range(attempts):
            if attempt > 0:
                LOGGER.warning("Retrying collapsed run (attempt %d/%d)", attempt + 1, attempts)
            config["seed"] = int(base_config.get("seed", 42)) + attempt
            best_macro_f1, collapsed = train_and_eval(config, run_name=run_name, resume_path=None)
            if not collapsed:
                break
        results.append(
            {
                "weight_decay": float(wd),
                "batch_size": float(batch),
                "max_length": float(max_len),
                "learning_rate": float(lr),
                "best_macro_f1": float(best_macro_f1),
                "collapsed": bool(collapsed),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "weight_decay",
                "batch_size",
                "max_length",
                "learning_rate",
                "best_macro_f1",
                "collapsed",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    LOGGER.info("Saved grid results to %s", output_path)


if __name__ == "__main__":
    main()
