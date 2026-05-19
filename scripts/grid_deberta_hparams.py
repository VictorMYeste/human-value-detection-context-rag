"""Grid search for DeBERTa hyperparameters (sentence, no RAG)."""

from __future__ import annotations

import argparse
import csv
import os
from itertools import product
from pathlib import Path

from value_context_rag.models.training import train_and_eval
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)
CSV_FIELDS = [
    "weight_decay",
    "batch_size",
    "max_length",
    "learning_rate",
    "best_macro_f1",
    "collapsed",
]


def _grid_key(*, lr: float, wd: float, batch: int, max_len: int) -> tuple[str, str, int, int]:
    """Stable key for one grid configuration."""
    return (f"{float(lr):.12g}", f"{float(wd):.12g}", int(batch), int(max_len))


def _completed_keys_from_csv(path: Path) -> set[tuple[str, str, int, int]]:
    """Load already completed grid points from existing CSV output."""
    if not path.exists() or path.stat().st_size == 0:
        return set()

    completed: set[tuple[str, str, int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                key = _grid_key(
                    lr=float(row["learning_rate"]),
                    wd=float(row["weight_decay"]),
                    batch=int(float(row["batch_size"])),
                    max_len=int(float(row["max_length"])),
                )
            except Exception:
                LOGGER.warning("Skipping malformed row while resuming: %s", row)
                continue
            completed.add(key)
    return completed


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

    weight_decays = [0.1, 0.12, 0.15, 0.18, 0.2]
    batch_sizes = [16]
    # max_lengths = [512, 1024]
    max_lengths = [1024]
    learning_rates = [6e-6, 7e-6, 8e-6, 9e-6, 1e-5]
    base_grad_accum_steps = int(
        base_config.get("training", {}).get("grad_accum_steps", 2)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_keys = _completed_keys_from_csv(output_path)
    if completed_keys:
        LOGGER.info(
            "Resume mode: found %d completed configurations in %s",
            len(completed_keys),
            output_path,
        )

    grid = list(product(learning_rates, weight_decays, batch_sizes, max_lengths))
    LOGGER.info("Running grid with %d configurations", len(grid))

    rows_appended = 0
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if output_path.stat().st_size == 0:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

        for idx, (lr, wd, batch, max_len) in enumerate(grid, start=1):
            key = _grid_key(lr=lr, wd=wd, batch=batch, max_len=max_len)
            if key in completed_keys:
                LOGGER.info(
                    "[%d/%d] SKIP (already done) lr=%.1e wd=%.2f batch=%d max_len=%d",
                    idx,
                    len(grid),
                    lr,
                    wd,
                    batch,
                    max_len,
                )
                continue

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
            config["training"]["grad_accum_steps"] = base_grad_accum_steps
            config["training"]["num_epochs"] = int(
                config["training"].get("num_epochs", 10)
            )
            config["training"]["early_stopping_patience"] = int(
                config["training"].get("early_stopping_patience", 3)
            )

            if args.max_samples is not None:
                config["max_samples"] = args.max_samples

            run_name = f"grid_deberta_sentence_lr{lr}_wd{wd}_b{batch}_ml{max_len}"
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
                    LOGGER.warning(
                        "Retrying collapsed run (attempt %d/%d)", attempt + 1, attempts
                    )
                config["seed"] = int(base_config.get("seed", 42)) + attempt
                best_macro_f1, collapsed = train_and_eval(
                    config, run_name=run_name, resume_path=None
                )
                if not collapsed:
                    break

            row = {
                "weight_decay": float(wd),
                "batch_size": float(batch),
                "max_length": float(max_len),
                "learning_rate": float(lr),
                "best_macro_f1": float(best_macro_f1),
                "collapsed": bool(collapsed),
            }
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            completed_keys.add(key)
            rows_appended += 1

    LOGGER.info("Saved grid results to %s", output_path)
    LOGGER.info("Appended %d new rows in this execution", rows_appended)


if __name__ == "__main__":
    main()
