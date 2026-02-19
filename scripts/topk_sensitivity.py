"""Sensitivity check for RAG top_k on sentence baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from value_context_rag.models.training import run_eval, train_and_eval
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging
from value_context_rag.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="top_k sensitivity (sentence+RAG).")
    parser.add_argument(
        "--config",
        default="configs/deberta_sentence.yaml",
        help="Base config to use.",
    )
    parser.add_argument(
        "--top_k_values",
        default="1,2,3,4,5",
        help="Comma-separated top_k values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit samples for quick check.",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/topk_sensitivity.csv",
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
    set_seed(args.seed)

    top_k_values = [int(x) for x in args.top_k_values.split(",") if x.strip()]
    results: list[dict[str, float]] = []

    for top_k in top_k_values:
        config = dict(base_config)
        config["context"] = dict(base_config.get("context", {}))
        config["rag"] = dict(base_config.get("rag", {}))
        config["training"] = dict(base_config.get("training", {}))
        config["seed"] = int(args.seed)

        config["context"]["type"] = "sentence"
        config["rag"]["enabled"] = True
        config["rag"]["top_k"] = int(top_k)

        if args.max_samples is not None:
            config["max_samples"] = args.max_samples

        run_name = f"sens_topk_{top_k}_seed{args.seed}"
        LOGGER.info("Running top_k=%d", top_k)
        attempts = max(args.retry_collapsed, 0) + 1
        best_macro = float("-inf")
        collapsed = True
        for attempt in range(attempts):
            if attempt > 0:
                LOGGER.warning(
                    "Retrying collapsed run (attempt %d/%d)", attempt + 1, attempts
                )
            config["seed"] = int(args.seed) + attempt
            best_macro, collapsed = train_and_eval(
                config, run_name=run_name, resume_path=None
            )
            if not collapsed:
                break

        results_dir = Path(config.get("results_dir", "results"))
        ckpt_path = results_dir / "checkpoints" / f"{run_name}.pt"
        pred_path = results_dir / "predictions" / f"{run_name}_test.jsonl"
        metrics_path = results_dir / "logs" / f"{run_name}_test_metrics.json"
        metrics = run_eval(
            config,
            checkpoint_path=ckpt_path,
            split="test",
            output_pred_path=pred_path,
            output_metrics_path=metrics_path,
            debug=args.debug,
        )

        results.append(
            {
                "top_k": float(top_k),
                "best_val_macro_f1": float(best_macro),
                "test_macro_f1": float(metrics.get("macro_f1", 0.0)),
                "test_micro_f1": float(metrics.get("micro_f1", 0.0)),
                "collapsed": bool(collapsed),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "top_k",
                "best_val_macro_f1",
                "test_macro_f1",
                "test_micro_f1",
                "collapsed",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    LOGGER.info("Saved top_k sensitivity results to %s", output_path)


if __name__ == "__main__":
    main()
