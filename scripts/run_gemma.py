"""Run Gemma zero-shot inference for one config."""

from __future__ import annotations

import argparse
from pathlib import Path

from value_context_rag.llm.inference import run_inference
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma inference.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--split", default="test", help="Split to run on.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit for number of samples (for quick runs).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate predictions after inference.",
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
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    LOGGER.info("Running Gemma with config %s on split %s", args.config, args.split)
    run_inference(config, split=args.split)

    if args.eval:
        from scripts.eval_gemma import evaluate_predictions

        context_type = config.get("context", {}).get("type", "sentence")
        use_rag = bool(config.get("rag", {}).get("enabled", False))
        rag_suffix = "rag" if use_rag else "no_rag"

        results_dir = config.get("results_dir", "results")
        pred_path = (
            Path(results_dir)
            / "predictions"
            / f"gemma_{context_type}_{rag_suffix}_{args.split}.jsonl"
        )
        evaluate_predictions(pred_path, debug=args.debug)


if __name__ == "__main__":
    main()
