"""Train a single late-fusion or cross-attention RAG config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from value_context_rag.models.rag_training import run_eval_rag, train_and_eval_rag
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging
from value_context_rag.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1] if model_name else "deberta"
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", base).strip("-").lower()
    return slug or "deberta"


def _unique_log_path(path: Path) -> Path:
    """Return a non-existing log file path to avoid overwriting prior runs."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix or ".log"
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RAG architecture for one config."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Optional run name.")
    parser.add_argument(
        "--eval",
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit for number of samples (for quick runs).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use a temporary results directory and avoid persisting outputs.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume training from.",
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
    config["debug"] = bool(args.debug)
    config["seed"] = int(args.seed)
    set_seed(int(args.seed), debug=args.debug)
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    model_slug = _model_slug(model_name)
    context_type = context_cfg.get("type", "sentence")
    use_rag = bool(rag_cfg.get("enabled", False))
    rag_mode = rag_cfg.get("mode", "none") if use_rag else "none"
    mode_tag = f"{rag_mode}_rag" if use_rag else "no_rag"

    results_dir = Path(config.get("results_dir", "results"))
    if args.dry_run:
        results_dir = Path(".tmp/value-context-rag-smoke")
        config["results_dir"] = str(results_dir)
        config["save_checkpoints"] = False

    run_name = args.run_name or f"rag_{rag_mode}_{context_type}_{model_slug}"
    config["eval"] = True if args.eval else config.get("eval", False)

    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = (
        f"deberta_{context_type}_{mode_tag}_seed{args.seed}_{model_slug}"
    )
    base_log_file = log_dir / f"{artifact_prefix}.log"
    log_file = _unique_log_path(base_log_file)

    logger = get_logger(__name__, log_file=str(log_file), overwrite=True)
    get_logger("value_context_rag.models.rag_training", log_file=str(log_file))
    get_logger("value_context_rag.data.dataset", log_file=str(log_file))
    get_logger("value_context_rag.models.rag_factory", log_file=str(log_file))

    logger.info("=" * 80)
    logger.info("Starting RAG architecture training with config %s", args.config)
    logger.debug(
        "Run name=%s seed=%d context=%s rag_mode=%s dry_run=%s",
        run_name,
        int(args.seed),
        context_type,
        rag_mode,
        args.dry_run,
    )
    logger.info(
        "Run: model=deberta variant=%s context=%s rag_mode=%s seed=%d eval=%s dry_run=%s",
        model_name,
        context_type,
        rag_mode,
        int(args.seed),
        args.eval,
        args.dry_run,
    )

    if args.dry_run:
        resume_path = None
    elif args.resume:
        resume_path = Path(args.resume)
    else:
        auto_path = results_dir / "checkpoints" / f"{run_name}_last.pt"
        resume_path = auto_path if auto_path.exists() else None

    best = train_and_eval_rag(config, run_name=run_name, resume_path=resume_path)
    logger.info("Best macro_f1=%.4f", float(best.get("macro_f1", 0.0)))

    if args.eval:
        if not config.get("save_checkpoints", True):
            logger.warning(
                "Skipping eval: checkpoints disabled (dry_run=%s)", args.dry_run
            )
            logger.info("=" * 80)
            return
        ckpt_path = results_dir / "checkpoints" / f"{run_name}.pt"
        pred_path = (
            results_dir
            / "predictions"
            / f"{artifact_prefix}.jsonl"
        )
        metrics_path = (
            results_dir
            / "logs"
            / f"{artifact_prefix}_test_metrics.json"
        )
        metrics = run_eval_rag(
            config,
            checkpoint_path=ckpt_path,
            split="test",
            output_pred_path=pred_path,
            output_metrics_path=metrics_path,
            debug=args.debug,
        )
        logger.info(
            "Test metrics - macro_f1=%.4f micro_f1=%.4f",
            metrics.get("macro_f1", 0.0),
            metrics.get("micro_f1", 0.0),
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
