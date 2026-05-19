"""Train and evaluate DeBERTa for one config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from value_context_rag.data.dataset import get_label_names, load_split
from value_context_rag.models.deberta import build_deberta_model
from value_context_rag.models.rag_training import run_eval_rag, train_and_eval_rag
from value_context_rag.models.training import run_eval, train_and_eval
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging
from value_context_rag.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1] if model_name else "deberta"
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", base).strip("-").lower()
    return slug or "deberta"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeBERTa for one config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate and save predictions on test split after training.",
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
    parser.add_argument(
        "--retry_collapsed",
        type=int,
        default=1,
        help="Retries for collapsed runs.",
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
    LOGGER.debug("Loaded config keys: %s", list(config.keys()))
    seed = int(args.seed)
    config["seed"] = seed
    set_seed(seed, debug=args.debug)
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    model_slug = _model_slug(model_name)
    context_type = context_cfg.get("type", "sentence")
    use_rag = bool(rag_cfg.get("enabled", False))
    top_k = int(rag_cfg.get("top_k", 5))
    rag_mode = rag_cfg.get("mode", "none") if use_rag else "none"
    if rag_mode in {"late", "cross_attention"}:
        run_tag = f"{rag_mode}_rag"
    elif use_rag:
        run_tag = "rag"
    else:
        run_tag = "no_rag"
    LOGGER.debug(
        "Train config: context=%s rag=%s top_k=%d seed=%d",
        context_type,
        use_rag,
        top_k,
        seed,
    )

    results_dir = Path(config.get("results_dir", "results"))
    if args.dry_run:
        results_dir = Path(".tmp/value-context-rag-smoke")
        config["results_dir"] = str(results_dir)
        config["save_checkpoints"] = False
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = (
        f"deberta_{context_type}_{run_tag}_seed{seed}_{model_slug}"
    )
    log_file = log_dir / f"{artifact_prefix}.log"

    logger = get_logger(__name__, log_file=str(log_file), overwrite=True)
    # Attach the same log file to core module loggers so their INFO appears in one file.
    get_logger("value_context_rag.models.training", log_file=str(log_file))
    get_logger("value_context_rag.data.dataset", log_file=str(log_file))
    get_logger("value_context_rag.models.deberta", log_file=str(log_file))
    logger.info("=" * 80)
    logger.info("Starting DeBERTa training with config %s", args.config)
    logger.debug("Run name seed=%d context=%s rag=%s", seed, context_type, use_rag)
    logger.info(
        "Run: model=deberta variant=%s context=%s rag=%s seed=%d eval=%s dry_run=%s",
        model_name,
        context_type,
        use_rag,
        seed,
        args.eval,
        args.dry_run,
    )

    config["eval"] = True if args.eval else config.get("eval", False)
    run_name = f"{artifact_prefix}_best"
    LOGGER.debug("Checkpoint run name: %s", run_name)
    if args.dry_run:
        resume_path = None
    elif args.resume:
        resume_path = Path(args.resume)
    else:
        auto_path = results_dir / "checkpoints" / f"{run_name}_last.pt"
        resume_path = auto_path if auto_path.exists() else None
    attempts = max(args.retry_collapsed, 0) + 1
    best_macro = float("-inf")
    collapsed = True
    for attempt in range(attempts):
        if attempt > 0:
            logger.warning(
                "Retrying collapsed run (attempt %d/%d)", attempt + 1, attempts
            )
        config["seed"] = seed + attempt
        if rag_mode in {"late", "cross_attention"}:
            best = train_and_eval_rag(config, run_name=run_name)
            best_macro = float(best.get("macro_f1", float("-inf")))
            collapsed = False
        else:
            best_macro, collapsed = train_and_eval(
                config,
                run_name=run_name,
                resume_path=resume_path if attempt == 0 else None,
            )
        if not collapsed:
            break

    if rag_mode in {"late", "cross_attention"}:
        if args.eval or config.get("eval", False):
            if not config.get("save_checkpoints", True):
                logger.warning(
                    "Skipping eval: checkpoints disabled (dry_run=%s)", args.dry_run
                )
                logger.info("=" * 80)
                return
            predictions_dir = results_dir / "predictions"
            pred_path = (
                predictions_dir
                / f"{artifact_prefix}.jsonl"
            )
            metrics_path = (
                results_dir
                / "logs"
                / f"{artifact_prefix}_test_metrics.json"
            )
            ckpt_path = results_dir / "checkpoints" / f"{run_name}.pt"
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
        return
    if args.eval or config.get("eval", False):
        if not config.get("save_checkpoints", True):
            logger.warning(
                "Skipping eval: checkpoints disabled (dry_run=%s)", args.dry_run
            )
            logger.info("=" * 80)
            return
        label_names = get_label_names()
        model, tokenizer = build_deberta_model(
            num_labels=len(label_names),
            model_name=model_name,
            label_names=label_names,
        )
        ckpt_path = results_dir / "checkpoints" / f"{run_name}.pt"
        if not ckpt_path.exists():
            if args.dry_run:
                logger.warning(
                    "Skipping eval: best checkpoint not found at %s", ckpt_path
                )
                logger.info("=" * 80)
                return
            raise FileNotFoundError(f"Best checkpoint not found at {ckpt_path}")
        LOGGER.debug("Loading checkpoint from %s", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        test_df = load_split("test")
        if args.max_samples is not None:
            test_df = test_df.head(int(args.max_samples))
        predictions_dir = results_dir / "predictions"
        pred_path = (
            predictions_dir / f"{artifact_prefix}.jsonl"
        )

        metrics_path = (
            results_dir
            / "logs"
            / f"{artifact_prefix}_test_metrics.json"
        )
        metrics = run_eval(
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
