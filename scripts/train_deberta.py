"""Train and evaluate DeBERTa for one config."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from value_context_rag.data.dataset import get_label_names, load_split
from value_context_rag.kb.retriever import init_retriever
from value_context_rag.models.deberta import build_deberta_model
from value_context_rag.models.training import save_predictions_jsonl, train_and_eval
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger
from value_context_rag.utils.seed import set_seed

LOGGER = get_logger(__name__)


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    config = load_config(args.config)
    LOGGER.debug("Loaded config keys: %s", list(config.keys()))
    seed = int(args.seed)
    config["seed"] = seed
    set_seed(seed, debug=args.debug)
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    training_cfg = config.get("training", {})

    context_type = context_cfg.get("type", "sentence")
    n_prev = int(context_cfg.get("n_prev", 2))
    n_next = int(context_cfg.get("n_next", 2))
    use_rag = bool(rag_cfg.get("enabled", False))
    top_k = int(rag_cfg.get("top_k", 5))
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
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rag_suffix = "rag" if use_rag else "no_rag"
    log_file = log_dir / f"deberta_{context_type}_{rag_suffix}_seed{seed}.log"

    logger = get_logger(__name__, log_file=str(log_file))
    logger.info("Starting DeBERTa training with config %s", args.config)
    logger.debug("Run name seed=%d context=%s rag=%s", seed, context_type, use_rag)

    config["eval"] = True if args.eval else config.get("eval", False)
    run_name = f"deberta_{context_type}_{rag_suffix}_seed{seed}_best"
    LOGGER.debug("Checkpoint run name: %s", run_name)
    if args.resume:
        resume_path = Path(args.resume)
    else:
        auto_path = results_dir / "checkpoints" / f"{run_name}_last.pt"
        resume_path = auto_path if auto_path.exists() else None
    train_and_eval(config, run_name=run_name, resume_path=resume_path)

    if args.eval or config.get("eval", False):
        label_names = get_label_names()
        model, tokenizer = build_deberta_model(num_labels=len(label_names))
        ckpt_path = results_dir / "checkpoints" / f"{run_name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {ckpt_path}")
        LOGGER.debug("Loading checkpoint from %s", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        retriever = (
            init_retriever(
                rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
                rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
                debug=args.debug,
            )
            if use_rag
            else None
        )

        test_df = load_split("test")
        if args.max_samples is not None:
            test_df = test_df.head(int(args.max_samples))
        predictions_dir = results_dir / "predictions"
        pred_path = (
            predictions_dir / f"deberta_{context_type}_{rag_suffix}_seed{seed}.jsonl"
        )

        save_predictions_jsonl(
            model,
            tokenizer,
            test_df,
            label_names,
            pred_path,
            context_type=context_type,
            n_prev=n_prev,
            n_next=n_next,
            use_rag=use_rag,
            top_k=top_k,
            retriever=retriever,
            max_length=int(training_cfg.get("max_length", 1024)),
            batch_size=int(training_cfg.get("batch_size", 16)),
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
