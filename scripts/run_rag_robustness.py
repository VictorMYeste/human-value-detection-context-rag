"""Run robustness experiments for a trained RAG model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from value_context_rag.data.dataset import get_label_names, load_split
from value_context_rag.eval.robustness import run_robustness_experiment
from value_context_rag.kb.retriever import init_retriever
from value_context_rag.models.rag_factory import build_rag_model
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG robustness experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--noise-type",
        required=True,
        choices=["drop_top", "inject_noise", "limit_k"],
        help="Noise type to apply.",
    )
    parser.add_argument(
        "--noise-level",
        required=True,
        type=float,
        help="Noise level (probability, ratio, or k).",
    )
    parser.add_argument(
        "--noise-levels",
        default=None,
        help="Comma-separated list of noise levels to sweep (overrides --noise-level).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path for metrics.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _build_noise_config(noise_type: str, noise_level: float) -> dict:
    if noise_type == "drop_top":
        return {"type": "drop_top", "drop_prob": float(noise_level)}
    if noise_type == "inject_noise":
        return {"type": "inject_noise", "noise_ratio": float(noise_level)}
    if noise_type == "limit_k":
        return {"type": "limit_k", "k": int(noise_level)}
    raise ValueError(f"Unsupported noise_type: {noise_type}")


def main() -> None:
    args = _parse_args()
    silence_transformers_logging()

    config = load_config(args.config)
    label_names = get_label_names()

    model, tokenizers = build_rag_model(config, num_labels=len(label_names))
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rag_cfg = config.get("rag", {})
    retriever = init_retriever(
        rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
        rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
        debug=args.debug,
    )

    results_dir = Path(config.get("results_dir", "results"))
    if args.output:
        output_path = Path(args.output)
    else:
        logs_dir = results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        level_str = str(args.noise_level).replace(".", "p")
        output_path = logs_dir / f"rag_robustness_{args.noise_type}_{level_str}.json"
    pred_path = output_path.with_suffix(".jsonl")

    df = load_split("test")
    if args.noise_levels:
        noise_levels = [
            float(x.strip()) for x in args.noise_levels.split(",") if x.strip()
        ]
    else:
        noise_levels = [float(args.noise_level)]

    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for level in noise_levels:
        noise_config = _build_noise_config(args.noise_type, level)
        level_str = str(level).replace(".", "p").replace("-", "m")
        if args.output and len(noise_levels) == 1:
            output_path = Path(args.output)
        elif args.output:
            base = Path(args.output)
            output_path = (
                base.parent / f"{base.stem}_{args.noise_type}_{level_str}{base.suffix}"
            )
        else:
            output_path = (
                logs_dir / f"rag_robustness_{args.noise_type}_{level_str}.json"
            )
        pred_path = output_path.with_suffix(".jsonl")

        metrics = run_robustness_experiment(
            model,
            tokenizers,
            df,
            label_names,
            retriever,
            noise_config,
            config,
            output_pred_path=pred_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        LOGGER.info("Saved robustness metrics to %s", output_path)
        LOGGER.info("Saved robustness predictions to %s", pred_path)
        all_metrics.append(metrics)

    if len(noise_levels) > 1:
        if args.output:
            sweep_path = Path(args.output)
        else:
            sweep_path = logs_dir / f"rag_robustness_{args.noise_type}_sweep.json"
        with sweep_path.open("w", encoding="utf-8") as handle:
            json.dump(all_metrics, handle, indent=2)
        LOGGER.info("Saved robustness sweep metrics to %s", sweep_path)


if __name__ == "__main__":
    main()
