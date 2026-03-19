"""Estimate truncation rate without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from value_context_rag.data.dataset import get_label_names, load_split
from value_context_rag.models.deberta import build_deberta_model, encode_batch
from value_context_rag.kb.retriever import init_retriever
from value_context_rag.models.training import build_contexts
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate truncation rate.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--split",
        choices=["training", "validation", "test", "all"],
        default="all",
        help="Dataset split to use (default: all).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Number of samples to inspect (default: full split).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON stats.",
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
    config = load_config(args.config)

    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    training_cfg = config.get("training", {})

    context_type = context_cfg.get("type", "sentence")
    n_prev = int(context_cfg.get("n_prev", 2))
    n_next = int(context_cfg.get("n_next", 2))
    use_rag = bool(rag_cfg.get("enabled", False))
    top_k = int(rag_cfg.get("top_k", 5))
    max_length = int(training_cfg.get("max_length", 1024))
    kb_budget_tokens = rag_cfg.get("kb_max_tokens")
    doc_budget_tokens = context_cfg.get("doc_max_tokens")
    if doc_budget_tokens is None and kb_budget_tokens is not None:
        doc_budget_tokens = max_length - int(kb_budget_tokens)

    splits = ["training", "validation", "test"]
    if args.split != "all":
        splits = [args.split]

    dfs = {}
    for split in splits:
        df = load_split(split)
        if args.max_samples is not None:
            df = df.head(int(args.max_samples))
        dfs[split] = df

    label_names = get_label_names()
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    _, tokenizer = build_deberta_model(
        num_labels=len(label_names),
        model_name=model_name,
        label_names=label_names,
    )

    total_samples = sum(len(df) for df in dfs.values())
    LOGGER.info(
        "Estimating truncation (split=%s samples=%d context=%s rag=%s top_k=%d)",
        args.split,
        total_samples,
        context_type,
        use_rag,
        top_k,
    )

    retriever = (
        init_retriever(
            rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
            rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
            debug=args.debug,
        )
        if use_rag
        else None
    )

    texts_by_split = {}
    for split, df in dfs.items():
        texts_by_split[split] = build_contexts(
            df,
            context_type=context_type,
            n_prev=n_prev,
            n_next=n_next,
            use_rag=use_rag,
            top_k=top_k,
            retriever=retriever,
            debug=args.debug,
            tokenizer=tokenizer,
            doc_budget_tokens=doc_budget_tokens if context_type == "doc" else None,
            kb_budget_tokens=kb_budget_tokens if use_rag else None,
        )

    stats_by_split = {}
    all_lengths = []
    for split, texts in texts_by_split.items():
        lengths = []
        truncated = 0
        total = 0
        for i in range(0, len(texts), 64):
            batch_texts = texts[i : i + 64]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                return_overflowing_tokens=True,
                return_length=True,
                padding=False,
            )
            mapping = encoded.get("overflow_to_sample_mapping", [])
            lengths_list = encoded.get("length", [])
            counts = [0 for _ in batch_texts]
            first_lengths = [0 for _ in batch_texts]
            for idx, sample_idx in enumerate(mapping):
                if 0 <= sample_idx < len(counts):
                    if counts[sample_idx] == 0:
                        first_lengths[sample_idx] = int(lengths_list[idx])
                    counts[sample_idx] += 1
            lengths.extend(first_lengths)
            truncated += sum(1 for c in counts if c > 1)
            total += len(counts)

        lengths_arr = np.asarray(lengths, dtype=float)
        rate = truncated / total if total else 0.0
        stats_by_split[split] = {
            "samples": total,
            "truncated": truncated,
            "truncated_rate": rate,
            "length_mean": float(lengths_arr.mean()) if total else 0.0,
            "length_p95": float(np.percentile(lengths_arr, 95)) if total else 0.0,
            "length_p99": float(np.percentile(lengths_arr, 99)) if total else 0.0,
        }
        all_lengths.extend(lengths)
        LOGGER.info(
            "[%s] Overflow-based truncation %d/%d (%.2f%%)",
            split,
            truncated,
            total,
            rate * 100,
        )

    all_arr = np.asarray(all_lengths, dtype=float)
    total_all = int(all_arr.size)
    truncated_all = int(sum(stats["truncated"] for stats in stats_by_split.values()))
    rate_all = truncated_all / total_all if total_all else 0.0
    stats = {
        "max_length": max_length,
        "samples": total_all,
        "truncated": truncated_all,
        "truncated_rate": rate_all,
        "length_mean": float(all_arr.mean()) if total_all else 0.0,
        "length_p95": float(np.percentile(all_arr, 95)) if total_all else 0.0,
        "length_p99": float(np.percentile(all_arr, 99)) if total_all else 0.0,
        "by_split": stats_by_split,
        "truncation_method": "overflow_to_sample_mapping",
    }

    LOGGER.info(
        "Total overflow-based truncation %d/%d (%.2f%%)",
        truncated_all,
        total_all,
        rate_all * 100,
    )
    LOGGER.info(
        "Total length mean=%.1f p95=%.1f p99=%.1f",
        stats["length_mean"],
        stats["length_p95"],
        stats["length_p99"],
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Saved stats to %s", output_path)


if __name__ == "__main__":
    main()
