"""Estimate KB token usage for a given top_k and KB token budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from value_context_rag.data.dataset import get_label_names, load_split
from value_context_rag.kb.retriever import init_retriever
from value_context_rag.models.deberta import build_deberta_model
from value_context_rag.models.training import build_contexts
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate KB token budget usage.")
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
        "--top_k",
        type=int,
        default=None,
        help="Override top_k (default: from config).",
    )
    parser.add_argument(
        "--kb_budget",
        type=int,
        default=None,
        help="Override KB token budget (default: from config).",
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


def _kb_length(tokenizer, chunks: list[dict]) -> int:
    if not chunks:
        return 0
    kb_text = "\n\n".join(chunk["text"] for chunk in chunks)
    return len(tokenizer.encode(kb_text, add_special_tokens=False))


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
    max_length = int(training_cfg.get("max_length", 1024))

    top_k = int(args.top_k if args.top_k is not None else rag_cfg.get("top_k", 5))
    kb_budget = (
        int(args.kb_budget)
        if args.kb_budget is not None
        else rag_cfg.get("kb_max_tokens", 200)
    )
    doc_budget = context_cfg.get("doc_max_tokens")
    if doc_budget is None and kb_budget is not None:
        doc_budget = max_length - int(kb_budget)

    splits = ["training", "validation", "test"]
    if args.split != "all":
        splits = [args.split]

    dfs = {}
    for split in splits:
        df = load_split(split)
        if args.max_samples is not None:
            df = df.head(int(args.max_samples))
        dfs[split] = df

    total_samples = sum(len(df) for df in dfs.values())
    LOGGER.info(
        "Estimating KB budget (split=%s samples=%d top_k=%d kb_budget=%s)",
        args.split,
        total_samples,
        top_k,
        kb_budget,
    )

    label_names = get_label_names()
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    _, tokenizer = build_deberta_model(
        num_labels=len(label_names),
        model_name=model_name,
        label_names=label_names,
    )

    retriever = init_retriever(
        rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
        rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
        debug=args.debug,
    )

    stats_by_split: dict[str, dict] = {}
    all_lengths: list[int] = []

    for split, df in dfs.items():
        contexts = build_contexts(
            df,
            context_type=context_type,
            n_prev=n_prev,
            n_next=n_next,
            use_rag=False,
            top_k=top_k,
            retriever=None,
            debug=args.debug,
            tokenizer=tokenizer,
            doc_budget_tokens=doc_budget if context_type == "doc" else None,
            kb_budget_tokens=None,
        )
        lengths = []
        truncated = 0
        for context in contexts:
            chunks = retriever.retrieve(context, top_k=top_k)
            kb_len = _kb_length(tokenizer, chunks)
            lengths.append(kb_len)
            if kb_budget is not None and kb_len > kb_budget:
                truncated += 1

        lengths_arr = np.asarray(lengths, dtype=float)
        total = int(lengths_arr.size)
        rate = truncated / total if total else 0.0
        stats_by_split[split] = {
            "samples": total,
            "kb_truncated": truncated,
            "kb_truncated_rate": rate,
            "kb_length_mean": float(lengths_arr.mean()) if total else 0.0,
            "kb_length_p95": float(np.percentile(lengths_arr, 95)) if total else 0.0,
            "kb_length_p99": float(np.percentile(lengths_arr, 99)) if total else 0.0,
        }
        all_lengths.extend(lengths)
        LOGGER.info(
            "[%s] KB truncation %d/%d (%.2f%%)",
            split,
            truncated,
            total,
            rate * 100,
        )

    all_arr = np.asarray(all_lengths, dtype=float)
    total_all = int(all_arr.size)
    truncated_all = int(sum(s["kb_truncated"] for s in stats_by_split.values()))
    rate_all = truncated_all / total_all if total_all else 0.0
    stats = {
        "top_k": top_k,
        "kb_budget": kb_budget,
        "max_length": max_length,
        "doc_budget": doc_budget if context_type == "doc" else None,
        "samples": total_all,
        "kb_truncated": truncated_all,
        "kb_truncated_rate": rate_all,
        "kb_length_mean": float(all_arr.mean()) if total_all else 0.0,
        "kb_length_p95": float(np.percentile(all_arr, 95)) if total_all else 0.0,
        "kb_length_p99": float(np.percentile(all_arr, 99)) if total_all else 0.0,
        "by_split": stats_by_split,
    }

    LOGGER.info(
        "Total KB truncation %d/%d (%.2f%%)",
        truncated_all,
        total_all,
        rate_all * 100,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Saved stats to %s", output_path)


if __name__ == "__main__":
    main()
