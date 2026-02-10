"""Gemma inference and output parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from value_context_rag.data.context import (
    build_doc_context,
    build_sentence_context,
    build_window_context,
)
from value_context_rag.data.dataset import get_label_names, load_split
from value_context_rag.kb.retriever import init_retriever
from value_context_rag.llm.gemma_client import GemmaClient, GemmaConfig
from value_context_rag.llm.prompts import (
    build_prompt_doc,
    build_prompt_sentence,
    build_prompt_window,
)
from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_labels(raw_output: str, label_names: List[str]) -> List[str]:
    """Parse comma-separated labels, normalize, and filter unknowns."""
    if not raw_output:
        return []

    normalized = raw_output.strip()
    if normalized.upper().startswith("NONE"):
        LOGGER.debug("Model returned NONE")
        return []

    candidates = [c.strip() for c in normalized.split(",") if c.strip()]
    if not candidates:
        return []

    canonical = {name.lower(): name for name in label_names}
    parsed: List[str] = []
    for cand in candidates:
        key = cand.lower()
        if key in canonical and canonical[key] not in parsed:
            parsed.append(canonical[key])
        else:
            LOGGER.debug("Filtered unknown label: %s", cand)
    return parsed


def run_inference(config: Dict, split: str) -> None:
    """Run Gemma inference on a split and save predictions."""
    label_names = get_label_names()
    df = load_split(split)
    max_samples = config.get("max_samples")
    if max_samples is not None:
        df = df.head(int(max_samples))
        LOGGER.info("Limiting inference to %d samples", len(df))

    context_cfg = config.get("context", {})
    rag_cfg = config.get("rag", {})
    llm_cfg = config.get("llm", {})

    context_type = context_cfg.get("type", "sentence")
    n_prev = int(context_cfg.get("n_prev", 2))
    n_next = int(context_cfg.get("n_next", 2))
    use_rag = bool(rag_cfg.get("enabled", False))
    top_k = int(rag_cfg.get("top_k", 5))

    client = GemmaClient(
        GemmaConfig(
            model_name=config.get("model", {}).get("name", "google/gemma-3-12b-it"),
            device=llm_cfg.get("device"),
            quantization=llm_cfg.get("quantization", "8bit"),
            max_new_tokens=int(llm_cfg.get("max_tokens", 256)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            top_p=float(llm_cfg.get("top_p", 1.0)),
        )
    )

    token_stats = {
        "max_prompt_tokens": llm_cfg.get("max_prompt_tokens"),
        "prompt_over_limit": 0,
        "prompt_total": 0,
    }

    retriever = (
        init_retriever(
            rag_cfg.get("kb_path", "data/kb/kb_chunks.jsonl"),
            rag_cfg.get("index_path", "data/kb/kb_index.faiss"),
            debug=False,
        )
        if use_rag
        else None
    )

    results_dir = Path(config.get("results_dir", "results"))
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rag_suffix = "rag" if use_rag else "no_rag"
    output_path = predictions_dir / f"gemma_{context_type}_{rag_suffix}_{split}.jsonl"

    LOGGER.info("Running Gemma inference for %s split", split)
    LOGGER.debug(
        "Inference config: context=%s rag=%s top_k=%d output=%s",
        context_type,
        use_rag,
        top_k,
        output_path,
    )

    existing: Set[Tuple[str, str]] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                existing.add((str(record.get("text_id")), str(record.get("sent_id"))))
        if existing:
            LOGGER.info("Resuming inference: %d predictions already exist", len(existing))

    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            text_id = str(row["text_id"])
            sent_id = str(row["sent_id"])
            if (text_id, sent_id) in existing:
                continue
            target_text = str(row["text"])

            doc_sentences = df[df["text_id"] == text_id]["text"].tolist()
            target_idx = doc_sentences.index(target_text)

            if context_type == "sentence":
                context = build_sentence_context(doc_sentences, target_idx)
                prompt = build_prompt_sentence(context)
            elif context_type == "window":
                context = build_window_context(
                    doc_sentences,
                    target_idx,
                    n_prev=n_prev,
                    n_next=n_next,
                    marker_style="gemma",
                )
                prompt = build_prompt_window(context, target_text)
            elif context_type == "doc":
                context = build_doc_context(
                    doc_sentences,
                    target_idx,
                    marker_style="gemma",
                )
                prompt = build_prompt_doc(context, target_text)
            else:
                raise ValueError(f"Unknown context type: {context_type}")

            chunks: List[dict] = []
            if use_rag and retriever is not None:
                chunks = retriever.retrieve(target_text, top_k=top_k)
                snippets = [chunk["text"] for chunk in chunks]
                LOGGER.debug("Retrieved %d KB snippets for %s/%s", len(snippets), text_id, sent_id)
                LOGGER.debug(
                    "KB chunk ids=%s values=%s",
                    [c.get("id") for c in chunks],
                    [c.get("values", []) for c in chunks],
                )
                if context_type == "sentence":
                    prompt = build_prompt_sentence(target_text, kb_snippets=snippets)
                elif context_type == "window":
                    prompt = build_prompt_window(
                        context, target_text, kb_snippets=snippets
                    )
                else:
                    prompt = build_prompt_doc(context, target_text, kb_snippets=snippets)

            if token_stats["max_prompt_tokens"] is not None:
                prompt_len = len(client.tokenizer(prompt)["input_ids"])
                token_stats["prompt_total"] += 1
                if prompt_len >= int(token_stats["max_prompt_tokens"]):
                    token_stats["prompt_over_limit"] += 1
                LOGGER.debug(
                    "Prompt tokens=%d (limit=%s)",
                    prompt_len,
                    token_stats["max_prompt_tokens"],
                )

            raw = client.generate(
                prompt,
                max_tokens=int(llm_cfg.get("max_tokens", 256)),
                temperature=float(llm_cfg.get("temperature", 0.0)),
                top_p=float(llm_cfg.get("top_p", 1.0)),
            )
            pred_labels = parse_labels(raw, label_names)
            gold_labels = [
                name for name in label_names if int(row.get(name, 0)) == 1
            ]
            LOGGER.debug(
                "Parsed labels for %s/%s: pred=%d gold=%d",
                text_id,
                sent_id,
                len(pred_labels),
                len(gold_labels),
            )

            record = {
                "text_id": text_id,
                "sent_id": sent_id,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "kb_chunk_ids": [c.get("id") for c in chunks] if use_rag and retriever is not None else [],
                "kb_values": sorted({v for c in chunks for v in c.get("values", [])}) if use_rag and retriever is not None else [],
                "raw_output": raw,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Saved predictions to %s", output_path)

    stats_path = results_dir / "logs" / f"gemma_token_stats_{context_type}_{rag_suffix}_{split}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if token_stats["prompt_total"] > 0:
        token_stats["prompt_over_limit_rate"] = token_stats["prompt_over_limit"] / token_stats["prompt_total"]
    stats_path.write_text(
        json.dumps(token_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved token stats to %s", stats_path)
