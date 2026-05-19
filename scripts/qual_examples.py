"""Extract qualitative examples and optionally build final paper bundles."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from value_context_rag.data.dataset import load_split
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)

CONTEXT_NAMES = {"sentence", "window", "doc"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select qualitative examples with dataset + KB enrichment."
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline predictions JSONL (pairwise mode).",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to comparison predictions JSONL (pairwise mode).",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/qual_examples.jsonl",
        help="Output JSONL path in pairwise mode.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/analysis/final",
        help="Directory for final bundle outputs.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["training", "validation", "test"],
        help="Dataset split used to enrich rows.",
    )
    parser.add_argument(
        "--kb_path",
        default=str(REPO_ROOT / "data" / "kb" / "kb_chunks.jsonl"),
        help="Path to KB chunks JSONL.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=50,
        help="Maximum examples in each output file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking/sampling.",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help=(
            "Generate final bundle outputs. If --baseline/--compare are omitted, "
            "bundle mode is enabled automatically."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> dict[tuple[str, str], dict]:
    data: dict[tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record.get("text_id")), str(record.get("sent_id")))
            data[key] = record
    return data


def _load_kb_chunks(path: Path) -> dict[str, str]:
    kb_text_by_id: dict[str, str] = {}
    if not path.exists():
        LOGGER.warning("KB path not found, continuing without KB texts: %s", path)
        return kb_text_by_id
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            chunk_id = str(item.get("id", "")).strip()
            if chunk_id:
                kb_text_by_id[chunk_id] = str(item.get("text", "")).strip()
    return kb_text_by_id


def _f1_for_labels(gold: set[str], pred: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def _infer_context_type(path: Path) -> str:
    tokens = path.stem.split("_")
    if len(tokens) > 1 and tokens[1] in CONTEXT_NAMES:
        return tokens[1]
    return "sentence"


def _build_doc_index(split_df) -> tuple[dict[tuple[str, str], str], dict[str, list[str]], dict[tuple[str, str], int]]:
    target_text_by_key: dict[tuple[str, str], str] = {}
    doc_sentences: dict[str, list[str]] = {}
    sent_pos_by_key: dict[tuple[str, str], int] = {}

    for text_id, group in split_df.groupby("text_id", sort=False):
        rows = group.to_dict(orient="records")
        rows.sort(key=lambda r: _safe_sent_sort_key(str(r.get("sent_id", ""))))
        sentences = [str(r.get("text", "")) for r in rows]
        doc_sentences[str(text_id)] = sentences
        for idx, row in enumerate(rows):
            key = (str(row.get("text_id")), str(row.get("sent_id")))
            target_text_by_key[key] = str(row.get("text", ""))
            sent_pos_by_key[key] = idx

    return target_text_by_key, doc_sentences, sent_pos_by_key


def _safe_sent_sort_key(sent_id: str) -> tuple[int, str]:
    if sent_id.isdigit():
        return (0, f"{int(sent_id):09d}")
    return (1, sent_id)


def _context_excerpt(
    *,
    context_type: str,
    text_id: str,
    sent_idx: int,
    doc_sentences: dict[str, list[str]],
    n_prev: int = 2,
    n_next: int = 2,
    max_chars: int = 1200,
) -> str:
    sentences = doc_sentences.get(text_id, [])
    if not sentences:
        return ""
    if sent_idx < 0 or sent_idx >= len(sentences):
        sent_idx = max(0, min(sent_idx, len(sentences) - 1))

    if context_type == "sentence":
        excerpt = sentences[sent_idx]
    elif context_type == "window":
        start = max(0, sent_idx - n_prev)
        end = min(len(sentences), sent_idx + n_next + 1)
        excerpt = " ".join(sentences[start:end])
    else:
        excerpt = " ".join(sentences)

    if len(excerpt) > max_chars:
        return excerpt[: max_chars - 3] + "..."
    return excerpt


def _enriched_candidates(
    *,
    baseline_path: Path,
    compare_path: Path,
    split_df,
    kb_text_by_id: dict[str, str],
) -> list[dict]:
    baseline = _load_jsonl(baseline_path)
    compare = _load_jsonl(compare_path)
    target_text_by_key, doc_sentences, sent_pos_by_key = _build_doc_index(split_df)
    context_type = _infer_context_type(compare_path)

    candidates: list[dict] = []
    for key, base in baseline.items():
        if key not in compare:
            continue
        comp = compare[key]
        text_id, sent_id = key
        sent_idx = sent_pos_by_key.get(key, 0)
        gold = set(base.get("gold_labels", []) or [])
        base_pred = set(base.get("pred_labels", []) or [])
        comp_pred = set(comp.get("pred_labels", []) or [])
        base_f1 = _f1_for_labels(gold, base_pred)
        comp_f1 = _f1_for_labels(gold, comp_pred)
        delta = comp_f1 - base_f1

        chunk_ids = comp.get("kb_chunk_ids", []) or base.get("kb_chunk_ids", []) or []
        kb_texts = [kb_text_by_id.get(str(cid), "") for cid in chunk_ids]
        kb_texts = [txt for txt in kb_texts if txt]
        kb_values = comp.get("kb_values", []) or base.get("kb_values", []) or []

        candidates.append(
            {
                "text_id": text_id,
                "sent_id": sent_id,
                "target_sentence": target_text_by_key.get(key, ""),
                "context_excerpt": _context_excerpt(
                    context_type=context_type,
                    text_id=text_id,
                    sent_idx=sent_idx,
                    doc_sentences=doc_sentences,
                ),
                "gold_labels": sorted(gold),
                "baseline_pred": sorted(base_pred),
                "compare_pred": sorted(comp_pred),
                "baseline_f1": base_f1,
                "compare_f1": comp_f1,
                "delta_f1": delta,
                "retrieved_kb_values": kb_values,
                "retrieved_kb_texts": kb_texts,
                "raw_llm_output": comp.get("raw_output", base.get("raw_output", "")),
            }
        )
    return candidates


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_improved_and_worsened(candidates: list[dict], max_examples: int) -> list[dict]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x["delta_f1"], reverse=True)
    half = max(1, max_examples // 2)
    improved = candidates[:half]
    worsened = list(reversed(candidates[-half:]))
    return improved + worsened


def _find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _build_bundle(args: argparse.Namespace) -> None:
    split_df = load_split(args.split)
    kb_text_by_id = _load_kb_chunks(Path(args.kb_path))
    pred_dir = REPO_ROOT / "results" / "predictions"
    output_dir = Path(args.output_dir)

    deberta_baseline = _find_first_existing(
        [
            pred_dir / "deberta_sentence_no_rag_seed42.jsonl",
            pred_dir / "deberta_sentence_no_rag_seed42_deberta-v3-large.jsonl",
        ]
    )
    deberta_compare = _find_first_existing(
        [
            pred_dir / "deberta_doc_rag_seed42.jsonl",
            pred_dir / "deberta_doc_rag_seed42_deberta-v3-large.jsonl",
        ]
    )
    if deberta_baseline is None or deberta_compare is None:
        raise FileNotFoundError(
            "Could not find default DeBERTa baseline/compare predictions for bundle."
        )
    d_candidates = _enriched_candidates(
        baseline_path=deberta_baseline,
        compare_path=deberta_compare,
        split_df=split_df,
        kb_text_by_id=kb_text_by_id,
    )
    d_rows = _select_improved_and_worsened(d_candidates, args.max_examples)
    _write_jsonl(output_dir / "qual_examples_deberta_context_rag.jsonl", d_rows)

    llm_baseline = _find_first_existing(
        [
            pred_dir / "mistral_sentence_no_rag_Mistral-Large-Instruct-2407_test.jsonl",
            pred_dir / "qwen_sentence_no_rag_Qwen2.5-72B-Instruct_test.jsonl",
            pred_dir / "gemma_sentence_no_rag_test.jsonl",
        ]
    )
    llm_compare = _find_first_existing(
        [
            pred_dir / "mistral_sentence_rag_Mistral-Large-Instruct-2407_test.jsonl",
            pred_dir / "qwen_sentence_rag_Qwen2.5-72B-Instruct_test.jsonl",
            pred_dir / "gemma_sentence_rag_test.jsonl",
        ]
    )
    if llm_baseline is None or llm_compare is None:
        raise FileNotFoundError(
            "Could not find default LLM baseline/compare predictions for bundle."
        )
    llm_candidates = _enriched_candidates(
        baseline_path=llm_baseline,
        compare_path=llm_compare,
        split_df=split_df,
        kb_text_by_id=kb_text_by_id,
    )
    llm_rows = _select_improved_and_worsened(llm_candidates, args.max_examples)
    _write_jsonl(output_dir / "qual_examples_llm_rag.jsonl", llm_rows)

    failure_rows = [
        row
        for row in d_candidates
        if row["compare_f1"] == 0.0 and len(row.get("gold_labels", [])) > 0
    ]
    failure_rows = sorted(
        failure_rows,
        key=lambda x: (len(x.get("gold_labels", [])), x["delta_f1"]),
    )
    if len(failure_rows) > args.max_examples:
        failure_rows = failure_rows[-args.max_examples:]
    _write_jsonl(output_dir / "qual_examples_failure_cases.jsonl", failure_rows)

    LOGGER.info(
        "Saved bundle files to %s: deberta=%d llm=%d failures=%d",
        output_dir,
        len(d_rows),
        len(llm_rows),
        len(failure_rows),
    )


def _pair_mode(args: argparse.Namespace) -> None:
    if not args.baseline or not args.compare:
        raise ValueError("Pairwise mode requires both --baseline and --compare.")
    split_df = load_split(args.split)
    kb_text_by_id = _load_kb_chunks(Path(args.kb_path))
    candidates = _enriched_candidates(
        baseline_path=Path(args.baseline),
        compare_path=Path(args.compare),
        split_df=split_df,
        kb_text_by_id=kb_text_by_id,
    )
    rows = _select_improved_and_worsened(candidates, args.max_examples)
    output_path = Path(args.output)
    _write_jsonl(output_path, rows)
    LOGGER.info("Saved %d qualitative examples to %s", len(rows), output_path)


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")
    silence_transformers_logging()
    random.seed(args.seed)

    bundle_mode = args.bundle or not (args.baseline and args.compare)
    if bundle_mode:
        _build_bundle(args)
    else:
        _pair_mode(args)


if __name__ == "__main__":
    main()
