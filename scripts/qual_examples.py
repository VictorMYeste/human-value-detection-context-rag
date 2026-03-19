"""Extract qualitative examples from prediction files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select qualitative examples.")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline predictions JSONL (e.g., sentence no RAG).",
    )
    parser.add_argument(
        "--compare",
        required=True,
        help="Path to comparison predictions JSONL (e.g., doc+RAG).",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/qual_examples.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=50,
        help="Maximum examples to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
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


def _f1_for_labels(gold: set[str], pred: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold and pred:
        return 0.0
    if gold and not pred:
        return 0.0
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    silence_transformers_logging()
    random.seed(args.seed)

    baseline_path = Path(args.baseline)
    compare_path = Path(args.compare)
    baseline = _load_jsonl(baseline_path)
    compare = _load_jsonl(compare_path)

    candidates: list[dict] = []
    for key, base in baseline.items():
        if key not in compare:
            continue
        comp = compare[key]
        gold = set(base.get("gold_labels", []))
        base_pred = set(base.get("pred_labels", []))
        comp_pred = set(comp.get("pred_labels", []))
        base_f1 = _f1_for_labels(gold, base_pred)
        comp_f1 = _f1_for_labels(gold, comp_pred)
        delta = comp_f1 - base_f1
        candidates.append(
            {
                "text_id": base.get("text_id"),
                "sent_id": base.get("sent_id"),
                "gold_labels": sorted(gold),
                "baseline_pred": sorted(base_pred),
                "compare_pred": sorted(comp_pred),
                "baseline_f1": base_f1,
                "compare_f1": comp_f1,
                "delta_f1": delta,
                "baseline_kb_values": base.get("kb_values", []),
                "compare_kb_values": comp.get("kb_values", []),
            }
        )

    candidates.sort(key=lambda x: x["delta_f1"], reverse=True)
    improved = candidates[: args.max_examples // 2]
    worsened = list(reversed(candidates[-(args.max_examples // 2) :]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in improved + worsened:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    LOGGER.info(
        "Saved %d qualitative examples to %s", len(improved) + len(worsened), output_path
    )


if __name__ == "__main__":
    main()
