"""Build a CSV summary of KB value -> predicted label counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from value_context_rag.eval.analysis import kb_values_per_prediction
from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize KB value usage in predictions."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: alongside predictions).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    records = _load_records(pred_path)
    df = kb_values_per_prediction(records)
    if df.empty:
        LOGGER.warning("No KB values found in predictions")
        return

    summary = (
        df.groupby(["kb_value", "pred_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    output_path = (
        Path(args.output)
        if args.output
        else pred_path.with_suffix(".kb_value_summary.csv")
    )
    summary.to_csv(output_path, index=False)
    LOGGER.info("Saved KB value summary to %s", output_path)


if __name__ == "__main__":
    main()
