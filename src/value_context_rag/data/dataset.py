"""Dataset loading utilities for value context RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)

SPLITS = {"train", "validation", "test"}

TEXT_ID_COL = "Text-ID"
SENT_ID_COL = "Sentence-ID"
TEXT_COL = "Text"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _ensure_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    return split


def _read_tsv(path: Path, *, debug: bool) -> pd.DataFrame:
    if debug:
        LOGGER.debug("Reading TSV from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Missing TSV file: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    if debug:
        LOGGER.debug("Loaded %s with shape %s", path.name, df.shape)
        LOGGER.debug("Columns for %s: %s", path.name, list(df.columns))
    return df


def _split_label_columns(
    columns: Iterable[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    label_pairs: Dict[str, List[str]] = {}
    non_label_cols: List[str] = []
    for col in columns:
        if col.endswith(" attained"):
            base = col[: -len(" attained")]
            label_pairs.setdefault(base, []).append(col)
        elif col.endswith(" constrained"):
            base = col[: -len(" constrained")]
            label_pairs.setdefault(base, []).append(col)
        else:
            non_label_cols.append(col)
    return label_pairs, non_label_cols


def _collapse_attained_constrained(
    labels_df: pd.DataFrame, *, debug: bool
) -> pd.DataFrame:
    label_pairs, _ = _split_label_columns(labels_df.columns)
    if not label_pairs:
        raise ValueError("No attained/constrained label columns found in labels.tsv")

    if debug:
        LOGGER.debug("Found %d label pairs", len(label_pairs))

    collapsed = labels_df[[TEXT_ID_COL, SENT_ID_COL]].copy()
    for base, cols in label_pairs.items():
        if len(cols) != 2:
            raise ValueError(
                f"Expected attained+constrained columns for '{base}', got {cols}"
            )
        series = (
            labels_df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .gt(0)
            .any(axis=1)
            .astype(float)
        )
        collapsed[base] = series

    return collapsed


def get_label_names(*, debug: bool = False) -> List[str]:
    """Return the label column names in the correct order."""
    LOGGER.info("Loading label names from raw data")
    for split in ("train", "validation", "test"):
        labels_path = RAW_DATA_DIR / split / "labels.tsv"
        if labels_path.exists():
            labels_df = _read_tsv(labels_path, debug=debug)
            label_pairs, _ = _split_label_columns(labels_df.columns)
            label_cols = list(label_pairs.keys())
            if debug:
                LOGGER.debug("Label columns (%d): %s", len(label_cols), label_cols)
            return label_cols
    raise FileNotFoundError("Could not locate labels.tsv in any split under data/raw")


def load_split(split: str, *, debug: bool = False) -> pd.DataFrame:
    """Load and merge sentences + labels for a split."""
    split = _ensure_split(split)
    LOGGER.info("Loading split %s", split)

    sentences_path = RAW_DATA_DIR / split / "sentences.tsv"
    labels_path = RAW_DATA_DIR / split / "labels.tsv"

    sentences_df = _read_tsv(sentences_path, debug=debug)
    labels_df = _read_tsv(labels_path, debug=debug)

    missing_sentence_cols = {TEXT_ID_COL, SENT_ID_COL, TEXT_COL} - set(
        sentences_df.columns
    )
    if missing_sentence_cols:
        raise ValueError(
            f"sentences.tsv missing columns: {sorted(missing_sentence_cols)}"
        )

    missing_label_cols = {TEXT_ID_COL, SENT_ID_COL} - set(labels_df.columns)
    if missing_label_cols:
        raise ValueError(f"labels.tsv missing columns: {sorted(missing_label_cols)}")

    labels_df = _collapse_attained_constrained(labels_df, debug=debug)
    label_cols = [c for c in labels_df.columns if c not in {TEXT_ID_COL, SENT_ID_COL}]
    if debug:
        LOGGER.debug("Merging on %s and %s", TEXT_ID_COL, SENT_ID_COL)
        LOGGER.debug("Label columns (%d): %s", len(label_cols), label_cols)

    merged = pd.merge(
        sentences_df,
        labels_df,
        on=[TEXT_ID_COL, SENT_ID_COL],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        LOGGER.warning("Merged dataframe is empty for split %s", split)

    merged = merged.rename(
        columns={
            TEXT_ID_COL: "text_id",
            SENT_ID_COL: "sent_id",
            TEXT_COL: "text",
        }
    )

    if label_cols:
        merged[label_cols] = merged[label_cols].astype(float)

    merged["text_id"] = merged["text_id"].astype(str)
    merged["sent_id"] = merged["sent_id"].astype(str)
    merged["text"] = merged["text"].astype(str)

    if debug:
        LOGGER.debug("Final merged shape: %s", merged.shape)
        LOGGER.debug("Final columns: %s", list(merged.columns))

    return merged


def group_by_document(
    df: pd.DataFrame, *, debug: bool = False
) -> Dict[str, List[dict]]:
    """Group rows by document (text_id) into a dict of lists."""
    if "text_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'text_id' column")

    LOGGER.info("Grouping dataframe by text_id")
    grouped: Dict[str, List[dict]] = {}
    for text_id, group in df.groupby("text_id", sort=False):
        grouped[str(text_id)] = group.to_dict(orient="records")

    if debug:
        LOGGER.debug("Grouped into %d documents", len(grouped))

    return grouped
