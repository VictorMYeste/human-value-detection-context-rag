"""Higher-level analyses for RQ4–RQ5."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from value_context_rag.eval.metrics import compute_per_label_f1
from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


Condition = tuple[str, str, bool]


def per_value_metrics_across_conditions(
    results_dict: Mapping[tuple, dict[str, np.ndarray]],
    *,
    label_names: list[str],
) -> pd.DataFrame:
    """Return per-value F1 for each condition."""
    rows: list[dict] = []
    LOGGER.debug("Computing per-value metrics for %d conditions", len(results_dict))
    for key, payload in results_dict.items():
        if len(key) == 3:
            model, context, rag = key
            seed = None
        elif len(key) == 4:
            model, context, rag, seed = key
        else:
            raise ValueError(f"Unexpected condition key format: {key}")
        gold = payload["gold"]
        pred = payload["pred"]
        per_label = compute_per_label_f1(gold, pred, label_names=label_names)
        for value, f1 in per_label.items():
            rows.append(
                {
                    "model": model,
                    "context": context,
                    "rag": rag,
                    "seed": seed,
                    "value": value,
                    "f1": f1,
                }
            )

    df = pd.DataFrame(rows)
    LOGGER.info("Built per-value metrics dataframe with %d rows", len(df))
    return df


def compute_deltas(per_value_df: pd.DataFrame) -> pd.DataFrame:
    """Compute context and RAG deltas per value."""
    required_cols = {"model", "context", "rag", "value", "f1"}
    missing = required_cols - set(per_value_df.columns)
    if missing:
        raise ValueError(f"per_value_df missing columns: {sorted(missing)}")

    df = per_value_df.copy()
    deltas: list[dict] = []
    LOGGER.debug("Computing deltas across %d rows", len(df))

    for (model, value), group in df.groupby(["model", "value"]):
        rag_on = group[group["rag"]]
        rag_off = group[~group["rag"]]
        if not rag_on.empty and not rag_off.empty:
            for context in group["context"].unique():
                f1_on = rag_on[rag_on["context"] == context]["f1"].mean()
                f1_off = rag_off[rag_off["context"] == context]["f1"].mean()
                if not np.isnan(f1_on) and not np.isnan(f1_off):
                    deltas.append(
                        {
                            "model": model,
                            "value": value,
                            "context": context,
                            "delta_rag": float(f1_on - f1_off),
                        }
                    )

        # context deltas (window - sentence, doc - sentence)
        base = group[group["context"] == "sentence"]
        if not base.empty:
            base_f1 = base["f1"].mean()
            for ctx in ["window", "doc"]:
                ctx_rows = group[group["context"] == ctx]
                if not ctx_rows.empty:
                    ctx_f1 = ctx_rows["f1"].mean()
                    deltas.append(
                        {
                            "model": model,
                            "value": value,
                            "context": ctx,
                            "delta_context": float(ctx_f1 - base_f1),
                        }
                    )

    delta_df = pd.DataFrame(deltas)
    LOGGER.info("Computed delta dataframe with %d rows", len(delta_df))
    return delta_df


def prediction_change_stats(
    predictions_by_condition: Mapping[tuple[str, bool], dict[str, np.ndarray]],
) -> dict[str, float]:
    """Compute change stats across contexts (sentence/window/doc)."""
    required_contexts = {"sentence", "window", "doc"}
    available = {ctx for (ctx, _rag) in predictions_by_condition.keys()}
    LOGGER.debug("Computing prediction changes for contexts=%s", sorted(available))
    if not required_contexts.issubset(available):
        raise ValueError(
            f"predictions_by_condition must include {sorted(required_contexts)}"
        )

    def _count_changes(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
        changed = (a != b).sum()
        improved = ((a == 0) & (b == 1)).sum()
        worsened = ((a == 1) & (b == 0)).sum()
        return {
            "changed": int(changed),
            "improved": int(improved),
            "worsened": int(worsened),
        }

    stats = {}
    pairs = [("sentence", "window"), ("sentence", "doc"), ("window", "doc")]
    for ctx_a, ctx_b in pairs:
        for rag in {False, True}:
            key_a = (ctx_a, rag)
            key_b = (ctx_b, rag)
            if (
                key_a not in predictions_by_condition
                or key_b not in predictions_by_condition
            ):
                continue
            pred_a = predictions_by_condition[key_a]["pred"]
            pred_b = predictions_by_condition[key_b]["pred"]
            counts = _count_changes(pred_a, pred_b)
            denom = pred_a.size if pred_a.size else 1
            stats[f"{ctx_a}_to_{ctx_b}_rag_{rag}"] = {
                "changed_rate": counts["changed"] / denom,
                "improved_rate": counts["improved"] / denom,
                "worsened_rate": counts["worsened"] / denom,
            }

    LOGGER.info("Computed prediction change stats for %d condition pairs", len(stats))
    return stats


def kb_values_per_prediction(records: Iterable[dict]) -> pd.DataFrame:
    """Expand prediction records into per-(sentence, predicted value, KB value) rows."""
    rows: list[dict] = []
    for record in records:
        text_id = record.get("text_id")
        sent_id = record.get("sent_id")
        pred_labels = record.get("pred_labels", []) or []
        kb_values = record.get("kb_values", []) or []
        for pred in pred_labels:
            for kb_val in kb_values:
                rows.append(
                    {
                        "text_id": text_id,
                        "sent_id": sent_id,
                        "pred_label": pred,
                        "kb_value": kb_val,
                    }
                )
    return pd.DataFrame(rows)
