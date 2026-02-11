"""Aggregate prediction files and compute analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_context_rag.data.dataset import get_label_names
from value_context_rag.eval.analysis import (
    compute_deltas,
    per_value_metrics_across_conditions,
    prediction_change_stats,
)
from value_context_rag.eval.metrics import compute_global_metrics, macro_f1_from_arrays
from value_context_rag.eval.stats import paired_bootstrap_delta, paired_permutation_test
from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prediction files.")
    parser.add_argument(
        "--results_dir",
        default="results/predictions",
        help="Directory containing prediction JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/analysis",
        help="Directory to write analysis outputs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--canonical_seed",
        type=int,
        default=None,
        help="Canonical DeBERTa seed to use for comparisons. If omitted, the best validation macro-F1 seed is chosen.",
    )
    return parser.parse_args()


def _parse_filename(name: str) -> tuple[str, str, bool, str, int | None]:
    # Expected patterns:
    # deberta_<context>_<rag|no_rag>_seed<seed>_<split>.jsonl
    # gemma_<context>_<rag|no_rag>_<split>.jsonl
    parts = name.replace(".jsonl", "").split("_")
    model = parts[0]
    context = parts[1]
    rag = parts[2] == "rag"

    split = parts[-1]
    seed = None
    for part in parts:
        if part.startswith("seed"):
            try:
                seed = int(part.replace("seed", ""))
            except Exception:
                seed = None
            break
    return model, context, rag, split, seed


def _load_predictions(
    path: Path, label_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    gold = []
    pred = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            gold_labels = set(record.get("gold_labels", []))
            pred_labels = set(record.get("pred_labels", []))
            gold.append([1 if name in gold_labels else 0 for name in label_names])
            pred.append([1 if name in pred_labels else 0 for name in label_names])
    return np.asarray(gold, dtype=int), np.asarray(pred, dtype=int)


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = get_label_names()
    results: dict[tuple[str, str, bool, int | None], dict[str, np.ndarray]] = {}
    metrics_rows: list[dict] = []

    for path in results_dir.glob("*.jsonl"):
        model, context, rag, split, seed = _parse_filename(path.name)
        LOGGER.debug(
            "Loading predictions: %s (model=%s context=%s rag=%s split=%s seed=%s)",
            path,
            model,
            context,
            rag,
            split,
            seed,
        )
        gold, pred = _load_predictions(path, label_names)
        results[(model, context, rag, seed)] = {"gold": gold, "pred": pred}
        global_metrics = compute_global_metrics(gold, pred)
        metrics_rows.append(
            {
                "model": model,
                "context": context,
                "rag": rag,
                "split": split,
                "seed": seed,
                **global_metrics,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    LOGGER.debug("Wrote %d rows to metrics_summary.csv", len(metrics_df))

    # Aggregate metrics across seeds (DeBERTa only)
    deberta_df = metrics_df[metrics_df["model"] == "deberta"].copy()
    if not deberta_df.empty:
        agg = deberta_df.groupby(
            ["model", "context", "rag", "split"], as_index=False
        ).agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            micro_f1_mean=("micro_f1", "mean"),
            micro_f1_std=("micro_f1", "std"),
        )
        agg.to_csv(output_dir / "metrics_summary_agg.csv", index=False)
        LOGGER.debug("Wrote %d rows to metrics_summary_agg.csv", len(agg))

    # Canonical seed filtering for DeBERTa comparisons
    canonical_seed = args.canonical_seed
    if canonical_seed is None and not deberta_df.empty:
        val_df = deberta_df[deberta_df["split"] == "validation"].copy()
        if not val_df.empty:
            seed_scores = val_df.groupby("seed", as_index=False)["macro_f1"].mean()
            if not seed_scores.empty:
                canonical_seed = int(
                    seed_scores.sort_values("macro_f1", ascending=False).iloc[0]["seed"]
                )
                LOGGER.info(
                    "Selected canonical seed from validation: %d", canonical_seed
                )

    if canonical_seed is not None:
        results_for_analysis = {
            (m, c, r, s): v
            for (m, c, r, s), v in results.items()
            if (m != "deberta") or (s == canonical_seed)
        }
        (output_dir / "canonical_seed.json").write_text(
            json.dumps(
                {
                    "canonical_seed": canonical_seed,
                    "source": (
                        "validation_macro_f1"
                        if args.canonical_seed is None
                        else "manual"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        LOGGER.info("Saved canonical seed to %s", output_dir / "canonical_seed.json")
    else:
        results_for_analysis = results

    per_value_df = per_value_metrics_across_conditions(
        results_for_analysis, label_names=label_names
    )
    per_value_df.to_csv(output_dir / "per_value_metrics.csv", index=False)
    LOGGER.debug("Wrote %d rows to per_value_metrics.csv", len(per_value_df))

    deltas_df = compute_deltas(per_value_df)
    deltas_df.to_csv(output_dir / "deltas.csv", index=False)
    LOGGER.debug("Wrote %d rows to deltas.csv", len(deltas_df))

    # Prediction change stats for DeBERTa conditions (if present)
    pred_change_by_seed: dict[str, dict] = {}
    seed_set = {s for (_m, _c, _r, s) in results.keys() if s is not None}
    for seed in seed_set:
        pred_change = prediction_change_stats(
            {
                (context, rag): payload
                for (model, context, rag, s), payload in results.items()
                if model == "deberta" and s == seed
            }
        )
        pred_change_by_seed[str(seed)] = pred_change
    (output_dir / "prediction_change_stats.json").write_text(
        json.dumps(pred_change_by_seed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.debug("Saved prediction change stats for %d seeds", len(pred_change_by_seed))

    LOGGER.info("Saved metrics summary to %s", output_dir / "metrics_summary.csv")
    LOGGER.info("Saved per-value metrics to %s", output_dir / "per_value_metrics.csv")
    LOGGER.info("Saved deltas to %s", output_dir / "deltas.csv")

    # Significance tests (paired) for key condition contrasts
    test_rows: list[dict] = []
    for model in {m for (m, _, _, _) in results_for_analysis.keys()}:
        for rag in [False, True]:
            for seed in {
                s for (m, _, _, s) in results_for_analysis.keys() if m == model
            }:
                key_sentence = (model, "sentence", rag, seed)
                key_window = (model, "window", rag, seed)
                key_doc = (model, "doc", rag, seed)
                if (
                    key_sentence in results_for_analysis
                    and key_window in results_for_analysis
                ):
                    y = results_for_analysis[key_sentence]["gold"]
                    a = results_for_analysis[key_window]["pred"]
                    b = results_for_analysis[key_sentence]["pred"]
                    boot = paired_bootstrap_delta(
                        y, a, b, metric_fn=macro_f1_from_arrays
                    )
                    perm = paired_permutation_test(
                        y, a, b, metric_fn=macro_f1_from_arrays
                    )
                    test_rows.append(
                        {
                            "model": model,
                            "contrast": "window_vs_sentence",
                            "rag": rag,
                            "seed": seed,
                            "method": boot.method,
                            "delta": boot.delta,
                            "ci_low": boot.ci_low,
                            "ci_high": boot.ci_high,
                            "p_value": boot.p_value,
                        }
                    )
                    test_rows.append(
                        {
                            "model": model,
                            "contrast": "window_vs_sentence",
                            "rag": rag,
                            "seed": seed,
                            "method": perm.method,
                            "delta": perm.delta,
                            "ci_low": perm.ci_low,
                            "ci_high": perm.ci_high,
                            "p_value": perm.p_value,
                        }
                    )

                if (
                    key_sentence in results_for_analysis
                    and key_doc in results_for_analysis
                ):
                    y = results_for_analysis[key_sentence]["gold"]
                    a = results_for_analysis[key_doc]["pred"]
                    b = results_for_analysis[key_sentence]["pred"]
                    boot = paired_bootstrap_delta(
                        y, a, b, metric_fn=macro_f1_from_arrays
                    )
                    perm = paired_permutation_test(
                        y, a, b, metric_fn=macro_f1_from_arrays
                    )
                    test_rows.append(
                        {
                            "model": model,
                            "contrast": "doc_vs_sentence",
                            "rag": rag,
                            "seed": seed,
                            "method": boot.method,
                            "delta": boot.delta,
                            "ci_low": boot.ci_low,
                            "ci_high": boot.ci_high,
                            "p_value": boot.p_value,
                        }
                    )
                    test_rows.append(
                        {
                            "model": model,
                            "contrast": "doc_vs_sentence",
                            "rag": rag,
                            "seed": seed,
                            "method": perm.method,
                            "delta": perm.delta,
                            "ci_low": perm.ci_low,
                            "ci_high": perm.ci_high,
                            "p_value": perm.p_value,
                        }
                    )

        # RAG effect per context
        for context in ["sentence", "window", "doc"]:
            for seed in {
                s for (m, _, _, s) in results_for_analysis.keys() if m == model
            }:
                key_no_rag = (model, context, False, seed)
                key_rag = (model, context, True, seed)
                if (
                    key_no_rag in results_for_analysis
                    and key_rag in results_for_analysis
                ):
                    y = results_for_analysis[key_no_rag]["gold"]
                    a = results_for_analysis[key_rag]["pred"]
                    b = results_for_analysis[key_no_rag]["pred"]
                    boot = paired_bootstrap_delta(
                        y, a, b, metric_fn=macro_f1_from_arrays
                    )
                    perm = paired_permutation_test(
                        y, a, b, metric_fn=macro_f1_from_arrays
                    )
                    test_rows.append(
                        {
                            "model": model,
                            "contrast": f"rag_vs_no_rag_{context}",
                            "rag": True,
                            "seed": seed,
                            "method": boot.method,
                            "delta": boot.delta,
                            "ci_low": boot.ci_low,
                            "ci_high": boot.ci_high,
                            "p_value": boot.p_value,
                        }
                    )
                    test_rows.append(
                        {
                            "model": model,
                            "contrast": f"rag_vs_no_rag_{context}",
                            "rag": True,
                            "seed": seed,
                            "method": perm.method,
                            "delta": perm.delta,
                            "ci_low": perm.ci_low,
                            "ci_high": perm.ci_high,
                            "p_value": perm.p_value,
                        }
                    )

    if test_rows:
        tests_df = pd.DataFrame(test_rows)
        tests_df.to_csv(output_dir / "significance_tests.csv", index=False)
        LOGGER.info(
            "Saved significance tests to %s", output_dir / "significance_tests.csv"
        )
        LOGGER.debug("Wrote %d rows to significance_tests.csv", len(tests_df))

    # Model comparison: DeBERTa vs Gemma (same context + RAG)
    model_rows: list[dict] = []
    for context in ["sentence", "window", "doc"]:
        for rag in [False, True]:
            for seed in {s for (_m, _c, _r, s) in results_for_analysis.keys()}:
                key_deberta = ("deberta", context, rag, seed)
                key_gemma = ("gemma", context, rag, seed)
                if (
                    key_deberta in results_for_analysis
                    and key_gemma in results_for_analysis
                ):
                    y = results_for_analysis[key_deberta]["gold"]
                    a = results_for_analysis[key_deberta]["pred"]
                    b = results_for_analysis[key_gemma]["pred"]
                    boot = paired_bootstrap_delta(
                        y, a, b, metric_fn=macro_f1_from_arrays
                    )
                    perm = paired_permutation_test(
                        y, a, b, metric_fn=macro_f1_from_arrays
                    )
                    model_rows.append(
                        {
                            "contrast": f"deberta_vs_gemma_{context}",
                            "rag": rag,
                            "seed": seed,
                            "method": boot.method,
                            "delta": boot.delta,
                            "ci_low": boot.ci_low,
                            "ci_high": boot.ci_high,
                            "p_value": boot.p_value,
                        }
                    )
                    model_rows.append(
                        {
                            "contrast": f"deberta_vs_gemma_{context}",
                            "rag": rag,
                            "seed": seed,
                            "method": perm.method,
                            "delta": perm.delta,
                            "ci_low": perm.ci_low,
                            "ci_high": perm.ci_high,
                            "p_value": perm.p_value,
                        }
                    )

    if model_rows:
        model_df = pd.DataFrame(model_rows)
        model_df.to_csv(output_dir / "model_comparisons.csv", index=False)
        LOGGER.info(
            "Saved model comparisons to %s", output_dir / "model_comparisons.csv"
        )
        LOGGER.debug("Wrote %d rows to model_comparisons.csv", len(model_df))


if __name__ == "__main__":
    main()
