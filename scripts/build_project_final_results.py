"""Build paper-ready result tables for Project-final.md."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repository root without editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from value_context_rag.data.dataset import get_label_names
from value_context_rag.eval.metrics import macro_f1_from_arrays
from value_context_rag.eval.stats import paired_bootstrap_delta, paired_permutation_test
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)

SPLIT_NAMES = {"train", "validation", "test"}
CONTEXT_NAMES = {"sentence", "window", "doc"}
DEFAULT_VARIANTS = {
    "deberta": "microsoft/deberta-v3-base",
    "gemma": "google/gemma-3-12b-it",
    "qwen": "Qwen/Qwen2.5-72B-Instruct",
    "mistral": "mistralai/Mistral-Large-Instruct-2407",
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
}


@dataclass
class RunMeta:
    model_family: str
    model_variant: str
    context: str
    split: str
    rag_mode: str
    use_rag: bool
    seed: int | None
    run_id: int | None
    source: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Project-final result tables.")
    parser.add_argument(
        "--results_root",
        default="results",
        help="Root directory containing logs/predictions/analysis subdirs.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/analysis/final",
        help="Directory where final analysis CSV files will be written.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to keep by default (test/validation/train).",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=2000,
        help="Iterations for paired bootstrap/permutation tests.",
    )
    parser.add_argument(
        "--include_sensitivity",
        action="store_true",
        help="Include sens_topk files in outputs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _infer_variant(model_family: str, variant_tokens: list[str]) -> str:
    if model_family == "deberta":
        joined = "_".join(variant_tokens).lower()
        if "large" in joined:
            return "microsoft/deberta-v3-large"
        return DEFAULT_VARIANTS["deberta"]
    if variant_tokens:
        raw = "_".join(variant_tokens)
        if "/" in raw:
            return raw
        if model_family == "qwen":
            return f"Qwen/{raw}"
        if model_family == "mistral":
            return f"mistralai/{raw}"
        if model_family == "llama":
            return f"meta-llama/{raw}"
        return raw
    return DEFAULT_VARIANTS.get(model_family, model_family)


def _parse_condition(tokens: list[str], start: int) -> tuple[str, bool, int]:
    i = start
    if i + 1 < len(tokens) and tokens[i] == "no" and tokens[i + 1] == "rag":
        return "none", False, i + 2
    if (
        i + 2 < len(tokens)
        and tokens[i] == "cross"
        and tokens[i + 1] == "attention"
        and tokens[i + 2] == "rag"
    ):
        return "cross_attention", True, i + 3
    if i + 1 < len(tokens) and tokens[i] in {"crossattn", "cross_attention"}:
        if tokens[i + 1] == "rag":
            return "cross_attention", True, i + 2
    if i + 1 < len(tokens) and tokens[i] in {"early", "late"}:
        if tokens[i + 1] == "rag":
            return tokens[i], True, i + 2
    if i < len(tokens) and tokens[i] == "rag":
        return "early", True, i + 1
    return "none", False, i


def _parse_run_meta(stem: str, source: str) -> RunMeta | None:
    if stem.endswith("_metrics"):
        stem = stem[: -len("_metrics")]
    tokens = stem.split("_")
    if not tokens:
        return None
    if tokens[0] == "sens" and len(tokens) > 1 and tokens[1] == "topk":
        return None
    model_family = tokens[0].lower()
    if len(tokens) < 3:
        return None
    context = tokens[1].lower()
    if context not in CONTEXT_NAMES:
        return None

    rag_mode, use_rag, idx = _parse_condition(tokens, 2)

    seed: int | None = None
    if idx < len(tokens) and tokens[idx].startswith("seed"):
        raw_seed = tokens[idx].replace("seed", "")
        if raw_seed.isdigit():
            seed = int(raw_seed)
            idx += 1

    split = "test"
    if tokens and tokens[-1].lower() in SPLIT_NAMES:
        split = tokens[-1].lower()
        tail = len(tokens) - 1
    else:
        tail = len(tokens)

    variant_tokens = tokens[idx:tail]
    run_id: int | None = None
    if variant_tokens and variant_tokens[0].isdigit():
        run_id = int(variant_tokens[0])
        variant_tokens = variant_tokens[1:]

    model_variant = _infer_variant(model_family, variant_tokens)
    return RunMeta(
        model_family=model_family,
        model_variant=model_variant,
        context=context,
        split=split,
        rag_mode=rag_mode,
        use_rag=use_rag,
        seed=seed,
        run_id=run_id,
        source=source,
    )


def _apply_meta_overrides(meta: RunMeta, payload: dict) -> RunMeta:
    metrics_meta = payload.get("meta", {})
    if not isinstance(metrics_meta, dict):
        return meta
    model_variant = str(metrics_meta.get("model_name", meta.model_variant))
    context = str(metrics_meta.get("context_type", meta.context)).lower()
    rag_mode = str(metrics_meta.get("rag_mode", meta.rag_mode)).lower()
    use_rag = bool(metrics_meta.get("use_rag", meta.use_rag))
    split = str(metrics_meta.get("split", meta.split)).lower()
    seed = meta.seed
    if metrics_meta.get("seed") is not None:
        try:
            seed = int(metrics_meta["seed"])
        except (TypeError, ValueError):
            pass
    if rag_mode == "rag":
        rag_mode = "early"
    return RunMeta(
        model_family=meta.model_family,
        model_variant=model_variant,
        context=context if context in CONTEXT_NAMES else meta.context,
        split=split if split in SPLIT_NAMES else meta.split,
        rag_mode=rag_mode,
        use_rag=use_rag if rag_mode != "none" else False,
        seed=seed,
        run_id=meta.run_id,
        source=meta.source,
    )


def _load_metrics_rows(
    metrics_paths: list[Path],
    *,
    selected_split: str,
    include_sensitivity: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_rows: list[dict] = []
    per_value_rows: list[dict] = []
    for path in metrics_paths:
        stem = path.stem
        if not include_sensitivity and stem.startswith("sens_topk"):
            continue
        meta = _parse_run_meta(stem, str(path))
        if meta is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Skipping invalid JSON metrics file: %s", path)
            continue
        meta = _apply_meta_overrides(meta, payload)
        if meta.split != selected_split:
            continue
        macro_f1 = float(payload.get("macro_f1", 0.0))
        micro_f1 = float(payload.get("micro_f1", 0.0))
        main_rows.append(
            {
                "model_family": meta.model_family,
                "model_variant": meta.model_variant,
                "context": meta.context,
                "split": meta.split,
                "rag_mode": meta.rag_mode,
                "use_rag": meta.use_rag,
                "seed": meta.seed,
                "run_id": meta.run_id,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "source_file": str(path),
            }
        )

        per_label_f1 = payload.get("per_label_f1", {}) or {}
        per_label_support = payload.get("per_label_support", {}) or {}
        for label, f1_value in per_label_f1.items():
            support = per_label_support.get(label, {}) if per_label_support else {}
            per_value_rows.append(
                {
                    "model_family": meta.model_family,
                    "model_variant": meta.model_variant,
                    "context": meta.context,
                    "split": meta.split,
                    "rag_mode": meta.rag_mode,
                    "use_rag": meta.use_rag,
                    "seed": meta.seed,
                    "run_id": meta.run_id,
                    "value": label,
                    "f1": float(f1_value),
                    "gold_support": float(support.get("gold", 0.0)),
                    "pred_support": float(support.get("pred", 0.0)),
                    "gold_rate": float(support.get("gold_rate", 0.0)),
                    "pred_rate": float(support.get("pred_rate", 0.0)),
                    "source_file": str(path),
                }
            )

    main_df = pd.DataFrame(main_rows)
    per_value_df = pd.DataFrame(per_value_rows)
    return main_df, per_value_df


def _load_prediction_records(path: Path) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record.get("text_id")), str(record.get("sent_id")))
            records[key] = record
    return records


def _paired_arrays(
    left: dict[tuple[str, str], dict],
    right: dict[tuple[str, str], dict],
    label_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(set(left.keys()) & set(right.keys()))
    gold_rows: list[list[int]] = []
    left_rows: list[list[int]] = []
    right_rows: list[list[int]] = []
    for key in keys:
        left_row = left[key]
        right_row = right[key]
        gold_set = set(left_row.get("gold_labels", []) or [])
        left_set = set(left_row.get("pred_labels", []) or [])
        right_set = set(right_row.get("pred_labels", []) or [])
        gold_rows.append([1 if name in gold_set else 0 for name in label_names])
        left_rows.append([1 if name in left_set else 0 for name in label_names])
        right_rows.append([1 if name in right_set else 0 for name in label_names])
    return (
        np.asarray(gold_rows, dtype=int),
        np.asarray(left_rows, dtype=int),
        np.asarray(right_rows, dtype=int),
    )


def _aggregate_main(main_df: pd.DataFrame) -> pd.DataFrame:
    if main_df.empty:
        return pd.DataFrame()
    group_cols = [
        "model_family",
        "model_variant",
        "context",
        "split",
        "rag_mode",
        "use_rag",
    ]
    agg_df = (
        main_df.groupby(group_cols, as_index=False)
        .agg(
            n=("macro_f1", "size"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            micro_f1_mean=("micro_f1", "mean"),
            micro_f1_std=("micro_f1", "std"),
        )
        .sort_values(group_cols)
    )
    for col in ["macro_f1_std", "micro_f1_std"]:
        agg_df[col] = agg_df[col].fillna(0.0)
    return agg_df


def _compute_per_value_deltas(per_value_df: pd.DataFrame) -> pd.DataFrame:
    if per_value_df.empty:
        return pd.DataFrame()
    grouped = (
        per_value_df.groupby(
            ["model_family", "model_variant", "value", "context", "rag_mode", "use_rag"],
            as_index=False,
        )["f1"]
        .mean()
        .rename(columns={"f1": "mean_f1"})
    )
    rows: list[dict] = []
    for (family, variant, value), sub in grouped.groupby(
        ["model_family", "model_variant", "value"]
    ):
        def _get(context: str, rag_mode: str) -> float | None:
            match = sub[(sub["context"] == context) & (sub["rag_mode"] == rag_mode)]
            if match.empty:
                return None
            return float(match.iloc[0]["mean_f1"])

        sentence_none = _get("sentence", "none")
        window_none = _get("window", "none")
        doc_none = _get("doc", "none")
        sentence_early = _get("sentence", "early")
        window_early = _get("window", "early")
        doc_early = _get("doc", "early")
        doc_late = _get("doc", "late")
        doc_cross = _get("doc", "cross_attention")

        if sentence_none is not None and window_none is not None:
            rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "value": value,
                    "delta_type": "context_gain_no_rag",
                    "contrast": "window_none - sentence_none",
                    "delta_f1": window_none - sentence_none,
                }
            )
        if sentence_none is not None and doc_none is not None:
            rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "value": value,
                    "delta_type": "context_gain_no_rag",
                    "contrast": "doc_none - sentence_none",
                    "delta_f1": doc_none - sentence_none,
                }
            )
        if sentence_early is not None and window_early is not None:
            rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "value": value,
                    "delta_type": "context_gain_rag",
                    "contrast": "window_early - sentence_early",
                    "delta_f1": window_early - sentence_early,
                }
            )
        if sentence_early is not None and doc_early is not None:
            rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "value": value,
                    "delta_type": "context_gain_rag",
                    "contrast": "doc_early - sentence_early",
                    "delta_f1": doc_early - sentence_early,
                }
            )
        if sentence_none is not None and sentence_early is not None:
            rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "value": value,
                    "delta_type": "rag_gain",
                    "contrast": "sentence_early - sentence_none",
                    "delta_f1": sentence_early - sentence_none,
                }
            )
        if window_none is not None and window_early is not None:
            rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "value": value,
                    "delta_type": "rag_gain",
                    "contrast": "window_early - window_none",
                    "delta_f1": window_early - window_none,
                }
            )
        if doc_none is not None and doc_early is not None:
            rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "value": value,
                    "delta_type": "rag_gain",
                    "contrast": "doc_early - doc_none",
                    "delta_f1": doc_early - doc_none,
                }
            )
        if doc_early is not None and doc_late is not None:
            rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "value": value,
                    "delta_type": "architecture_gain",
                    "contrast": "doc_late - doc_early",
                    "delta_f1": doc_late - doc_early,
                }
            )
        if doc_early is not None and doc_cross is not None:
            rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "value": value,
                    "delta_type": "architecture_gain",
                    "contrast": "doc_cross_attention - doc_early",
                    "delta_f1": doc_cross - doc_early,
                }
            )

    return pd.DataFrame(rows)


def _prepare_prediction_runs(
    prediction_paths: list[Path],
    *,
    selected_split: str,
    include_sensitivity: bool,
) -> dict[tuple, dict]:
    loaded: dict[tuple, dict] = {}
    for path in prediction_paths:
        stem = path.stem
        if not include_sensitivity and stem.startswith("sens_topk"):
            continue
        meta = _parse_run_meta(stem, str(path))
        if meta is None:
            continue
        if meta.split != selected_split:
            continue
        records = _load_prediction_records(path)
        loaded[
            (
                meta.model_family,
                meta.model_variant,
                meta.context,
                meta.rag_mode,
                meta.seed,
                meta.split,
            )
        ] = {
            "meta": meta,
            "records": records,
            "source_file": str(path),
        }
    return loaded


def _compute_prediction_change(a_pred: np.ndarray, b_pred: np.ndarray) -> dict[str, float]:
    total = float(a_pred.size if a_pred.size else 1)
    changed = float((a_pred != b_pred).sum())
    to_positive = float(((a_pred == 0) & (b_pred == 1)).sum())
    to_negative = float(((a_pred == 1) & (b_pred == 0)).sum())
    unchanged = total - changed
    return {
        "changed_rate": changed / total,
        "to_positive_rate": to_positive / total,
        "to_negative_rate": to_negative / total,
        "unchanged_rate": unchanged / total,
    }


def _build_significance_and_change_tables(
    prediction_runs: dict[tuple, dict],
    *,
    label_names: list[str],
    n_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    significance_rows: list[dict] = []
    change_rows: list[dict] = []

    grouped: dict[tuple[str, str, int | None, str], dict[tuple[str, str], dict]] = {}
    for key, payload in prediction_runs.items():
        family, variant, context, rag_mode, seed, split = key
        group_key = (family, variant, seed, split)
        grouped.setdefault(group_key, {})
        grouped[group_key][(context, rag_mode)] = payload

    contrasts = [
        ("window_vs_sentence_no_rag", ("window", "none"), ("sentence", "none")),
        ("doc_vs_sentence_no_rag", ("doc", "none"), ("sentence", "none")),
        ("sentence_rag_vs_no_rag", ("sentence", "early"), ("sentence", "none")),
        ("window_rag_vs_no_rag", ("window", "early"), ("window", "none")),
        ("doc_rag_vs_no_rag", ("doc", "early"), ("doc", "none")),
        ("doc_late_vs_early", ("doc", "late"), ("doc", "early")),
        (
            "doc_cross_attention_vs_early",
            ("doc", "cross_attention"),
            ("doc", "early"),
        ),
    ]

    for (family, variant, seed, split), cond_map in grouped.items():
        for contrast_name, left_key, right_key in contrasts:
            if left_key not in cond_map or right_key not in cond_map:
                continue
            left_records = cond_map[left_key]["records"]
            right_records = cond_map[right_key]["records"]
            y_true, y_pred_left, y_pred_right = _paired_arrays(
                left_records, right_records, label_names
            )
            if y_true.size == 0:
                continue
            boot = paired_bootstrap_delta(
                y_true,
                y_pred_left,
                y_pred_right,
                metric_fn=macro_f1_from_arrays,
                n_iterations=n_iterations,
            )
            perm = paired_permutation_test(
                y_true,
                y_pred_left,
                y_pred_right,
                metric_fn=macro_f1_from_arrays,
                n_iterations=n_iterations,
            )
            for result in [boot, perm]:
                significance_rows.append(
                    {
                        "model_family": family,
                        "model_variant": variant,
                        "seed": seed,
                        "split": split,
                        "contrast": contrast_name,
                        "left_context": left_key[0],
                        "left_rag_mode": left_key[1],
                        "right_context": right_key[0],
                        "right_rag_mode": right_key[1],
                        "method": result.method,
                        "delta_macro_f1": result.delta,
                        "ci_low": result.ci_low,
                        "ci_high": result.ci_high,
                        "p_value": result.p_value,
                        "n_samples": result.n_samples,
                        "n_iterations": result.n_iterations,
                    }
                )

        for rag_mode in ["none", "early"]:
            trio = [(ctx, rag_mode) for ctx in ["sentence", "window", "doc"]]
            if not all(t in cond_map for t in trio):
                continue
            pairs = [("sentence", "window"), ("sentence", "doc"), ("window", "doc")]
            for ctx_a, ctx_b in pairs:
                a_records = cond_map[(ctx_a, rag_mode)]["records"]
                b_records = cond_map[(ctx_b, rag_mode)]["records"]
                _, a_pred, b_pred = _paired_arrays(a_records, b_records, label_names)
                rates = _compute_prediction_change(a_pred, b_pred)
                change_rows.append(
                    {
                        "model_family": family,
                        "model_variant": variant,
                        "seed": seed,
                        "split": split,
                        "rag_mode": rag_mode,
                        "contrast": f"{ctx_a}_to_{ctx_b}",
                        **rates,
                    }
                )

    return pd.DataFrame(significance_rows), pd.DataFrame(change_rows)


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")
    silence_transformers_logging()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_paths = sorted((results_root / "logs").glob("*_metrics.json")) + sorted(
        (results_root / "rag_architectures" / "logs").glob("*_metrics.json")
    )
    prediction_paths = sorted((results_root / "predictions").glob("*.jsonl")) + sorted(
        (results_root / "rag_architectures" / "predictions").glob("*.jsonl")
    )
    LOGGER.info("Found %d metrics files", len(metric_paths))
    LOGGER.info("Found %d prediction files", len(prediction_paths))

    main_df, per_value_df = _load_metrics_rows(
        metric_paths,
        selected_split=args.split,
        include_sensitivity=args.include_sensitivity,
    )
    main_df.to_csv(output_dir / "main_results.csv", index=False)
    per_value_df.to_csv(output_dir / "per_value_results.csv", index=False)

    main_agg_df = _aggregate_main(main_df)
    main_agg_df.to_csv(output_dir / "main_results_agg.csv", index=False)

    if not main_agg_df.empty:
        llm_families = {"gemma", "qwen", "mistral", "llama"}
        llm_df = main_agg_df[main_agg_df["model_family"].isin(llm_families)].copy()
    else:
        llm_df = pd.DataFrame()
    llm_df.to_csv(output_dir / "llm_results.csv", index=False)

    if not main_agg_df.empty:
        rag_arch_df = main_agg_df[
            (main_agg_df["model_family"] == "deberta")
            & (main_agg_df["context"] == "doc")
            & (main_agg_df["rag_mode"].isin(["early", "late", "cross_attention"]))
        ].copy()
    else:
        rag_arch_df = pd.DataFrame()
    rag_arch_df.to_csv(output_dir / "rag_architectures.csv", index=False)

    if not main_agg_df.empty:
        deberta_df = main_agg_df[main_agg_df["model_family"] == "deberta"].copy()
        deberta_df["model_scale"] = np.where(
            deberta_df["model_variant"].str.lower().str.contains("large"),
            "large",
            "base",
        )
        base = deberta_df[deberta_df["model_scale"] == "base"][
            [
                "context",
                "split",
                "rag_mode",
                "use_rag",
                "macro_f1_mean",
                "micro_f1_mean",
            ]
        ].rename(
            columns={
                "macro_f1_mean": "base_macro_f1_mean",
                "micro_f1_mean": "base_micro_f1_mean",
            }
        )
        large = deberta_df[deberta_df["model_scale"] == "large"][
            [
                "context",
                "split",
                "rag_mode",
                "use_rag",
                "macro_f1_mean",
                "micro_f1_mean",
            ]
        ].rename(
            columns={
                "macro_f1_mean": "large_macro_f1_mean",
                "micro_f1_mean": "large_micro_f1_mean",
            }
        )
        scale_df = base.merge(
            large, on=["context", "split", "rag_mode", "use_rag"], how="inner"
        )
        if not scale_df.empty:
            scale_df["delta_macro_f1_large_minus_base"] = (
                scale_df["large_macro_f1_mean"] - scale_df["base_macro_f1_mean"]
            )
            scale_df["delta_micro_f1_large_minus_base"] = (
                scale_df["large_micro_f1_mean"] - scale_df["base_micro_f1_mean"]
            )
    else:
        scale_df = pd.DataFrame()
    scale_df.to_csv(output_dir / "deberta_base_vs_large.csv", index=False)

    per_value_delta_df = _compute_per_value_deltas(per_value_df)
    per_value_delta_df.to_csv(output_dir / "per_value_deltas.csv", index=False)

    label_names = get_label_names()
    prediction_runs = _prepare_prediction_runs(
        prediction_paths,
        selected_split=args.split,
        include_sensitivity=args.include_sensitivity,
    )
    significance_df, prediction_changes_df = _build_significance_and_change_tables(
        prediction_runs,
        label_names=label_names,
        n_iterations=args.n_iterations,
    )
    significance_df.to_csv(output_dir / "significance_tests.csv", index=False)
    prediction_changes_df.to_csv(output_dir / "prediction_changes.csv", index=False)

    LOGGER.info("Wrote: %s", output_dir / "main_results.csv")
    LOGGER.info("Wrote: %s", output_dir / "main_results_agg.csv")
    LOGGER.info("Wrote: %s", output_dir / "per_value_results.csv")
    LOGGER.info("Wrote: %s", output_dir / "per_value_deltas.csv")
    LOGGER.info("Wrote: %s", output_dir / "rag_architectures.csv")
    LOGGER.info("Wrote: %s", output_dir / "llm_results.csv")
    LOGGER.info("Wrote: %s", output_dir / "deberta_base_vs_large.csv")
    LOGGER.info("Wrote: %s", output_dir / "significance_tests.csv")
    LOGGER.info("Wrote: %s", output_dir / "prediction_changes.csv")


if __name__ == "__main__":
    main()
