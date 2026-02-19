"""Analyze RAG architecture results from JSON metrics."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _parse_key(path: Path) -> dict:
    name = path.name
    # Examples: deberta_doc_rag_seed42_test_metrics.json
    #           deberta_doc_no_rag_seed42_test_metrics.json
    m = re.match(
        r"(?P<model>[^_]+)_(?P<context>[^_]+)_(?P<rag>rag|no_rag)_seed(?P<seed>\d+)",
        name,
    )
    info = {
        "model": None,
        "context": None,
        "rag": None,
        "seed": None,
        "mode": None,
    }
    if m:
        info.update(m.groupdict())
    # Try to infer mode from filename if present.
    for mode in ("early", "late", "cross_attention"):
        if mode in name:
            info["mode"] = mode
            break
    if info["mode"] is None:
        if info.get("rag") == "no_rag":
            info["mode"] = "none"
        elif info.get("rag") == "rag":
            info["mode"] = "rag"
    return info


def load_metrics(pattern: str) -> dict[str, dict]:
    """Load metrics JSON files matching a glob pattern."""
    metrics_dict: dict[str, dict] = {}
    for path in sorted(Path().glob(pattern)):
        if path.is_dir():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key_info = _parse_key(path)
        key = "|".join(
            [
                str(key_info.get("mode")),
                str(key_info.get("context")),
                str(key_info.get("seed")),
                path.name,
            ]
        )
        metrics_dict[key] = {"meta": key_info, "metrics": data, "path": str(path)}
    return metrics_dict


def summarize_overall(metrics_dict: dict[str, dict]) -> None:
    rows = defaultdict(list)
    for entry in metrics_dict.values():
        meta = entry.get("meta", {})
        metrics = entry.get("metrics", {})
        mode = meta.get("mode", "unknown")
        rows[mode].append(
            (float(metrics.get("macro_f1", 0.0)), float(metrics.get("micro_f1", 0.0)))
        )

    print("Architecture\tMacro-F1 (mean±std)\tMicro-F1 (mean±std)\tN")
    for mode, vals in rows.items():
        if not vals:
            continue
        macros = [v[0] for v in vals]
        micros = [v[1] for v in vals]
        macro_mean = sum(macros) / len(macros)
        micro_mean = sum(micros) / len(micros)
        macro_std = (sum((m - macro_mean) ** 2 for m in macros) / len(macros)) ** 0.5
        micro_std = (sum((m - micro_mean) ** 2 for m in micros) / len(micros)) ** 0.5
        print(
            f"{mode}\t{macro_mean:.4f}±{macro_std:.4f}\t"
            f"{micro_mean:.4f}±{micro_std:.4f}\t{len(vals)}"
        )


def summarize_per_value(metrics_dict: dict[str, dict]) -> None:
    # Aggregate per-label F1 by architecture.
    per_mode = defaultdict(lambda: defaultdict(list))
    for entry in metrics_dict.values():
        meta = entry.get("meta", {})
        metrics = entry.get("metrics", {})
        mode = meta.get("mode", "unknown")
        per_label = metrics.get("per_label_f1", {}) or {}
        for label, f1 in per_label.items():
            per_mode[mode][label].append(float(f1))

    # Print mean per-label F1 by mode (wide table).
    labels = sorted({label for mode in per_mode for label in per_mode[mode]})
    if not labels:
        print("No per-label F1 data available.")
        return

    header = "Label\t" + "\t".join(sorted(per_mode.keys()))
    print(header)
    for label in labels:
        row = [label]
        for mode in sorted(per_mode.keys()):
            vals = per_mode[mode].get(label, [])
            mean = sum(vals) / len(vals) if vals else 0.0
            row.append(f"{mean:.4f}")
        print("\t".join(row))


def summarize_efficiency_logs(pattern: str) -> None:
    # Best-effort log parsing. Looks for lines with "Epoch" and optional "time=".
    rows = defaultdict(list)
    for path in sorted(Path().glob(pattern)):
        if path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "Epoch" not in text:
            continue
        meta = _parse_key(path)
        mode = meta.get("mode", "unknown")
        epoch_times = []
        for line in text.splitlines():
            if "Epoch" in line and "time=" in line:
                m = re.search(r"time=([0-9\.]+)", line)
                if m:
                    epoch_times.append(float(m.group(1)))
        if epoch_times:
            rows[mode].append(sum(epoch_times) / len(epoch_times))

    if not rows:
        print("No efficiency logs with time= found.")
        return
    print("Architecture\tAvg epoch time")
    for mode, vals in rows.items():
        mean = sum(vals) / len(vals)
        print(f"{mode}\t{mean:.2f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RAG architecture results.")
    parser.add_argument(
        "--pattern",
        default="results/**/deberta_*_test_metrics.json",
        help="Glob pattern for metrics JSON files.",
    )
    parser.add_argument(
        "--logs",
        default="results/**/logs/*.log",
        help="Glob pattern for training logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = load_metrics(args.pattern)
    print("== Overall ==")
    summarize_overall(metrics)
    print("\n== Per value ==")
    summarize_per_value(metrics)
    print("\n== Efficiency ==")
    summarize_efficiency_logs(args.logs)


if __name__ == "__main__":
    main()
