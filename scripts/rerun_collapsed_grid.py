"""Rerun collapsed grid rows and update metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from value_context_rag.models.training import train_and_eval
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger, silence_transformers_logging

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun collapsed grid configs.")
    parser.add_argument(
        "--config",
        default="configs/deberta_sentence.yaml",
        help="Base config to use.",
    )
    parser.add_argument(
        "--input_csv",
        default="results/analysis/grid_deberta_hparams.csv",
        help="CSV with grid results.",
    )
    parser.add_argument(
        "--output_csv",
        default="results/analysis/grid_deberta_hparams_rerun.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Macro-F1 threshold to consider a run collapsed.",
    )
    parser.add_argument(
        "--retry_collapsed",
        type=int,
        default=1,
        help="Retries for collapsed runs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    silence_transformers_logging()
    base_config = load_config(args.config)

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    updated: list[dict[str, str]] = []
    for row in rows:
        try:
            best_macro = float(row.get("best_macro_f1", 0.0))
        except Exception:
            best_macro = 0.0
        if best_macro >= args.threshold:
            row["collapsed"] = row.get("collapsed", "False")
            updated.append(row)
            continue

        wd = float(row["weight_decay"])
        batch = int(float(row["batch_size"]))
        max_len = int(float(row["max_length"]))
        lr = float(row["learning_rate"])
        LOGGER.warning(
            "Re-running collapsed config lr=%.1e wd=%.2f b=%d ml=%d",
            lr,
            wd,
            batch,
            max_len,
        )

        config = dict(base_config)
        config["context"] = dict(base_config.get("context", {}))
        config["rag"] = dict(base_config.get("rag", {}))
        config["training"] = dict(base_config.get("training", {}))

        config["context"]["type"] = "sentence"
        config["rag"]["enabled"] = False
        config["training"]["learning_rate"] = float(lr)
        config["training"]["weight_decay"] = float(wd)
        config["training"]["batch_size"] = int(batch)
        config["training"]["max_length"] = int(max_len)
        config["training"]["grad_accum_steps"] = 1

        attempts = max(args.retry_collapsed, 0) + 1
        new_best = float("-inf")
        collapsed = True
        run_name = f"rerun_lr{lr}_wd{wd}_b{batch}_ml{max_len}"
        for attempt in range(attempts):
            if attempt > 0:
                LOGGER.warning("Retrying collapsed run (attempt %d/%d)", attempt + 1, attempts)
            config["seed"] = int(base_config.get("seed", 42)) + attempt
            new_best, collapsed = train_and_eval(
                config, run_name=run_name, resume_path=None
            )
            if not collapsed:
                break

        if collapsed:
            results_dir = Path(config.get("results_dir", "results"))
            ckpt_dir = results_dir / "checkpoints"
            hf_dir = results_dir / "hf_models" / run_name
            for suffix in (".pt", "_last.pt"):
                path = ckpt_dir / f"{run_name}{suffix}"
                if path.exists():
                    path.unlink()
            if hf_dir.exists():
                for child in hf_dir.iterdir():
                    if child.is_file():
                        child.unlink()
                try:
                    hf_dir.rmdir()
                except OSError:
                    pass

        row["best_macro_f1"] = f"{new_best}"
        row["collapsed"] = str(collapsed)
        updated.append(row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(updated[0].keys()) if updated else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated)
    LOGGER.info("Saved rerun results to %s", output_path)


if __name__ == "__main__":
    main()
