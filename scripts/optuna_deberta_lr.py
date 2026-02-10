"""Optuna search for DeBERTa learning rate (sentence, no RAG)."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna

from value_context_rag.models.training import train_and_eval
from value_context_rag.utils.config import load_config
from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna LR search for DeBERTa.")
    parser.add_argument(
        "--config",
        default="configs/deberta_sentence.yaml",
        help="Base config to use.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit samples for quick search.",
    )
    parser.add_argument(
        "--study_name",
        default="deberta_lr_search",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/optuna_deberta_lr.csv",
        help="CSV path to store trial results.",
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

    base_config = load_config(args.config)

    def objective(trial: optuna.Trial) -> float:
        config = dict(base_config)
        config["context"] = dict(base_config.get("context", {}))
        config["rag"] = dict(base_config.get("rag", {}))
        config["training"] = dict(base_config.get("training", {}))

        config["context"]["type"] = "sentence"
        config["rag"]["enabled"] = False

        lr = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
        config["training"]["learning_rate"] = lr
        config["training"]["weight_decay"] = 0.01
        config["training"]["num_epochs"] = 10
        config["training"]["early_stopping_patience"] = 3

        if args.max_samples is not None:
            config["max_samples"] = args.max_samples

        run_name = f"optuna_deberta_sentence_lr_{trial.number}"
        best_macro_f1 = train_and_eval(config, run_name=run_name)
        return float(best_macro_f1)

    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=args.trials)

    LOGGER.info("Best trial: %s", study.best_trial.params)

    results_df = study.trials_dataframe()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    LOGGER.info("Saved Optuna results to %s", output_path)


if __name__ == "__main__":
    main()
