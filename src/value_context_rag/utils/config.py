"""YAML config loader with minimal defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


DEFAULTS: Dict[str, Any] = {
    "data": {
        "raw_dir": "data/raw",
        "kb_dir": "data/kb",
    },
    "results_dir": "results",
    "context": {
        "type": "sentence",
        "n_prev": 2,
        "n_next": 2,
    },
    "rag": {
        "enabled": False,
        "top_k": 5,
    },
    "llm": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 64,
        "max_prompt_tokens": 3072,
        "device": "cuda",
        "quantization": "8bit",
    },
    "training": {
        "batch_size": 16,
        "num_epochs": 10,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "max_length": 1024,
        "checkpoint_every_epochs": 1,
        "early_stopping_patience": 3,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and apply minimal defaults."""
    config_path = Path(path)
    LOGGER.info("Loading config from %s", config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    config = _deep_merge(DEFAULTS, raw)
    return config
