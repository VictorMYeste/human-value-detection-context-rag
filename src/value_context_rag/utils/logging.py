"""Simple unified logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _ensure_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    for existing in logger.handlers:
        if type(existing) is type(handler):
            return
    logger.addHandler(handler)


def get_logger(
    name: str,
    log_file: str | None = None,
    *,
    overwrite: bool = False,
) -> logging.Logger:
    """Configure console + optional file logging and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    _ensure_handler(logger, console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "a"
        file_handler = logging.FileHandler(log_path, encoding="utf-8", mode=mode)
        file_handler.setFormatter(formatter)
        _ensure_handler(logger, file_handler)

    logger.propagate = False
    return logger


def silence_transformers_logging() -> None:
    """Reduce noisy Transformers logging (e.g., load reports)."""
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_default_handler()
        hf_logging.disable_propagation()
    except Exception:
        return
