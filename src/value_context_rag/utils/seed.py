"""Random seed utilities for reproducibility."""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_SEED = 42
ALT_SEEDS = (7, 1701)


def get_default_seeds() -> tuple[int, ...]:
    """Return the default list of seeds used across the project."""
    return (DEFAULT_SEED,) + ALT_SEEDS


def set_seed(seed: int = DEFAULT_SEED, *, debug: bool = False) -> None:
    """Set seeds for Python, NumPy, and PyTorch (if installed)."""
    if debug:
        LOGGER.debug("Setting random seeds with seed=%d", seed)
    else:
        LOGGER.info("Setting random seeds")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if debug:
            LOGGER.debug(
                "PyTorch seed set (cuda_available=%s)", torch.cuda.is_available()
            )
    except Exception as exc:  # pragma: no cover - optional dependency
        if debug:
            LOGGER.debug("PyTorch not available or failed to seed: %s", exc)
