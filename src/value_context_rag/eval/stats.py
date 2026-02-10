"""Statistical tests for paired condition comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from value_context_rag.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PairedTestResult:
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str
    n_samples: int
    n_iterations: int


def paired_bootstrap_delta(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> PairedTestResult:
    """Paired bootstrap over samples; returns CI and p-value for delta."""
    if y_true.shape != y_pred_a.shape or y_true.shape != y_pred_b.shape:
        raise ValueError("Shapes of y_true, y_pred_a, and y_pred_b must match")

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    if n == 0:
        raise ValueError("Empty arrays provided to paired_bootstrap_delta")
    LOGGER.debug(
        "Running paired bootstrap (n=%d, iterations=%d, seed=%d)",
        n,
        n_iterations,
        seed,
    )

    deltas = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        score_a = metric_fn(y_true[idx], y_pred_a[idx])
        score_b = metric_fn(y_true[idx], y_pred_b[idx])
        deltas[i] = score_a - score_b

    delta_obs = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)
    ci_low = float(np.quantile(deltas, alpha / 2))
    ci_high = float(np.quantile(deltas, 1 - alpha / 2))
    p_value = float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))

    LOGGER.debug(
        "Bootstrap delta=%.4f CI=[%.4f, %.4f] p=%.4f",
        delta_obs,
        ci_low,
        ci_high,
        p_value,
    )

    return PairedTestResult(
        delta=float(delta_obs),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        method="bootstrap",
        n_samples=n,
        n_iterations=n_iterations,
    )


def paired_permutation_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 2000,
    seed: int = 42,
) -> PairedTestResult:
    """Paired permutation test by swapping predictions per sample."""
    if y_true.shape != y_pred_a.shape or y_true.shape != y_pred_b.shape:
        raise ValueError("Shapes of y_true, y_pred_a, and y_pred_b must match")

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    if n == 0:
        raise ValueError("Empty arrays provided to paired_permutation_test")
    LOGGER.debug(
        "Running paired permutation test (n=%d, iterations=%d, seed=%d)",
        n,
        n_iterations,
        seed,
    )

    delta_obs = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)
    deltas = np.empty(n_iterations, dtype=float)

    for i in range(n_iterations):
        mask = rng.integers(0, 2, size=n).astype(bool)
        perm_a = y_pred_a.copy()
        perm_b = y_pred_b.copy()
        perm_a[mask] = y_pred_b[mask]
        perm_b[mask] = y_pred_a[mask]
        deltas[i] = metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b)

    p_value = float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))

    LOGGER.debug("Permutation delta=%.4f p=%.4f", delta_obs, p_value)

    return PairedTestResult(
        delta=float(delta_obs),
        ci_low=float(np.quantile(deltas, 0.025)),
        ci_high=float(np.quantile(deltas, 0.975)),
        p_value=p_value,
        method="permutation",
        n_samples=n,
        n_iterations=n_iterations,
    )
