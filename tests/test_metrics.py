import numpy as np

from value_context_rag.eval.metrics import compute_global_metrics, macro_f1_from_arrays


def test_macro_f1_from_arrays():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [1, 0]])
    macro_f1 = macro_f1_from_arrays(y_true, y_pred)
    assert 0.0 <= macro_f1 <= 1.0


def test_compute_global_metrics_keys():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [1, 0]])
    metrics = compute_global_metrics(y_true, y_pred)
    assert "micro_f1" in metrics
    assert "macro_f1" in metrics
