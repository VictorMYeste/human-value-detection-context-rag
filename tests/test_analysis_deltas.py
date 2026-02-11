import pandas as pd

from value_context_rag.eval.analysis import compute_deltas


def test_compute_deltas_basic():
    df = pd.DataFrame(
        [
            {"model": "m", "context": "sentence", "rag": False, "value": "v", "f1": 0.5},
            {"model": "m", "context": "doc", "rag": False, "value": "v", "f1": 0.7},
            {"model": "m", "context": "doc", "rag": True, "value": "v", "f1": 0.8},
        ]
    )
    deltas = compute_deltas(df)
    assert not deltas.empty
