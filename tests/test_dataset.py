import pandas as pd

from value_context_rag.data.dataset import _collapse_attained_constrained


def test_collapse_attained_constrained():
    df = pd.DataFrame(
        {
            "Text-ID": ["t1"],
            "Sentence-ID": ["s1"],
            "Self-direction: thought attained": ["0"],
            "Self-direction: thought constrained": ["0.5"],
            "Stimulation attained": ["0"],
            "Stimulation constrained": ["0"],
        }
    )
    collapsed = _collapse_attained_constrained(df, debug=False)
    assert list(collapsed.columns) == [
        "Text-ID",
        "Sentence-ID",
        "Self-direction: thought",
        "Stimulation",
    ]
    assert collapsed.loc[0, "Self-direction: thought"] == 1.0
    assert collapsed.loc[0, "Stimulation"] == 0.0
