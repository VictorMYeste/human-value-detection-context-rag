from value_context_rag.kb.retriever import _normalize_values


def test_normalize_values_none():
    assert _normalize_values(None) == []


def test_normalize_values_list():
    assert _normalize_values(["A", " B "]) == ["A", "B"]


def test_normalize_values_string():
    assert _normalize_values("A, B, C") == ["A", "B", "C"]
