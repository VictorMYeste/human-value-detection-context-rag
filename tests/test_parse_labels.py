from value_context_rag.llm.inference import parse_labels


def test_parse_labels_basic():
    label_names = ["A", "B", "C"]
    raw = "A, c, D"
    parsed = parse_labels(raw, label_names)
    assert parsed == ["A", "C"]


def test_parse_labels_none():
    label_names = ["A", "B"]
    assert parse_labels("NONE", label_names) == []
