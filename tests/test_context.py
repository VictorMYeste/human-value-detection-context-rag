from value_context_rag.data.context import (
    build_doc_context,
    build_sentence_context,
    build_window_context,
)


def test_build_sentence_context():
    doc = ["a", "b", "c"]
    assert build_sentence_context(doc, 1) == "b"


def test_build_window_context_deberta():
    doc = ["s0", "s1", "s2", "s3"]
    out = build_window_context(doc, 1, n_prev=1, n_next=1, marker_style="deberta")
    assert "<TGT>" in out and "</TGT>" in out
    assert "s0" in out and "s2" in out


def test_build_doc_context_gemma():
    doc = ["first", "second"]
    out = build_doc_context(doc, 0, marker_style="gemma")
    assert "<<<TARGET>>>" in out
    assert "1:" in out and "2:" in out
