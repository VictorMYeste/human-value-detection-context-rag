import numpy as np

from value_context_rag.kb.retriever import Retriever


def test_retriever_filter_by_value_and_source():
    chunks = [
        {"id": "a", "source": "theory", "values": ["A"], "text": "t1"},
        {"id": "b", "source": "guidelines", "values": ["B"], "text": "t2"},
        {"id": "c", "source": "theory", "values": ["A", "B"], "text": "t3"},
    ]

    class DummyIndex:
        def search(self, vec, top_k):
            return None, np.array([[0, 1, 2]])

    def embed(_text: str) -> np.ndarray:
        return np.zeros((1, 3), dtype="float32")

    retriever = Retriever(chunks=chunks, index=DummyIndex(), embed_query=embed)

    hits = retriever.retrieve("q", top_k=3, value="A")
    assert [h["id"] for h in hits] == ["a", "c"]

    hits = retriever.retrieve("q", top_k=3, source="guidelines")
    assert [h["id"] for h in hits] == ["b"]
