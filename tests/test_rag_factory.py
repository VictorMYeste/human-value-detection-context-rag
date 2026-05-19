import pytest

torch = pytest.importorskip("torch")

from value_context_rag.models import rag_factory


class DummyTokenizer:
    def __init__(self, name: str = "doc"):
        self.name = name


class DummyLateModel(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels

    def forward(
        self, doc_input_ids, doc_attention_mask, kb_input_ids, kb_attention_mask
    ):
        batch = doc_input_ids.size(0)
        return torch.zeros((batch, self.num_labels))


class DummyCrossModel(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels

    def forward(
        self, doc_input_ids, doc_attention_mask, kb_input_ids, kb_attention_mask
    ):
        batch = doc_input_ids.size(0)
        return torch.zeros((batch, self.num_labels))


def _base_config():
    return {
        "model": {"name": "microsoft/deberta-v3-base"},
        "context": {"type": "doc"},
        "rag": {"enabled": True, "mode": "late"},
    }


def test_build_late_returns_late_fusion_model(monkeypatch):
    def _mock_build(base_model_name, kb_model_name, num_labels):
        return (
            DummyLateModel(num_labels),
            DummyTokenizer("doc"),
            DummyTokenizer("kb"),
        )

    monkeypatch.setattr(rag_factory, "build_late_fusion_model", _mock_build)

    cfg = _base_config()
    cfg["rag"] = {"enabled": True, "mode": "late", "kb_encoder_name": "kb"}

    model, toks = rag_factory.build_rag_model(cfg, num_labels=5)

    assert isinstance(model, DummyLateModel)
    assert toks["doc"].name == "doc"
    assert toks["kb"].name == "kb"

    doc_ids = torch.ones((2, 4), dtype=torch.long)
    doc_mask = torch.ones((2, 4), dtype=torch.long)
    kb_ids = torch.ones((2, 3, 4), dtype=torch.long)
    kb_mask = torch.ones((2, 3, 4), dtype=torch.long)
    logits = model(
        doc_input_ids=doc_ids,
        doc_attention_mask=doc_mask,
        kb_input_ids=kb_ids,
        kb_attention_mask=kb_mask,
    )
    assert logits.shape == (2, 5)


def test_build_cross_attention_returns_cross_attention_model(monkeypatch):
    def _mock_build(base_model_name, num_labels, num_cross_layers=1):
        return (
            DummyCrossModel(num_labels),
            DummyTokenizer("doc"),
            DummyTokenizer("kb"),
        )

    monkeypatch.setattr(rag_factory, "build_cross_attention_model", _mock_build)

    cfg = _base_config()
    cfg["rag"] = {"enabled": True, "mode": "cross_attention", "num_cross_layers": 1}

    model, toks = rag_factory.build_rag_model(cfg, num_labels=11)

    assert isinstance(model, DummyCrossModel)
    assert toks["doc"].name == "doc"
    assert toks["kb"].name == "kb"

    doc_ids = torch.ones((1, 4), dtype=torch.long)
    doc_mask = torch.ones((1, 4), dtype=torch.long)
    kb_ids = torch.ones((1, 2, 4), dtype=torch.long)
    kb_mask = torch.ones((1, 2, 4), dtype=torch.long)
    logits = model(
        doc_input_ids=doc_ids,
        doc_attention_mask=doc_mask,
        kb_input_ids=kb_ids,
        kb_attention_mask=kb_mask,
    )
    assert logits.shape == (1, 11)
