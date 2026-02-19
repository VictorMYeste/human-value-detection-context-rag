import random

from value_context_rag.data.kb_noise import (
    drop_top_chunk,
    inject_offtopic_noise,
    limit_k,
)


def test_drop_top_chunk_reduces_list_length():
    chunks = [{"id": 1}, {"id": 2}, {"id": 3}]
    out = drop_top_chunk(chunks, drop_prob=1.0)
    assert len(out) == 2
    assert out[0]["id"] == 2


def test_inject_offtopic_noise_changes_some_ids():
    random.seed(0)
    chunks = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    global_kb = [{"id": 100}, {"id": 200}, {"id": 300}]
    out = inject_offtopic_noise(chunks, global_kb, noise_ratio=0.5)
    assert len(out) == len(chunks)
    # at least one id should come from global_kb
    assert any(item["id"] in {100, 200, 300} for item in out)


def test_limit_k_truncates_list():
    chunks = [{"id": 1}, {"id": 2}, {"id": 3}]
    out = limit_k(chunks, k=2)
    assert len(out) == 2
    assert out[-1]["id"] == 2
