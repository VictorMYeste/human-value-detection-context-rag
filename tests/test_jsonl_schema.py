import json
import tempfile
from pathlib import Path

import numpy as np

from value_context_rag.eval.analysis import kb_values_per_prediction


def test_kb_values_per_prediction_schema():
    records = [
        {
            "text_id": "t1",
            "sent_id": "s1",
            "pred_labels": ["A"],
            "kb_values": ["V1", "V2"],
        }
    ]
    df = kb_values_per_prediction(records)
    assert set(df.columns) == {"text_id", "sent_id", "pred_label", "kb_value"}
    assert len(df) == 2


def test_prediction_jsonl_roundtrip():
    record = {
        "text_id": "t1",
        "sent_id": "s1",
        "gold_labels": ["A"],
        "pred_labels": ["B"],
        "kb_chunk_ids": ["c1"],
        "kb_values": ["V1"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pred.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8").strip())
        assert loaded["text_id"] == "t1"
        assert loaded["kb_values"] == ["V1"]
