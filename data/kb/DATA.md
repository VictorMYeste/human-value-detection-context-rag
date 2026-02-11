# Knowledge Base (KB) Notes

This KB is a manually curated set of short, paraphrased chunks intended for retrieval-augmented generation (RAG). The file `data/kb/kb_chunks.jsonl` contains one JSON object per line:

```json
{
  "id": "definitions-0",
  "source": "definitions" | "guidelines" | "theory",
  "values": "Universalism: concern, Universalism: tolerance",
  "text": "Short paraphrased explanation or guideline..."
}
```

## Chunking principles

- Chunks are kept short (1–3 sentences) to improve retrieval precision.
- Text is paraphrased to avoid reproducing copyrighted sources.
- Each chunk is tagged with a `source`:
  - `definitions`: one chunk per refined Schwartz value (core definitions).
  - `guidelines`: operational annotation guidelines and disambiguation cues.
  - `theory`: higher-level theoretical structure and value contrasts.
- `values` can be null (no specific value), a single value, or multiple comma‑separated values.

## Current counts (example)

- Chunk count: 58
- Chunks per source:

```json
{"definition": 19, "theory": 14, "guidelines": 25}
```

- Chunks per value:

```json
{"Self-direction: thought": 4, "Self-direction: action": 3, "Stimulation": 4, "Hedonism": 4, "Achievement": 5, "Power: dominance": 5, "Power: resources": 4, "Face": 3, "Security: personal": 5, "Security: societal": 7, "Tradition": 6, "Conformity: rules": 6, "Conformity: interpersonal": 3, "Humility": 3, "Benevolence: caring": 6, "Benevolence: dependability": 4, "Universalism: concern": 7, "Universalism: nature": 4, "Universalism: tolerance": 3}
```

Update these counts when you change the KB.
