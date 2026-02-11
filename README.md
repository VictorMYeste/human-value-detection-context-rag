# human-value-detection-context-rag

## Quickstart

1. Install dependencies and setup your environment: see `INSTALL.md`.
2. Prepare the KB index:
   ```bash
   python scripts/build_kb.py --kb_output_dir data/kb --overwrite
   ```
3. Run training or inference:
   ```bash
   python scripts/train_deberta.py --config configs/deberta_sentence.yaml --seed 42 --eval_test
   ```

## Overview

Code for sentence-level Schwartz human value detection in political texts, comparing DeBERTa-v3-base and Gemma 3 12B across context windows and RAG with explicit moral knowledge (Schwartz theory + annotation guidelines).

## Data Sources

### Main dataset: Values in News and Political Manifestos

This repository uses the “Values in News and Political Manifestos” corpus released within the EU project ValuesML – Unravelling Expressed Values in the Media for Informed Policy Making.

- [Project page](https://knowledge4policy.ec.europa.eu/projects-activities/valuesml-unravelling-expressed-values-media-informed-policy-making_en)
- [Annotation guidelines (PDF)](https://data.europa.eu/doi/10.2760/7398)
- [Dataset on Zenodo](https://zenodo.org/records/13283288)

We do not redistribute the dataset due to licensing. To run experiments, place the English sentence/label TSVs at:

```
data/raw/
    training/
        sentences.tsv
        labels.tsv
    validation/
        sentences.tsv
        labels.tsv
    test/
        sentences.tsv
        labels.tsv
```

### Knowledge base for RAG

`data/kb/kb_chunks.jsonl` is a manually curated KB of short, paraphrased chunks distilled from Schwartz theory papers and the ValuesML guidelines. Each line is:

```json
{
  "id": "definitions-0",
  "source": "definitions" | "guidelines" | "theory",
  "values": "Universalism: concern, Universalism: tolerance",
  "text": "Short paraphrased explanation or guideline..."
}
```

Chunking principles and counts are documented in `data/kb/DATA.md`.
