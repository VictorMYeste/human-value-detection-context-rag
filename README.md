# human-value-detection-context-rag

## Description

Code for a paper on sentence-level Schwartz human value detection in political texts, comparing DeBERTa-v3-base and Gemma 3 12B across different document context windows and RAG with explicit moral knowledge (Schwartz theory + annotation guidelines).

Here’s some markdown you can drop straight into your README.md to document both the KB and the main dataset.

## Data sources

### Main dataset: Values in News and Political Manifestos

This repository uses the “Values in News and Political Manifestos” corpus released within the EU project ValuesML – Unravelling Expressed Values in the Media for Informed Policy Making.

- [Project page (background & references)](https://knowledge4policy.ec.europa.eu/projects-activities/valuesml-unravelling-expressed-values-media-informed-policy-making_en)
- [Official annotation guidelines (PDF)](https://data.europa.eu/doi/10.2760/7398)
- [Dataset on Zenodo (used in this repo)](https://zenodo.org/records/13283288)

In short, the dataset consists of:

- Domain: news articles and political manifestos. ￼
- Languages: nine languages (the experiments in this repo use only the English, machine-translated portion). ￼
- Unit of analysis: sentence-level annotations, grouped by document ID.
- Labels: 19 refined Schwartz basic values, each annotated with an attainment dimension (value (partially) attained vs (partially) constrained). ￼

We do not redistribute the original dataset here due to licensing. To run the experiments, please:

- Request / download the data from the official Zenodo record (and/or ValueEval’24 organisers if applicable), accepting their usage conditions. ￼
- Place the English sentence and label TSV files under:

```
data/raw/
    train/
        sentences.tsv
        labels.tsv
    validation/
        sentences.tsv
        labels.tsv
    test/
        sentences.tsv
        labels.tsv
```

The code expects the TSV format used in the original release (sentence-level labels with Text-ID and Sentence-ID).

### Knowledge base for RAG: data/kb/kb_chunks.jsonl

The file data/kb/kb_chunks.jsonl contains a manually curated knowledge base used for retrieval-augmented experiments. It is not a direct copy of any source; instead, it consists of short paraphrased chunks distilled from:

- Schwartz theory papers
- Schwartz, S. H. (2012). An Overview of the Schwartz Theory of Basic Values. Online Readings in Psychology and Culture, 2(1).  ￼
- Schwartz, S. H. et al. (2012). Refining the Theory of Basic Individual Values. Journal of Personality and Social Psychology, 103(4), 663–688.  ￼
- Official ValuesML annotation guidelines for “Values in News and Political Manifestos”. ￼

Each line in kb_chunks.jsonl is a JSON object of the form:

```json
{
  "id": "definitions-0",
  "source": "definitions" | "guidelines" | "theory",
  "values": "Universalism: concern, Universalism: tolerance",
  "text": "Short paraphrased explanation or guideline..."
}
```

Conceptually, the KB is organised into three layers:

1.	Layer 1 – Core definitions (source = "definitions")
    - One chunk per value (19 chunks total).
    - Concise, paraphrased descriptions of each refined Schwartz value, combining information from the theoretical papers and the guidelines (1–3 sentences per chunk).
2.	Layer 2 – Operational guidelines (source = "guidelines")
    - Value-specific chunks that describe when to annotate a value and when not to (typical cues, common confusions).
    - A few additional value-independent chunks covering general annotation principles (multiple values in a sentence, negation, hypotheticals, etc.). ￼
3.	Layer 3 – Theoretical structure & contrasts (source = "theory")
    - Chunks that summarise compatibilities and conflicts between values along the Schwartz motivational continuum (e.g. Self-direction / Stimulation vs Security / Conformity; Benevolence vs Universalism; Power vs Universalism). ￼
    - These are designed to help models disambiguate between nearby vs opposing values when interpreting political sentences.

All chunks are intentionally short and paraphrased to stay within fair-use limits and to work well as retrieval units in RAG. For authoritative definitions and full theoretical context, please refer to the original publications and EU guidelines cited above.

- Chunk count: 58
- Chunks per source:
```json
{'definition': 19, 'theory': 14, 'guidelines': 25}
```
- Chunks per value:
```json
{'Self-direction: thought': 4, 'Self-direction: action': 3, 'Stimulation': 4, 'Hedonism': 4, 'Achievement': 5, 'Power: dominance': 5, 'Power: resources': 4, 'Face': 3, 'Security: personal': 5, 'Security: societal': 7, 'Tradition': 6, 'Conformity: rules': 6, 'Conformity: interpersonal': 3, 'Humility': 3, 'Benevolence: caring': 6, 'Benevolence: dependability': 4, 'Universalism: concern': 7, 'Universalism: nature': 4, 'Universalism: tolerance': 3}
```

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate value-context-rag
python3 -m ipykernel install --user --name value-context-rag
pip install -e .
```

### venv + pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For GPU PyTorch (CUDA 12.2), install the CUDA wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu122
```

### Hugging Face Token (optional)

Set `HF_TOKEN` to avoid rate limits and speed up model downloads:

```bash
export HF_TOKEN=your_token_here
```

You can add this to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`) to persist it.

## Knowledge Base (build FAISS index)

```bash
python3 scripts/build_kb.py --kb_output_dir data/kb --overwrite
```

## Training / Inference Commands

### DeBERTa (all contexts, RAG/no RAG, seeds 42/7/1701, with eval_test)

```bash
for seed in 42 7 1701; do
  for cfg in \
    configs/deberta_sentence.yaml \
    configs/deberta_sentence_rag.yaml \
    configs/deberta_window.yaml \
    configs/deberta_window_rag.yaml \
    configs/deberta_doc.yaml \
    configs/deberta_doc_rag.yaml; do
      python3 scripts/train_deberta.py --config "$cfg" --seed "$seed" --eval_test
  done
done
```

### Gemma (all contexts, RAG/no RAG, with eval)

```bash
for cfg in \
  configs/gemma_sentence.yaml \
  configs/gemma_sentence_rag.yaml \
  configs/gemma_window.yaml \
  configs/gemma_window_rag.yaml \
  configs/gemma_doc.yaml \
  configs/gemma_doc_rag.yaml; do
    python3 scripts/run_gemma.py --config "$cfg" --split test --eval
done
```
