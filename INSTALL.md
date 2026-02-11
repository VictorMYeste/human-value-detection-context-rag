# Installation & Usage

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate value-context-rag
python -m ipykernel install --user --name value-context-rag
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

## Docker (artifact reproducibility)

### CPU image

```bash
docker build -t value-context-rag-cpu -f Dockerfile.cpu .
docker run --rm -it value-context-rag-cpu python -V
```

### GPU image

```bash
docker build -t value-context-rag-gpu -f Dockerfile.gpu .
docker run --rm -it --gpus all value-context-rag-gpu python -c "import torch; print(torch.cuda.is_available())"
```

## Knowledge Base (build FAISS index)

```bash
python scripts/build_kb.py --kb_output_dir data/kb --overwrite
```

## Training / Inference

### DeBERTa (all contexts, RAG/no RAG, seeds 42/7/1701)

```bash
for seed in 42 7 1701; do
  for cfg in \
    configs/deberta_sentence.yaml \
    configs/deberta_sentence_rag.yaml \
    configs/deberta_window.yaml \
    configs/deberta_window_rag.yaml \
    configs/deberta_doc.yaml \
    configs/deberta_doc_rag.yaml; do
      python scripts/train_deberta.py --config "$cfg" --seed "$seed" --eval_test
  done
done
```

### Gemma (all contexts, RAG/no RAG)

```bash
for cfg in \
  configs/gemma_sentence.yaml \
  configs/gemma_sentence_rag.yaml \
  configs/gemma_window.yaml \
  configs/gemma_window_rag.yaml \
  configs/gemma_doc.yaml \
  configs/gemma_doc_rag.yaml; do
    python scripts/run_gemma.py --config "$cfg" --split test --eval
done
```

## Analysis

Aggregate metrics, per-value tables, deltas, and significance tests:

```bash
python scripts/analyze_results.py
```

Select a canonical DeBERTa seed automatically (best validation macro-F1) or specify it explicitly:

```bash
python scripts/analyze_results.py --canonical_seed 42
```

## Utilities

### KB value → predicted label summary

```bash
python scripts/kb_value_summary.py --predictions results/predictions/deberta_doc_rag_seed42.jsonl
```

### Optuna learning-rate search (DeBERTa, sentence, no RAG)

```bash
python scripts/optuna_deberta_lr.py --trials 10 --max_samples 200
```

## Quality checks

```bash
make format
make lint
make test
```
