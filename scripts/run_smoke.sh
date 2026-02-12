#!/usr/bin/env bash
set -euo pipefail

for seed in 42 7 1701; do
  for cfg in \
    configs/deberta_sentence.yaml \
    configs/deberta_sentence_rag.yaml \
    configs/deberta_window.yaml \
    configs/deberta_window_rag.yaml \
    configs/deberta_doc.yaml \
    configs/deberta_doc_rag.yaml; do
      python scripts/train_deberta.py --config "$cfg" --seed "$seed" --max_samples 10 --dry_run --eval
  done
done

for cfg in \
  configs/gemma_sentence.yaml \
  configs/gemma_sentence_rag.yaml \
  configs/gemma_window.yaml \
  configs/gemma_window_rag.yaml \
  configs/gemma_doc.yaml \
  configs/gemma_doc_rag.yaml; do
    python scripts/run_gemma.py --config "$cfg" --split test --max_samples 5 --eval --dry_run
done
