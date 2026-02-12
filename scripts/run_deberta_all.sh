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
      python scripts/train_deberta.py --config "$cfg" --seed "$seed" --eval_test
  done
done
