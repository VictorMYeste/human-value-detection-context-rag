#!/usr/bin/env bash
set -euo pipefail

for seed in 1701; do
  for cfg in \
    configs/deberta_doc.yaml \
    configs/deberta_doc_rag.yaml; do
      python scripts/train_deberta.py --config "$cfg" --seed "$seed" --eval
  done
done
