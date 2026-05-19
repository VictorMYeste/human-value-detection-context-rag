#!/usr/bin/env bash
set -euo pipefail

SEEDS=(42 7 1701)

for seed in "${SEEDS[@]}"; do
  echo "=== Seed ${seed} | late_rag ==="
  python scripts/train_rag_architectures.py \
    --config configs/deberta_doc_late_rag.yaml \
    --seed "${seed}" \
    --eval

  echo "=== Seed ${seed} | cross_attention_rag ==="
  python scripts/train_rag_architectures.py \
    --config configs/deberta_doc_crossattn_rag.yaml \
    --seed "${seed}" \
    --eval
done
