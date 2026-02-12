#!/usr/bin/env bash
set -euo pipefail

for cfg in \
  configs/gemma_sentence.yaml \
  configs/gemma_sentence_rag.yaml \
  configs/gemma_window.yaml \
  configs/gemma_window_rag.yaml \
  configs/gemma_doc.yaml \
  configs/gemma_doc_rag.yaml; do
    python scripts/run_gemma.py --config "$cfg" --split test --eval
done
