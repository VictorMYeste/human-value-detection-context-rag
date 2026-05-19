# human-value-detection-context-rag

Reproducible experiments for human value detection with:
- `DeBERTa-v3-base`
- `DeBERTa-v3-large`
- `Gemma 3 12B`
- `Qwen2.5-72B-Instruct`
- `Mistral-Large-Instruct-2407`
- `Llama-3.3-70B-Instruct`

This README is the canonical runbook for training/inference and final paper result extraction.

For artifact-level documentation, including dataset access constraints, KB
documentation, checkpoint release policy, and model-card-style notes, see
`artifact_documentation.md`.

## 1) Data layout

Place ValuesML files under:

```text
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

KB artifacts live in `data/kb/`:
- `kb_chunks.jsonl`
- `kb_index.faiss`
- `kb_embeddings.npy`

If needed, rebuild KB index:

```bash
python scripts/build_kb.py --kb_output_dir data/kb --overwrite
```

## 2) Environment

Local setup: see `INSTALL.md`.

Sirius cluster setup (recommended):

```bash
sbatch --export=ALL,FORCE_REBUILD=1,BASE_PYTHON=$HOME/miniforge3/envs/py311/bin/python scripts/bootstrap_slurm_venv.sh
```

The Slurm bootstrap script recreates `.venv`, installs requirements, installs `torch` from CUDA 12.8 wheels, and creates:

```text
.venv/.bootstrap_complete
```

All Sirius launchers check this marker before running.

Optional HF auth:
- put token in `$HOME/.config/huggingface/token`, or
- export `HF_TOKEN` before `sbatch`.

## 3) Local single-run commands

### DeBERTa training

```bash
python scripts/train_deberta.py --config configs/deberta_sentence.yaml --seed 42 --eval
python scripts/train_deberta.py --config configs/deberta-v3-large_sentence.yaml --seed 42 --eval
```

### LLM inference

```bash
python scripts/run_gemma.py --config configs/gemma_sentence.yaml --split test --eval
python scripts/run_qwen.py --config configs/qwen2.5-72b-instruct_sentence.yaml --split test --eval
python scripts/run_mistral.py --config configs/mistral-large-instruct-2407_sentence.yaml --split test --eval
python scripts/run_llama.py --config configs/llama-3.3-70b-instruct_sentence.yaml --split test --eval
```

## 4) Sirius Slurm endpoints (UPV)

Run from repository root:

```bash
sbatch scripts/<launcher>.sh
```

Available Sirius launchers:
- `scripts/run_sirius_deberta.sh`
  - DeBERTa-v3-large supervised training grid/sequence (edit config loop inside script).
- `scripts/run_sirius_rag_architectures.sh`
  - DeBERTa-v3-large late-rag / cross-attention doc architectures.
- `scripts/run_sirius_qwen2.5_72b.sh`
  - Qwen2.5-72B-Instruct all context + rag/no-rag configs.
- `scripts/run_sirius_mistral_large_instruct_2407.sh`
  - Mistral-Large-Instruct-2407 all context + rag/no-rag configs.
- `scripts/run_sirius_llama3.3_70b_instruct.sh`
  - Llama-3.3-70B-Instruct all context + rag/no-rag configs.

Notes:
- These scripts are resumable by artifact existence checks (`*_test_metrics.json` and/or predictions JSONL).
- GPU/CPU/memory/time are set per launcher; edit `#SBATCH` headers if needed.

## 5) UEV Slurm endpoints

UEV launchers:
- `scripts/run_uev_mistral_large_instruct_2407.sh`
- `scripts/run_uev_llama3.3_70b_instruct.sh`

Use:

```bash
sbatch scripts/run_uev_mistral_large_instruct_2407.sh
sbatch scripts/run_uev_llama3.3_70b_instruct.sh
```

These scripts include UEV-specific resource requests and runtime checks.

## 6) Config inventory

### DeBERTa-v3-base
- `configs/deberta_sentence.yaml`
- `configs/deberta_sentence_rag.yaml`
- `configs/deberta_window.yaml`
- `configs/deberta_window_rag.yaml`
- `configs/deberta_doc.yaml`
- `configs/deberta_doc_rag.yaml`
- `configs/deberta_doc_late_rag.yaml`
- `configs/deberta_doc_crossattn_rag.yaml`

### DeBERTa-v3-large
- `configs/deberta-v3-large_sentence.yaml`
- `configs/deberta-v3-large_sentence_rag.yaml`
- `configs/deberta-v3-large_window.yaml`
- `configs/deberta-v3-large_window_rag.yaml`
- `configs/deberta-v3-large_doc.yaml`
- `configs/deberta-v3-large_doc_rag.yaml`
- `configs/deberta-v3-large_doc_late_rag.yaml`
- `configs/deberta-v3-large_doc_crossattn_rag.yaml`

### Gemma / Qwen / Mistral / Llama
- `configs/gemma_{sentence,window,doc}.yaml`
- `configs/gemma_{sentence,window,doc}_rag.yaml`
- `configs/qwen2.5-72b-instruct_{sentence,window,doc}.yaml`
- `configs/qwen2.5-72b-instruct_{sentence,window,doc}_rag.yaml`
- `configs/mistral-large-instruct-2407_{sentence,window,doc}.yaml`
- `configs/mistral-large-instruct-2407_{sentence,window,doc}_rag.yaml`
- `configs/llama-3.3-70b-instruct_{sentence,window,doc}.yaml`
- `configs/llama-3.3-70b-instruct_{sentence,window,doc}_rag.yaml`

## 7) Final paper extraction pipeline (canonical)

Run these two scripts after all training/inference outputs are available:

```bash
python scripts/build_project_final_results.py
python scripts/qual_examples.py --bundle --split test --max_examples 50
```

Outputs are written to `results/analysis/final/`.

`build_project_final_results.py` writes:
- `main_results.csv`
- `main_results_agg.csv`
- `per_value_results.csv`
- `per_value_deltas.csv`
- `rag_architectures.csv`
- `llm_results.csv`
- `deberta_base_vs_large.csv`
- `significance_tests.csv`
- `prediction_changes.csv`

`qual_examples.py --bundle` writes:
- `qual_examples_deberta_context_rag.jsonl`
- `qual_examples_llm_rag.jsonl`
- `qual_examples_failure_cases.jsonl`

## 8) Legacy analysis scripts

These are optional utilities, not required for final paper export:
- `scripts/analyze_results.py`
- `scripts/analyze_rag_architectures.py`

Use them for ad-hoc checks only.

## 9) Reproducibility checklist

Before final runs:
- Confirm `.venv/.bootstrap_complete` exists on the target cluster.
- Confirm `HF_TOKEN` is available (env var or token file).
- Confirm configs used are committed/tracked.
- Keep Slurm `.out/.err` logs under scratch for auditability.

After final runs:
- Archive `results/logs`, `results/predictions`, `results/rag_architectures`.
- Archive `results/analysis/final`.

## 10) Full run order (copy-paste)

Run in this order from the repo root.

### 10.1 Sirius setup

```bash
cd ~/human-value-detection-context-rag
mkdir -p /lustre/scratch/$USER/human-value-detection-context-rag/logs
sbatch --export=ALL,FORCE_REBUILD=1,BASE_PYTHON=$HOME/miniforge3/envs/py311/bin/python scripts/bootstrap_slurm_venv.sh
```

### 10.2 Supervised baselines and architectures (Sirius)

```bash
cd ~/human-value-detection-context-rag
sbatch scripts/run_sirius_deberta.sh
sbatch scripts/run_sirius_rag_architectures.sh
```

### 10.3 LLM runs on Sirius

```bash
cd ~/human-value-detection-context-rag
sbatch scripts/run_sirius_qwen2.5_72b.sh
sbatch scripts/run_sirius_mistral_large_instruct_2407.sh
sbatch scripts/run_sirius_llama3.3_70b_instruct.sh
```

### 10.4 Optional UEV runs

```bash
cd /home/hpc/34045staff/human-value-detection-context-rag
sbatch scripts/run_uev_mistral_large_instruct_2407.sh
sbatch scripts/run_uev_llama3.3_70b_instruct.sh
```

### 10.5 Monitor jobs

```bash
squeue --me
```

### 10.6 Build final paper tables and qualitative files

```bash
cd ~/human-value-detection-context-rag
python scripts/build_project_final_results.py
python scripts/qual_examples.py --bundle --split test --max_examples 50
```

### 10.7 Verify final artifacts

```bash
cd ~/human-value-detection-context-rag
ls -lah results/analysis/final
```

Expected final directory includes:
- `main_results.csv`
- `main_results_agg.csv`
- `per_value_results.csv`
- `per_value_deltas.csv`
- `rag_architectures.csv`
- `llm_results.csv`
- `deberta_base_vs_large.csv`
- `significance_tests.csv`
- `prediction_changes.csv`
- `qual_examples_deberta_context_rag.jsonl`
- `qual_examples_llm_rag.jsonl`
- `qual_examples_failure_cases.jsonl`

### 10.8 Archive for paper submission

```bash
cd ~/human-value-detection-context-rag
tar -czf results_bundle_$(date +%Y%m%d).tar.gz \
  results/analysis/final \
  results/logs \
  results/predictions \
  results/rag_architectures
```
