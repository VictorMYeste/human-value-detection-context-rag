# More Context, Larger Models, or Moral Knowledge?

Code and experiment configurations for:

> **More Context, Larger Models, or Moral Knowledge? A Systematic Study of Schwartz Value Detection in Political Texts**  
> Víctor Yeste and Paolo Rosso, 2026  
> arXiv: [2605.22641](https://arxiv.org/abs/2605.22641)

This repository supports the paper's experiments on **sentence-level Schwartz
value detection** with document context, retrieval-augmented moral knowledge,
supervised DeBERTa encoders, and zero-shot instruction-tuned LLMs.

If you use this code, configurations, released model checkpoints, or derived
results, please cite the paper. See [Citation](#citation).

---

## Contents

- [Overview](#overview)
- [What Is Included](#what-is-included)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Knowledge Base](#knowledge-base)
- [Quickstart](#quickstart)
- [Reproducing the Paper](#reproducing-the-paper)
- [Configuration Inventory](#configuration-inventory)
- [Cluster Launchers](#cluster-launchers)
- [Final Result Extraction](#final-result-extraction)
- [Released Model Checkpoint](#released-model-checkpoint)
- [Artifact and Data Release Policy](#artifact-and-data-release-policy)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

The paper studies when additional information helps or hurts sentence-level
classification of the **19 refined Schwartz human values** in political texts.
The main experimental factors are:

- **Input context:** target sentence only, local sentence window, or full document.
- **Retrieved moral knowledge:** no retrieval vs. retrieval from a curated
  Schwartz-value knowledge base.
- **Model family:** supervised DeBERTa-v3 encoders vs. zero-shot LLMs.
- **Model scale:** DeBERTa-v3-base vs. DeBERTa-v3-large, and LLMs from 12B to
  123B parameters.
- **RAG fusion strategy for encoders:** early fusion, late fusion, and
  cross-attention.
- **Per-value analysis:** which values benefit most from context, retrieval, and
  model family.

The core task is **multi-label classification** over the 19 values. The original
attained/constrained annotations are collapsed into one binary presence label per
value:

```text
value is active = attained OR constrained
```

---

## What Is Included

This repository contains:

- Data loading and label-collapsing utilities.
- Context builders for sentence, window, and document inputs.
- A curated moral knowledge base and FAISS retrieval pipeline.
- Supervised DeBERTa-v3-base and DeBERTa-v3-large training/evaluation code.
- Encoder RAG variants: early fusion, late fusion, and cross-attention.
- Zero-shot LLM inference wrappers for Gemma, Qwen, and Mistral.
- Optional Llama launchers/configurations for extended experiments not reported
  in the main paper.
- Slurm launchers for the Sirius and UEV cluster environments.
- Final result aggregation and qualitative-example extraction scripts.
- Artifact documentation for responsible release and reproducibility.

This repository does **not** redistribute the benchmark texts. See
[Data](#data) and [Artifact and Data Release Policy](#artifact-and-data-release-policy).

---

## Repository Structure

```text
human-value-detection-context-rag/
  configs/                      # YAML configs for all model/context/RAG conditions
  data/
    kb/                         # Curated moral KB and retrieval index
    raw/                        # Expected location for restricted dataset files
  scripts/
    train_deberta.py            # Supervised DeBERTa training/evaluation
    train_rag_architectures.py  # Late/cross-attention encoder RAG training
    run_gemma.py                # Gemma zero-shot inference
    run_qwen.py                 # Qwen zero-shot inference
    run_mistral.py              # Mistral zero-shot inference
    run_llama.py                # Optional Llama inference
    build_kb.py                 # Build FAISS KB index
    build_project_final_results.py
    qual_examples.py
    run_sirius_*.sh             # Sirius Slurm launchers
    run_uev_*.sh                # UEV Slurm launchers
  src/value_context_rag/        # Python package
  tests/                        # Unit tests
  results/analysis/final/       # Final derived tables, if regenerated locally
  INSTALL.md                    # Environment setup
  artifact_documentation.md     # Dataset/model/artifact documentation
```

---

## Installation

For full setup details, see [`INSTALL.md`](INSTALL.md).

Minimal local setup:

```bash
git clone https://github.com/VictorMYeste/human-value-detection-context-rag.git
cd human-value-detection-context-rag

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

Recommended Python version: `>=3.11`.

For large LLM inference, use a GPU environment with enough VRAM and a compatible
PyTorch/CUDA installation. The paper experiments used NVIDIA H100 80GB GPU nodes
for the largest models.

Optional Hugging Face authentication is recommended for faster model downloads:

```bash
export HF_TOKEN=your_token_here
```

or place the token at:

```text
~/.config/huggingface/token
```

---

## Data

The experiments use the official ValuesML / Touché ValueEval-style benchmark
files for human value detection. The dataset is distributed under a restricted
Data Usage Agreement and cannot be redistributed in this repository.

Expected local layout:

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

The scripts assume that each split provides sentence identifiers, document
identifiers, sentence order, sentence text, and the value-label files. Users must
obtain the dataset separately from the official organizers and follow the
original access conditions.

The paper uses sentence-level examples grouped by:

- `text_id`: document identifier.
- `sent_id`: sentence identifier/order within the document.
- 19 collapsed binary value labels.

---

## Knowledge Base

Retrieval uses a small curated knowledge base with compact descriptions of
Schwartz values, annotation guidance, and value contrasts.

Expected KB files:

```text
data/kb/
  kb_chunks.jsonl
  kb_index.faiss
  kb_embeddings.npy
```

Rebuild the FAISS index with:

```bash
python scripts/build_kb.py --kb_output_dir data/kb --overwrite
```

The main paper uses fixed retrieval with `top_k=2`, so comparisons between
early, late, and cross-attention RAG isolate the fusion mechanism rather than
changing the retrieval system.

---

## Quickstart

### Smoke Test

Run a small DeBERTa training pass without persisting full outputs:

```bash
python scripts/train_deberta.py \
  --config configs/deberta_sentence.yaml \
  --seed 42 \
  --max_samples 16 \
  --dry_run
```

Run a small LLM inference pass:

```bash
python scripts/run_gemma.py \
  --config configs/gemma_sentence.yaml \
  --split test \
  --max_samples 5 \
  --dry_run
```

### Train One DeBERTa Model

```bash
python scripts/train_deberta.py \
  --config configs/deberta_doc_rag.yaml \
  --seed 42 \
  --eval
```

### Run One Zero-Shot LLM Condition

```bash
python scripts/run_gemma.py \
  --config configs/gemma_doc_rag.yaml \
  --split test \
  --eval
```

Equivalent launchers exist for Qwen and Mistral:

```bash
python scripts/run_qwen.py \
  --config configs/qwen2.5-72b-instruct_doc_rag.yaml \
  --split test \
  --eval

python scripts/run_mistral.py \
  --config configs/mistral-large-instruct-2407_doc_rag.yaml \
  --split test \
  --eval
```

---

## Reproducing the Paper

The paper compares the following reported model families:

- `microsoft/deberta-v3-base`
- `microsoft/deberta-v3-large`
- `google/gemma-3-12b-it`
- `Qwen/Qwen2.5-72B-Instruct`
- `mistralai/Mistral-Large-Instruct-2407`

Llama configuration files are included for optional follow-up experiments, but
Llama is not part of the main paper comparison unless explicitly added by the
user.

### 1. Prepare Environment and KB

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/build_kb.py --kb_output_dir data/kb --overwrite
```

### 2. Run Supervised DeBERTa Conditions

For a single condition:

```bash
python scripts/train_deberta.py \
  --config configs/deberta_sentence.yaml \
  --seed 42 \
  --eval
```

Repeat over:

- Contexts: `sentence`, `window`, `doc`.
- RAG: no-RAG and early-fusion RAG.
- Seeds: the reported aggregate DeBERTa results use multiple seeds.
- Model scale: DeBERTa-v3-base and DeBERTa-v3-large configs.

### 3. Run Encoder RAG Architecture Variants

Late-fusion and cross-attention RAG variants are trained with:

```bash
python scripts/train_rag_architectures.py \
  --config configs/deberta_doc_late_rag.yaml \
  --seed 42 \
  --run-name deberta_doc_late_rag_seed42_deberta-v3-base \
  --eval

python scripts/train_rag_architectures.py \
  --config configs/deberta_doc_crossattn_rag.yaml \
  --seed 42 \
  --run-name deberta_doc_cross_attention_rag_seed42_deberta-v3-base \
  --eval
```

Use the corresponding `deberta-v3-large_*` configs for the large encoder.

### 4. Run Zero-Shot LLM Conditions

For Gemma:

```bash
python scripts/run_gemma.py \
  --config configs/gemma_sentence.yaml \
  --split test \
  --eval
```

For Qwen:

```bash
python scripts/run_qwen.py \
  --config configs/qwen2.5-72b-instruct_sentence.yaml \
  --split test \
  --eval
```

For Mistral:

```bash
python scripts/run_mistral.py \
  --config configs/mistral-large-instruct-2407_sentence.yaml \
  --split test \
  --eval
```

Repeat for `sentence`, `window`, `doc`, and their `_rag` counterparts.

### 5. Build Final Paper Tables

After all predictions and metrics are available:

```bash
python scripts/build_project_final_results.py
python scripts/qual_examples.py --bundle --split test --max_examples 50
```

Outputs are written to:

```text
results/analysis/final/
```

---

## Configuration Inventory

### DeBERTa-v3-base

```text
configs/deberta_sentence.yaml
configs/deberta_sentence_rag.yaml
configs/deberta_window.yaml
configs/deberta_window_rag.yaml
configs/deberta_doc.yaml
configs/deberta_doc_rag.yaml
configs/deberta_doc_late_rag.yaml
configs/deberta_doc_crossattn_rag.yaml
```

### DeBERTa-v3-large

```text
configs/deberta-v3-large_sentence.yaml
configs/deberta-v3-large_sentence_rag.yaml
configs/deberta-v3-large_window.yaml
configs/deberta-v3-large_window_rag.yaml
configs/deberta-v3-large_doc.yaml
configs/deberta-v3-large_doc_rag.yaml
configs/deberta-v3-large_doc_late_rag.yaml
configs/deberta-v3-large_doc_crossattn_rag.yaml
```

### Zero-Shot LLMs

```text
configs/gemma_{sentence,window,doc}.yaml
configs/gemma_{sentence,window,doc}_rag.yaml

configs/qwen2.5-72b-instruct_{sentence,window,doc}.yaml
configs/qwen2.5-72b-instruct_{sentence,window,doc}_rag.yaml

configs/mistral-large-instruct-2407_{sentence,window,doc}.yaml
configs/mistral-large-instruct-2407_{sentence,window,doc}_rag.yaml
```

### Optional Extra Llama Configs

```text
configs/llama-3.3-70b-instruct_{sentence,window,doc}.yaml
configs/llama-3.3-70b-instruct_{sentence,window,doc}_rag.yaml
```

These are included for extension experiments, not as part of the main paper.

---

## Cluster Launchers

The repository includes Slurm launchers used during the project. They are
cluster-specific templates and should be inspected before submission, because
the config loops may be edited for resumed or partial runs.

### Sirius Cluster

Bootstrap the environment once:

```bash
sbatch --export=ALL,FORCE_REBUILD=1,BASE_PYTHON=$HOME/miniforge3/envs/py311/bin/python \
  scripts/bootstrap_slurm_venv.sh
```

Run from the repository root:

```bash
sbatch scripts/run_sirius_deberta.sh
sbatch scripts/run_sirius_rag_architectures.sh
sbatch scripts/run_sirius_qwen2.5_72b.sh
sbatch scripts/run_sirius_mistral_large_instruct_2407.sh
```

Optional extra Llama run:

```bash
sbatch scripts/run_sirius_llama3.3_70b_instruct.sh
```

### UEV Cluster

```bash
sbatch scripts/run_uev_mistral_large_instruct_2407.sh
sbatch scripts/run_uev_llama3.3_70b_instruct.sh
```

Notes:

- The launchers check `.venv/.bootstrap_complete`.
- Large LLM launchers require Hugging Face access to the corresponding models.
- Resumability is implemented through prediction/metrics file existence checks.
- Logs are written to the scratch paths defined in the `#SBATCH` headers.

---

## Final Result Extraction

The canonical final-analysis command is:

```bash
python scripts/build_project_final_results.py
python scripts/qual_examples.py --bundle --split test --max_examples 50
```

`build_project_final_results.py` writes:

```text
results/analysis/final/main_results.csv
results/analysis/final/main_results_agg.csv
results/analysis/final/per_value_results.csv
results/analysis/final/per_value_deltas.csv
results/analysis/final/rag_architectures.csv
results/analysis/final/llm_results.csv
results/analysis/final/deberta_base_vs_large.csv
results/analysis/final/significance_tests.csv
results/analysis/final/prediction_changes.csv
```

`qual_examples.py --bundle` writes:

```text
results/analysis/final/qual_examples_deberta_context_rag.jsonl
results/analysis/final/qual_examples_llm_rag.jsonl
results/analysis/final/qual_examples_failure_cases.jsonl
```

Important: qualitative example files may contain restricted benchmark text.
Do not redistribute them publicly unless this is permitted by the dataset usage
agreement.

Legacy/ad-hoc scripts:

```text
scripts/analyze_results.py
scripts/analyze_rag_architectures.py
```

These are not required for the final paper tables.

---

## Released Model Checkpoint

The main released fine-tuned checkpoint is:

- [`VictorYeste/value-context-rag-deberta-v3-base-doc-rag`](https://huggingface.co/VictorYeste/value-context-rag-deberta-v3-base-doc-rag)

This model corresponds to a DeBERTa-v3-base document-context RAG condition. It is
released for research use with a model card documenting task, labels, training
setup, data restrictions, intended use, and limitations.

The repository does not redistribute third-party LLM weights.

---

## Artifact and Data Release Policy

For detailed artifact documentation, see
[`artifact_documentation.md`](artifact_documentation.md).

Publicly releasable artifacts:

- Code.
- Configurations.
- Prompt templates.
- Curated KB chunks and metadata.
- Aggregate metrics and derived analysis tables.
- Fine-tuned model bundles where permitted.

Not publicly redistributed:

- Raw benchmark texts.
- Full document contexts from the restricted dataset.
- Public qualitative examples containing verbatim restricted text.
- Third-party LLM weights.

Users must obtain the dataset and gated models separately and comply with their
licenses and access terms.

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{yeste2026contextlargermodelsmoral,
      title={More Context, Larger Models, or Moral Knowledge? A Systematic Study of Schwartz Value Detection in Political Texts}, 
      author={Víctor Yeste and Paolo Rosso},
      year={2026},
      eprint={2605.22641},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.22641}, 
}
```

---

## License

The code in this repository is released under the **Apache License 2.0**.
See [`LICENSE`](LICENSE) for details.

This license does not grant any rights over the underlying benchmark data or
third-party model weights. Please respect the corresponding dataset, model, and
software licenses.

---

## Contact

For questions, open a GitHub issue or contact:

Víctor Yeste — vicyesmo [at] upv [dot] es
