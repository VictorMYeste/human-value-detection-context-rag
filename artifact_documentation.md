# Artifact Documentation

This document summarizes the artifacts associated with the paper experiments on
sentence-level Schwartz value detection with document context, retrieval, and
large language models. It is intended to support reproducibility, responsible
release, and the ACL/EMNLP author checklist recommendation to document data and
model artifacts.

For anonymous review, public URLs may be omitted or replaced by anonymized
supplemental material. For the camera-ready version, this document should be
included in the public GitHub repository linked from the paper.

## 1. Released Artifacts

The repository is intended to release:

- Source code for data loading, context construction, retrieval, training,
  inference, evaluation, and result aggregation.
- Configuration files for all reported model, context, and RAG conditions.
- Prompt templates and model-specific launchers for zero-shot LLM inference.
- The curated moral knowledge base used for retrieval, including chunk metadata.
- Slurm launch scripts for the Sirius and UEV cluster environments.
- Analysis scripts that generate the paper tables and qualitative bundles.
- Aggregate metrics and derived result tables.
- Prediction files where permitted by the dataset usage agreement.
- Fine-tuned DeBERTa checkpoints or Hugging Face model bundles where permitted
  by the base-model and dataset terms.

The repository is not intended to release:

- Raw benchmark texts.
- Full document contexts from the benchmark data.
- Qualitative examples containing verbatim restricted dataset text.
- Weights for third-party instruction-tuned LLMs such as Gemma, Qwen, Mistral,
  or Llama.

## 2. Dataset Documentation

### Dataset Source

The experiments use the official ValuesML/Touché ValueEval-style benchmark
files for human value detection. Users must obtain the dataset from the official
organizers under the original access conditions.

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

### Access and Redistribution

The benchmark data may include third-party copyrighted content and is governed
by a restricted usage agreement. The agreement permits scientific research use
for human value detection but prohibits redistribution or sharing of the dataset
in part or in full.

Because of this restriction:

- Raw text files are not redistributed in this repository.
- Public qualitative examples use sentence identifiers and paraphrased target
  descriptions instead of verbatim target sentences.
- Any derived artifact that would reproduce restricted text should not be
  publicly released.
- Users with authorized dataset access can regenerate all text-dependent
  artifacts by running the released scripts locally.

### Task Definition

The prediction unit is a sentence identified by `text_id` and `sent_id`.
Sentences are grouped into documents by `text_id`, and `sent_id` gives sentence
order within the document.

The task is multi-label classification over the 19 refined Schwartz values:

- Self-direction: thought
- Self-direction: action
- Stimulation
- Hedonism
- Achievement
- Power: dominance
- Power: resources
- Face
- Security: personal
- Security: societal
- Tradition
- Conformity: rules
- Conformity: interpersonal
- Humility
- Benevolence: caring
- Benevolence: dependability
- Universalism: concern
- Universalism: nature
- Universalism: tolerance

The official labels distinguish attained and constrained values. The paper
experiments collapse both variants into value-presence labels, producing one
binary target per value.

### Dataset Statistics Used in the Paper

After collapsing attained/constrained labels:

| Split | Documents | Sentences | Labels per sentence | No-label sentences | Multi-label sentences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 1,603 | 44,758 | 0.58 | 48.5% | 5.9% |
| Validation | 523 | 14,904 | 0.58 | 49.0% | 5.9% |
| Test | 522 | 14,569 | 0.58 | 49.2% | 6.2% |

### Intended Use

The dataset and derived artifacts are intended for research on value detection,
context-sensitive classification, retrieval-augmented classification, and model
comparison under controlled experimental conditions.

### Out-of-Scope Use

The dataset and models should not be used for individual-level profiling,
automated moderation, surveillance, or high-stakes decisions about people,
political speakers, or social groups.

## 3. Knowledge Base Documentation

The retrieval knowledge base is located in `data/kb/`:

```text
data/kb/
  DATA.md
  kb_chunks.jsonl
  kb_embeddings.npy
  kb_index.faiss
```

Each KB chunk is a short, paraphrased description of value definitions,
annotation guidance, or theoretical contrasts. The KB is task-facing: it is used
to provide compact moral knowledge to the classifier or LLM prompt.

Current KB summary:

- Total chunks: 58
- Definition chunks: 19
- Theory chunks: 14
- Guideline chunks: 25
- Retrieval backend: FAISS index over sentence embeddings
- Main retrieval setting: top-k = 2
- Maximum KB budget in model input: 200 tokens by default

Retrieval is held fixed when comparing early, late, and cross-attention RAG
fusion variants. This isolates the fusion mechanism from the retrieval system.

To rebuild the KB index:

```bash
python scripts/build_kb.py --kb_output_dir data/kb --overwrite
```

## 4. Model Artifact Documentation

### Fine-Tuned DeBERTa Models

The supervised encoder experiments use:

- `microsoft/deberta-v3-base`
- `microsoft/deberta-v3-large`

The released checkpoints, where permitted, correspond to fine-tuned sequence
classification models trained on the official training split only. Checkpoints
are selected using validation macro-F1.

Main training settings:

- Multi-label binary classification with sigmoid outputs.
- Primary metric: macro-F1.
- Secondary metric: micro-F1.
- Maximum sequence length: 1024.
- Default gradient accumulation steps: 2.
- Default prediction threshold: 0.18 unless otherwise tuned on validation.
- Seeds: 42, 7, and 13 for reported DeBERTa aggregate results.
- DeBERTa-v3-base default hyperparameters are defined in
  `src/value_context_rag/utils/config.py`.
- DeBERTa-v3-large overrides use learning rate `3e-6`, weight decay `0.1`,
  batch size `16`, and gradient checkpointing.

Input conditions:

- `sentence`: target sentence only.
- `window`: target sentence plus neighboring sentences.
- `doc`: full document context, truncated to the model budget.
- `*_rag`: early-fusion retrieval with KB chunks appended to the input.
- `doc_late_rag`: document RAG with a separate KB representation.
- `doc_crossattn_rag`: document RAG with cross-attention over KB chunks.

Checkpoint release policy:

- Fine-tuned model bundles may be released on Hugging Face if permitted by the
  base model and dataset terms.
- Model cards should state that checkpoints were trained on restricted benchmark
  data and do not contain redistributed raw texts.
- Users must obtain the official dataset separately to reproduce training.

### Zero-Shot LLM Inference Configurations

The paper evaluates zero-shot instruction-tuned LLMs using prompts and
configuration files, not fine-tuning:

- `google/gemma-3-12b-it`
- `Qwen/Qwen2.5-72B-Instruct`
- `mistralai/Mistral-Large-Instruct-2407`

Additional Llama configuration files may be present in the repository for
extended experiments, but Llama is not part of the main paper comparison unless
explicitly reported in the paper tables.

LLM configuration notes:

- Decoding is deterministic by default: temperature `0.0`, top-p `1.0`.
- Maximum generation length is 64 tokens by default.
- LLM prompts ask the model to output a comma-separated list of Schwartz values
  or `NONE`.
- RAG variants insert the retrieved KB chunks into the prompt.
- Large models may use 8-bit quantization and CPU offload depending on hardware.

LLM artifact release policy:

- Third-party model weights are not redistributed.
- The repository releases only prompts, configs, scripts, and derived outputs
  that are permitted under the dataset and model-provider terms.
- Users must follow each model provider's license and access requirements.

## 5. Evaluation and Result Artifacts

The canonical final-result workflow is:

```bash
python scripts/build_project_final_results.py
python scripts/qual_examples.py --bundle --split test --max_examples 50
```

The main outputs are written to `results/analysis/final/`:

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

When sharing these files publicly, inspect them for restricted text. If an
artifact includes raw target sentences or document context, release the script
needed to regenerate it instead of releasing the artifact itself.

## 6. Reproducibility Notes

To reproduce the paper results, users need:

- Authorized access to the official dataset.
- This repository's source code and configuration files.
- The curated KB files or the script to rebuild them.
- Python environment specified by `requirements.txt` and `pyproject.toml`.
- Access to the base models from Hugging Face or the relevant model provider.
- Sufficient GPU resources for the selected model family.

Recommended setup and launch commands are documented in `README.md` and
`INSTALL.md`.

Cluster launchers are provided for:

- Sirius: DeBERTa, RAG architectures, Qwen, Mistral, and Llama launchers.
- UEV: Mistral and Llama launchers.

## 7. Limitations of the Artifacts

- The benchmark is sparse and imbalanced; macro-F1 should be interpreted
  together with per-value results.
- Prediction thresholds are validation-dependent and may not transfer to other
  domains.
- Quantization and hardware-specific loading behavior can affect LLM inference
  reproducibility.
- Zero-shot LLM outputs are sensitive to prompt templates and model-specific
  chat formatting.
- Fine-tuned DeBERTa checkpoints inherit the limitations of the training data,
  including annotation ambiguity and domain specificity.
- The KB reflects a compact operationalization of Schwartz value theory and may
  shape model behavior through its wording.

## 8. Ethical and Responsible Use

These artifacts are intended for transparent research on aggregate patterns in
value expression. They should not be used to infer a person's values, profile
political actors, rank viewpoints, or automate decisions in sensitive settings.

Researchers reusing the artifacts should:

- Respect the dataset usage agreement.
- Respect base-model licenses and provider terms.
- Report whether they use released checkpoints, regenerated checkpoints, or
  only configuration files.
- Document any changes to preprocessing, context construction, retrieval,
  prompts, thresholds, or hardware.
- Inspect qualitative examples and error cases before drawing substantive
  conclusions about value expression.

