#!/bin/bash -l
#SBATCH --job-name=llama70b
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.out
#SBATCH --error=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vicyesmo@upv.es

set -euo pipefail

PROJECT_DIR="$HOME/human-value-detection-context-rag"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
RUN_SCRIPT="$PROJECT_DIR/scripts/run_llama.py"
MODEL_SLUG="Llama-3.3-70B-Instruct"

cd "$PROJECT_DIR"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python in venv: $PYTHON_BIN" >&2
  echo "Run once: sbatch scripts/bootstrap_slurm_venv.sh" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/.bootstrap_complete" ]; then
  echo "Venv not marked as bootstrapped: $VENV_DIR/.bootstrap_complete missing" >&2
  echo "Run: sbatch --export=ALL,FORCE_REBUILD=1,BASE_PYTHON=$HOME/miniforge3/envs/py311/bin/python scripts/bootstrap_slurm_venv.sh" >&2
  exit 1
fi

PY_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
NVIDIA_LIB_BASE="$VENV_DIR/lib/python${PY_VER}/site-packages/nvidia"
if [ -d "$NVIDIA_LIB_BASE" ]; then
  for d in "$NVIDIA_LIB_BASE"/*/lib; do
    [ -d "$d" ] || continue
    export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
fi

"$PYTHON_BIN" -V
if ! "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__)" >/dev/null 2>&1; then
  echo "Torch import failed in venv: $VENV_DIR" >&2
  echo "Run bootstrap again to complete/install dependencies." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

infer_artifact_prefix() {
  local cfg_path="$1"
  local base_name cfg_tag context_type run_tag

  base_name="$(basename "$cfg_path" .yaml)"
  cfg_tag="${base_name#llama-3.3-70b-instruct_}"
  context_type="${cfg_tag%_rag}"
  if [[ "$cfg_tag" == *_rag ]]; then
    run_tag="rag"
  else
    run_tag="no_rag"
  fi

  printf 'llama_%s_%s_%s' "$context_type" "$run_tag" "$MODEL_SLUG"
}

for cfg in \
  "$PROJECT_DIR/configs/llama-3.3-70b-instruct_sentence.yaml" \
  "$PROJECT_DIR/configs/llama-3.3-70b-instruct_sentence_rag.yaml" \
  "$PROJECT_DIR/configs/llama-3.3-70b-instruct_window.yaml" \
  "$PROJECT_DIR/configs/llama-3.3-70b-instruct_window_rag.yaml" \
  "$PROJECT_DIR/configs/llama-3.3-70b-instruct_doc.yaml" \
  "$PROJECT_DIR/configs/llama-3.3-70b-instruct_doc_rag.yaml"; do
  artifact_prefix="$(infer_artifact_prefix "$cfg")"
  metrics_path="$PROJECT_DIR/results/logs/${artifact_prefix}_test_metrics.json"
  pred_path="$PROJECT_DIR/results/predictions/${artifact_prefix}_test.jsonl"

  if [[ -f "$metrics_path" ]]; then
    echo "Skipping completed run: cfg=$cfg"
    continue
  fi

  if [[ -f "$pred_path" ]]; then
    echo "Resuming interrupted inference: cfg=$cfg from $pred_path"
  else
    echo "Starting fresh inference: cfg=$cfg"
  fi

  "$PYTHON_BIN" "$RUN_SCRIPT" --config "$cfg" --split test --eval
done
