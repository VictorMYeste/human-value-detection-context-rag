#!/bin/bash -l
#SBATCH --job-name=pygpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.out
#SBATCH --error=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vicyesmo@upv.es

set -euo pipefail

PROJECT_DIR="$HOME/human-value-detection-context-rag"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
# TRAIN_SCRIPT="$PROJECT_DIR/scripts/eval_deberta.py"
# CONFIG_PATH="$PROJECT_DIR/configs/deberta-v3-large_sentence.yaml"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_deberta.py"
# OUTPUT_PATH="$PROJECT_DIR/results/analysis/grid_deberta_hparams.csv"

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
# export PROJECT_DIR
# export SCRATCH_DIR=/lustre/scratch/$USER/human-value-detection-context-rag

infer_artifact_prefix() {
  local cfg_path="$1"
  local seed="$2"
  local base_name cfg_tag context_type run_tag

  base_name="$(basename "$cfg_path" .yaml)"
  cfg_tag="${base_name#deberta-v3-large_}"
  context_type="${cfg_tag%_rag}"
  if [[ "$cfg_tag" == *_rag ]]; then
    run_tag="rag"
  else
    run_tag="no_rag"
  fi

  printf 'deberta_%s_%s_seed%s_deberta-v3-large' \
    "$context_type" "$run_tag" "$seed"
}

for seed in 42; do
  for cfg in \
    "$PROJECT_DIR/configs/deberta-v3-large_doc_rag.yaml"; do
      artifact_prefix="$(infer_artifact_prefix "$cfg" "$seed")"
      metrics_path="$PROJECT_DIR/results/logs/${artifact_prefix}_test_metrics.json"
      resume_path="$PROJECT_DIR/results/checkpoints/${artifact_prefix}_best_last.pt"

      if [[ -f "$metrics_path" ]]; then
        echo "Skipping completed run: cfg=$cfg seed=$seed"
        continue
      fi

      if [[ -f "$resume_path" ]]; then
        echo "Resuming interrupted run: cfg=$cfg seed=$seed from $resume_path"
      else
        echo "Starting fresh run: cfg=$cfg seed=$seed"
      fi

      "$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$cfg" --seed "$seed" --eval
  done
done

# "$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$CONFIG_PATH" --split validation --tune_threshold
