#!/bin/bash
#SBATCH --job-name=pygpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.out
#SBATCH --error=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.err
#SBATCH --hint=nomultithread

set -euo pipefail

PROJECT_DIR="$HOME/human-value-detection-context-rag"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_deberta.py"
CONFIG_PATH="$PROJECT_DIR/configs/deberta-v3-large_sentence.yaml"

cd "$PROJECT_DIR"

if [ ! -e "$PYTHON_BIN" ]; then
  echo "Python path does not exist: $PYTHON_BIN" >&2
  echo "HOME=$HOME PROJECT_DIR=$PROJECT_DIR" >&2
  ls -ld "$PROJECT_DIR" "$PROJECT_DIR/.venv" "$PROJECT_DIR/.venv/bin" 2>/dev/null || true
  ls -l "$PROJECT_DIR/.venv/bin/python"* 2>/dev/null || true
  exit 1
fi

if ! "$PYTHON_BIN" -V >/dev/null 2>&1; then
  echo "Python exists but is not runnable on this node: $PYTHON_BIN" >&2
  echo "Likely cause: .venv/bin/python points to a missing system interpreter on compute nodes." >&2
  ls -l "$PROJECT_DIR/.venv/bin/python"* 2>/dev/null || true
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PROJECT_DIR
export SCRATCH_DIR=/lustre/scratch/$USER/human-value-detection-context-rag

"$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$CONFIG_PATH" --seed "42" --eval --max_samples 10
