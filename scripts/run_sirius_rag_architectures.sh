#!/bin/bash -l
#SBATCH --job-name=ragarch
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
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_rag_architectures.py"
RESULTS_DIR="$PROJECT_DIR/results/rag_architectures"
MODEL_SLUG="deberta-v3-large"

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

infer_run_artifact_prefix() {
  local cfg_path="$1"
  local seed="$2"
  local base_name cfg_tag context_type mode_tag

  base_name="$(basename "$cfg_path" .yaml)"
  cfg_tag="${base_name#deberta-v3-large_}"
  context_type="${cfg_tag%%_*}"

  if [[ "$cfg_tag" == *crossattn_rag ]]; then
    mode_tag="cross_attention_rag"
  elif [[ "$cfg_tag" == *late_rag ]]; then
    mode_tag="late_rag"
  else
    echo "Unsupported config tag: $cfg_tag" >&2
    return 1
  fi

  printf 'deberta_%s_%s_seed%s_%s' \
    "$context_type" "$mode_tag" "$seed" "$MODEL_SLUG"
}

#for seed in 42 7 1701; do
for seed in 7; do
  for cfg in \
    "$PROJECT_DIR/configs/deberta-v3-large_doc_late_rag.yaml"; do
    run_prefix="$(infer_run_artifact_prefix "$cfg" "$seed")"
    metrics_path="$RESULTS_DIR/logs/${run_prefix}_test_metrics.json"
    resume_path="$RESULTS_DIR/checkpoints/${run_prefix}_last.pt"

    if [[ -f "$metrics_path" ]]; then
      echo "Skipping completed run: cfg=$cfg seed=$seed"
      continue
    fi

    if [[ -f "$resume_path" ]]; then
      echo "Resuming interrupted run: cfg=$cfg seed=$seed from $resume_path"
    else
      echo "Starting fresh run: cfg=$cfg seed=$seed"
    fi

    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
      --config "$cfg" \
      --seed "$seed" \
      --run-name "$run_prefix" \
      --resume "$resume_path" \
      --eval
  done
done
