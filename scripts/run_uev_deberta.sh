#!/bin/bash -l
#SBATCH --job-name=deberta
#SBATCH --partition=grupo_pro
#SBATCH --nodes=1
#SBATCH --nodelist=eevalcachcpro04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/hpc/34045staff/output/human-value-detection-context-rag/%x-%j.out
#SBATCH --error=/home/hpc/34045staff/output/human-value-detection-context-rag/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.yeste@universidadeuropea.es

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hpc/34045staff/human-value-detection-context-rag}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_deberta.py"

cd "$PROJECT_DIR"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python in venv: $PYTHON_BIN" >&2
  echo "Create or sync the UEV venv at: $VENV_DIR" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

PY_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
NVIDIA_LIB_BASE="$VENV_DIR/lib/python${PY_VER}/site-packages/nvidia"
if [ -d "$NVIDIA_LIB_BASE" ]; then
  for d in "$NVIDIA_LIB_BASE"/*/lib; do
    [ -d "$d" ] || continue
    export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
fi

"$PYTHON_BIN" - <<'PY'
import torch

print("python/torch CUDA check")
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this node.")
print("gpu", torch.cuda.get_device_name(0))
PY

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN

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

# Optional overrides:
#   sbatch --export=ALL,DEBERTA_SEEDS="42 7 1701" scripts/run_uev_deberta.sh
#   sbatch --export=ALL,DEBERTA_CONFIGS="/path/to/cfg1.yaml /path/to/cfg2.yaml" scripts/run_uev_deberta.sh
read -r -a SEEDS <<< "${DEBERTA_SEEDS:-42}"
if [ -n "${DEBERTA_CONFIGS:-}" ]; then
  read -r -a CONFIGS <<< "$DEBERTA_CONFIGS"
else
  CONFIGS=(
    "$PROJECT_DIR/configs/deberta-v3-large_doc_rag.yaml"
  )
fi

for seed in "${SEEDS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    artifact_prefix="$(infer_artifact_prefix "$cfg" "$seed")"
    metrics_path="$PROJECT_DIR/results/logs/${artifact_prefix}_test_metrics.json"
    resume_path="$PROJECT_DIR/results/checkpoints/${artifact_prefix}_best_last.pt"

    if [[ -f "$metrics_path" ]]; then
      echo "Skipping completed run: cfg=$cfg seed=$seed"
      continue
    fi

    train_args=("$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$cfg" --seed "$seed" --eval)
    if [[ -f "$resume_path" ]]; then
      echo "Resuming interrupted run: cfg=$cfg seed=$seed from $resume_path"
      train_args+=(--resume "$resume_path")
    else
      echo "Starting fresh run: cfg=$cfg seed=$seed"
    fi

    "${train_args[@]}"
  done
done
