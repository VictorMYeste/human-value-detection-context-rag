#!/bin/bash -l
#SBATCH --job-name=pysetup
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.out
#SBATCH --error=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.err
#SBATCH --hint=nomultithread

set -euo pipefail

PROJECT_DIR="$HOME/human-value-detection-context-rag"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniforge3/envs/py311/bin/python}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
VENV_READY_FILE=".bootstrap_complete"

cd "$PROJECT_DIR"

is_py311_plus() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

configure_cuda_lib_path() {
  local py_ver
  py_ver="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.11")"
  local base="$VENV_DIR/lib/python${py_ver}/site-packages/nvidia"
  if [ -d "$base" ]; then
    for d in "$base"/*/lib; do
      [ -d "$d" ] || continue
      export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
  fi
}

echo "Using system python from Slurm environment:"
echo "HOSTNAME=$(hostname)"
echo "PATH=$PATH"
echo "BASE_PYTHON=$BASE_PYTHON"

if [ ! -x "$BASE_PYTHON" ]; then
  echo "BASE_PYTHON does not exist or is not executable: $BASE_PYTHON" >&2
  echo "Pass a valid path with: --export=ALL,BASE_PYTHON=/path/to/python3.11" >&2
  exit 1
fi
if ! is_py311_plus "$BASE_PYTHON"; then
  echo "BASE_PYTHON is too old (need >=3.11): $BASE_PYTHON" >&2
  "$BASE_PYTHON" -V >&2 || true
  exit 1
fi

"$BASE_PYTHON" -V

echo "Recreating venv from scratch: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
  TRASH_DIR="${VENV_DIR}.trash_${SLURM_JOB_ID:-$$}_$(date +%s)"
  echo "Moving old venv to: $TRASH_DIR"
  if ! mv "$VENV_DIR" "$TRASH_DIR"; then
    echo "Could not move old venv; trying direct delete..." >&2
    rm -rf "$VENV_DIR" || true
    if [ -e "$VENV_DIR" ]; then
      echo "Could not remove $VENV_DIR. Stop all jobs using it and retry." >&2
      exit 1
    fi
  else
    # On Lustre, immediate recursive delete may fail with ESTALE.
    # Keep trash directory and clean it later from login node if desired.
    echo "Old venv moved to trash (not deleted now): $TRASH_DIR"
  fi
fi

"$BASE_PYTHON" -m venv "$VENV_DIR"

echo "Installing base tooling..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
echo "Installing project requirements..."
"$PYTHON_BIN" -m pip install -r requirements.txt
echo "Installing PyTorch from $TORCH_INDEX_URL ..."
"$PYTHON_BIN" -m pip install torch --index-url "$TORCH_INDEX_URL"
# Text-only DeBERTa training does not require vision/audio stacks.
# Remove them to avoid torchvision/torch mismatches in Transformers imports.
"$PYTHON_BIN" -m pip uninstall -y torchvision torchaudio >/dev/null 2>&1 || true
echo "Installing local package..."
"$PYTHON_BIN" -m pip install -e .

echo "Bootstrap complete:"
"$PYTHON_BIN" -V
configure_cuda_lib_path
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__)"
touch "$VENV_DIR/$VENV_READY_FILE"
