#!/bin/bash -l
#SBATCH --job-name=llama70b
#SBATCH --partition=grupo_pro
#SBATCH --nodes=1 
#SBATCH --nodelist=eevalcachcpro04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=125G
#SBATCH --time=168:00:00
#SBATCH --output=/home/hpc/34045staff/output/human-value-detection-context-rag/%x-%j.out
#SBATCH --error=/home/hpc/34045staff/output/human-value-detection-context-rag/%x-%j.err
#SBATCH --hint=nomultithread

set -euo pipefail

PROJECT_DIR="/home/hpc/34045staff/human-value-detection-context-rag"
RUN_SCRIPT="$PROJECT_DIR/scripts/run_llama.py"
MODEL_SLUG="Llama-3.3-70B-Instruct"

cd "$PROJECT_DIR"

source "$PROJECT_DIR"/.venv/bin/activate

PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
for d in .venv/lib/python${PY_VER}/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export BNB_CUDA_VERSION=128

python - <<'PY'
import inspect, bitsandbytes as bnb
from bitsandbytes.nn import Int8Params
print("bnb", bnb.__version__, bnb.__file__)
print("has _is_hf_initialized:", "_is_hf_initialized" in inspect.signature(Int8Params.__new__).parameters)
PY

python - <<'PY'
import inspect, torch
import bitsandbytes as bnb
from bitsandbytes.nn import Int8Params
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("bnb:", bnb.__version__)
print("gpu:", torch.cuda.get_device_name(0), "cc:", torch.cuda.get_device_capability(0))
print("has _is_hf_initialized:", "_is_hf_initialized" in inspect.signature(Int8Params.__new__).parameters)
PY

python -m bitsandbytes
python -m pip check

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
NVIDIA_LIB_BASE="$PROJECT_DIR/.venv/lib/python${PY_VER}/site-packages/nvidia"
if [ -d "$NVIDIA_LIB_BASE" ]; then
  for d in "$NVIDIA_LIB_BASE"/*/lib; do
    [ -d "$d" ] || continue
    export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
fi

# Force bitsandbytes to use CUDA 12.8 backend with this venv/torch stack.
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-128}"

# Llama 3.3 70B needs a modern GPU class; fail fast if scheduler lands on legacy cards.
python3 - <<'PY'
import inspect
from importlib.metadata import version
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this node.")

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"Detected GPU: {name} (cc={major}.{minor})")
if major < 12:
    raise SystemExit(
        f"Incompatible GPU for this run: {name} (cc={major}.{minor}). "
        "Need a Blackwell-class GPU (cc >= 12.0, e.g., RTX 5090)."
    )

# Fail fast on known int8 stack mismatch that crashes during accelerate dispatch:
# TypeError: Int8Params.__new__() got an unexpected keyword argument '_is_hf_initialized'
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Int8Params

    sig = inspect.signature(Int8Params.__new__)
    has_hf_kwarg = "_is_hf_initialized" in sig.parameters
except Exception as exc:
    raise SystemExit(
        f"bitsandbytes import/signature check failed: {exc}\n"
        "If this mentions libnvJitLink.so.13, your CUDA-13 runtime libs are missing "
        "from LD_LIBRARY_PATH or not installed in the venv.\n"
        "Reinstall a recent stack, e.g.:\n"
        "python -m pip install -U --no-cache-dir --force-reinstall "
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128\n"
        "python -m pip install -U --no-cache-dir --force-reinstall "
        "bitsandbytes accelerate transformers"
    )

print("Package versions:")
for pkg in ("torch", "bitsandbytes", "accelerate", "transformers"):
    try:
        print(f"  {pkg}={version(pkg)}")
    except Exception:
        pass
print("  bitsandbytes_path=", getattr(bnb, "__file__", "unknown"))
print("  BNB_CUDA_VERSION=", __import__("os").environ.get("BNB_CUDA_VERSION"))

if not has_hf_kwarg:
    raise SystemExit(
        "Incompatible bitsandbytes for current transformers/accelerate: "
        "Int8Params.__new__ lacks '_is_hf_initialized'.\n"
        "Reinstall with:\n"
        "python -m pip install -U --no-cache-dir --force-reinstall "
        "bitsandbytes accelerate transformers"
    )
PY

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN

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

  python3 "$RUN_SCRIPT" --config "$cfg" --split test --eval
done
