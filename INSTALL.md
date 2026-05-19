# Installation

This file only covers environment setup.
For experiment execution and final analysis workflow, see `README.md`.

## 1) Prerequisites

- Python `>= 3.11`
- `pip`
- Optional: Conda/Miniforge
- Optional (GPU local runs): NVIDIA driver + CUDA-compatible PyTorch wheel

## 2) Local install (recommended: venv)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you need GPU PyTorch locally, reinstall `torch` from the proper CUDA index for your machine.
Example (CUDA 12.8):

```bash
python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

## 3) Local install (Conda alternative)

```bash
conda env create -f environment.yml
conda activate value-context-rag
python -m pip install -e .
```

## 4) Sirius cluster install

Use the bootstrap job (this creates/recreates `.venv` on cluster filesystem):

```bash
sbatch --export=ALL,FORCE_REBUILD=1,BASE_PYTHON=$HOME/miniforge3/envs/py311/bin/python scripts/bootstrap_slurm_venv.sh
```

Successful bootstrap creates:

```text
.venv/.bootstrap_complete
```

## 5) Hugging Face token (optional, recommended)

Option A:

```bash
export HF_TOKEN=your_token_here
```

Option B:

```bash
mkdir -p ~/.config/huggingface
printf '%s' "your_token_here" > ~/.config/huggingface/token
chmod 600 ~/.config/huggingface/token
```

## 6) Verify installation

```bash
source .venv/bin/activate
python -V
python -c "import torch; print('torch', torch.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
```
