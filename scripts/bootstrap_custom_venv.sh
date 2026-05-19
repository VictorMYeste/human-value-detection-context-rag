#!/bin/bash -l
#SBATCH --job-name=pysetup
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.out
#SBATCH --error=/lustre/scratch/%u/human-value-detection-context-rag/logs/%x-%j.err
#SBATCH --hint=nomultithread

set -euo pipefail

source /home/vicyesmo@upvnet.upv.es/human-value-detection-context-rag/.venv/bin/activate
python -V
python -c "import torch; print(torch.__version__)"