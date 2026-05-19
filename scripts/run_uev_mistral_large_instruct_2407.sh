#!/bin/bash -l
#SBATCH --job-name=mistral123b
#SBATCH --partition=grupo_pro
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=125G
#SBATCH --time=168:00:00
#SBATCH --output=/home/hpc/34045staff/output/human-value-detection-context-rag/%x-%j.out
#SBATCH --error=/home/hpc/34045staff/output/human-value-detection-context-rag/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.yeste@universidadeuropea.es

PROJECT_DIR="/home/hpc/34045staff/human-value-detection-context-rag"
RUN_SCRIPT="$PROJECT_DIR/scripts/run_mistral.py"
MODEL_SLUG="Mistral-Large-Instruct-2407"

cd "$PROJECT_DIR"

source "$PROJECT_DIR"/.venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN

infer_artifact_prefix() {
  local cfg_path="$1"
  local base_name cfg_tag context_type run_tag

  base_name="$(basename "$cfg_path" .yaml)"
  cfg_tag="${base_name#mistral-large-instruct-2407_}"
  context_type="${cfg_tag%_rag}"
  if [[ "$cfg_tag" == *_rag ]]; then
    run_tag="rag"
  else
    run_tag="no_rag"
  fi

  printf 'mistral_%s_%s_%s' "$context_type" "$run_tag" "$MODEL_SLUG"
}

for cfg in \
  "$PROJECT_DIR/configs/mistral-large-instruct-2407_sentence.yaml" \
  "$PROJECT_DIR/configs/mistral-large-instruct-2407_sentence_rag.yaml" \
  "$PROJECT_DIR/configs/mistral-large-instruct-2407_window.yaml" \
  "$PROJECT_DIR/configs/mistral-large-instruct-2407_window_rag.yaml" \
  "$PROJECT_DIR/configs/mistral-large-instruct-2407_doc.yaml" \
  "$PROJECT_DIR/configs/mistral-large-instruct-2407_doc_rag.yaml"; do
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
