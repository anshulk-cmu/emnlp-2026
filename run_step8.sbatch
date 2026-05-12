#!/bin/bash
# =============================================================================
# Step 8 — Principal angles between concept subspaces.
#
# Per array task: one (model, task) pair → 3 modes × 5 layers = 15 cells.
# 6 array tasks (3 models × 2 tasks), max 3 concurrent.
# Wall: 2 days safety; expected ~15 min per task.
# =============================================================================

#SBATCH --job-name=emnlp_step8
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-step8-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-step8-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5%3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

MODEL_TASK_PAIRS=(
    "gpt-j-6b:addition"
    "gpt-j-6b:multiplication"
    "llama-3.1-8b:addition"
    "llama-3.1-8b:multiplication"
    "pythia-6.9b:addition"
    "pythia-6.9b:multiplication"
)
PAIR="${MODEL_TASK_PAIRS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${PAIR%:*}"
TASK="${PAIR#*:}"
CONFIG=/home/anshulk/emnlp2026/config.yaml

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID model=$MODEL task=$TASK host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python

for MODE in off answer norm; do
    echo "[$(date)] === Step 8: model=$MODEL task=$TASK mode=$MODE ==="
    $PYTHON principal_angles.py \
        --config "$CONFIG" \
        --model "$MODEL" \
        --task "$TASK" \
        --mode "$MODE" \
        --all-layers
done

echo "[$(date)] done task=$SLURM_ARRAY_TASK_ID model=$MODEL task=$TASK"
