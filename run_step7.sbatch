#!/bin/bash
# =============================================================================
# Step 7 — Residual hunting across all (model, task, mode, layer) cells.
#
# Granularity: one array task per (model, task, mode) = 18 tasks total.
# Each task runs all 5 layers sequentially on its assigned A6000.
# No concurrency cap — cluster will schedule up to the available GPU pool.
# Wall: 2 days safety; expected ~10 min per task.
# =============================================================================

#SBATCH --job-name=emnlp_step7
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-step7-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-step7-%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-17
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

# 18 cells: 3 models × 2 tasks × 3 modes. Sorted (model, task, mode).
MODEL_TASK_MODE=(
    "gpt-j-6b:addition:off"
    "gpt-j-6b:addition:answer"
    "gpt-j-6b:addition:norm"
    "gpt-j-6b:multiplication:off"
    "gpt-j-6b:multiplication:answer"
    "gpt-j-6b:multiplication:norm"
    "llama-3.1-8b:addition:off"
    "llama-3.1-8b:addition:answer"
    "llama-3.1-8b:addition:norm"
    "llama-3.1-8b:multiplication:off"
    "llama-3.1-8b:multiplication:answer"
    "llama-3.1-8b:multiplication:norm"
    "pythia-6.9b:addition:off"
    "pythia-6.9b:addition:answer"
    "pythia-6.9b:addition:norm"
    "pythia-6.9b:multiplication:off"
    "pythia-6.9b:multiplication:answer"
    "pythia-6.9b:multiplication:norm"
)
TRIPLE="${MODEL_TASK_MODE[$SLURM_ARRAY_TASK_ID]}"
MODEL="${TRIPLE%%:*}"
REST="${TRIPLE#*:}"
TASK="${REST%:*}"
MODE="${REST#*:}"
CONFIG=/home/anshulk/emnlp2026/config.yaml

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID model=$MODEL task=$TASK mode=$MODE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"
$PYTHON --version
$PYTHON -c "import numpy, scipy, sklearn, cupy, pandas; print(f'numpy={numpy.__version__} scipy={scipy.__version__} cupy={cupy.__version__}')"

echo "[$(date)] === Step 7: model=$MODEL task=$TASK mode=$MODE ==="
$PYTHON residual_hunting.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --task "$TASK" \
    --mode "$MODE" \
    --all-layers

echo "[$(date)] done task=$SLURM_ARRAY_TASK_ID model=$MODEL task=$TASK mode=$MODE"
