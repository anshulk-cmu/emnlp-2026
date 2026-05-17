#!/bin/bash
# =============================================================================
# Stage 2b — Spread-aware Mahalanobis test (d_SW).
#
# Inherits Stage 2a's discovery (period, axes, verdict) and runs the spread-aware
# test on eligible cells. Granularity: one array task per (model, task) = 6 tasks.
# Each task runs all 3 modes × 2 variants × 5 layers = 30 worker invocations
# sequentially. Per-cell runtime ~5-15s (bootstrap + null are the cost drivers).
# Per array task ~30-90 min. Wall: 2 days safety margin.
# =============================================================================

#SBATCH --job-name=emnlp_stage2b
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2b-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2b-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
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
echo "[$(date)] PYTHON=$PYTHON"
$PYTHON --version
$PYTHON -c "import numpy, scipy, sklearn, pandas; print(f'numpy={numpy.__version__} scipy={scipy.__version__} sklearn={sklearn.__version__}')"

for MODE in off answer norm; do
    for VARIANT in lda_a ccsvd; do
        echo "[$(date)] === Stage 2b: model=$MODEL task=$TASK mode=$MODE variant=$VARIANT ==="
        $PYTHON stage2b_dsw_spread_aware.py \
            --config "$CONFIG" \
            --model "$MODEL" \
            --task "$TASK" \
            --mode "$MODE" \
            --variant "$VARIANT" \
            --all-layers
    done
done

echo "[$(date)] done task=$SLURM_ARRAY_TASK_ID model=$MODEL task=$TASK"
