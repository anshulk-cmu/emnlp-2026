#!/bin/bash
# =============================================================================
# Stage 2a — Centroid Fourier helix fit (discover-then-fit, Whittle null).
#
# Granularity: one array task per (model, task) = 6 tasks. Each task runs
# all 3 modes × 2 variants (lda_a + ccsvd) × 5 layers = 30 worker invocations
# sequentially on its assigned A6000. Worker iterates all eligible concepts.
# Per-cell runtime ~0.5s on GPU (cupy.fft batched); per array task ~10-15 min.
# Wall: 2 days safety margin.
# =============================================================================

#SBATCH --job-name=emnlp_stage2a
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2a-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2a-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
$PYTHON -c "import numpy, scipy, sklearn, cupy, pandas; print(f'numpy={numpy.__version__} scipy={scipy.__version__} cupy={cupy.__version__}')"

for MODE in off answer norm; do
    for VARIANT in lda_a ccsvd; do
        echo "[$(date)] === Stage 2a: model=$MODEL task=$TASK mode=$MODE variant=$VARIANT ==="
        $PYTHON stage2a_fourier_helix.py \
            --config "$CONFIG" \
            --model "$MODEL" \
            --task "$TASK" \
            --mode "$MODE" \
            --variant "$VARIANT" \
            --all-layers
    done
done

echo "[$(date)] done task=$SLURM_ARRAY_TASK_ID model=$MODEL task=$TASK"
