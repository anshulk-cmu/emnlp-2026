#!/bin/bash
# =============================================================================
# Stage 2c BSMI-R — GPT-J 6B on the general partition (no preemption).
#
# Pipeline: Bayesian Shape Manifold Inference with Refusal. Reference:
# docs/gplvm.md. The new worker is label-collapsed (operates on per-value
# means + within-label scatter, not the full N x N point cloud) so the heavy
# Cholesky cost from the prior pipeline is gone. CPU-bound parts (PH, dim
# estimation, permutations) dominate; we keep one A6000 per array task as a
# placeholder for the GPU-aware path in case we add a GPU-backed evidence
# audit later.
#
# 4 array tasks; each task processes its hashed cell stripe.
# =============================================================================

#SBATCH --job-name=emnlp_stage2c_gptj
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-gptj-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-gptj-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

ARRAY_SIZE=4
ARRAY_IDX=$SLURM_ARRAY_TASK_ID
CONFIG=/home/anshulk/emnlp2026/config.yaml
MODEL=gpt-j-6b

echo "[$(date)] array_task=$ARRAY_IDX of $ARRAY_SIZE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"
$PYTHON --version
$PYTHON -c "import numpy, pandas, sklearn; print(f'numpy={numpy.__version__} pandas={pandas.__version__} sklearn={sklearn.__version__}')"
$PYTHON -c "import ripser; print(f'ripser={ripser.__version__}')" 2>&1 | head -1 || \
  echo "[warn] ripser not installed; PH module will use the gudhi fallback or report status=no_ph_lib"

# Single worker per array task processes its hashed cell stripe.
$PYTHON stage2c_gplvm.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --sweep all \
    --array-task "$ARRAY_IDX" \
    --array-size "$ARRAY_SIZE"

echo "[$(date)] done array_task=$ARRAY_IDX"
